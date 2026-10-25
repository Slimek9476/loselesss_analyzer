"""Ядро анализа: поиск «кирпичной стены» (brickwall cutoff) в спектре.

Идея метода
-----------
Lossy-кодеки (MP3, AAC, Vorbis и т.п.) физически не передают самые высокие
частоты. Спектр такого файла имеет почти вертикальный обрыв: выше некоторой
частоты энергия падает на десятки дБ и остаётся на уровне шума. Если такой
файл пересохранить во FLAC/WAV, обрыв никуда не денется.

Алгоритм:
1. Переводим спектр в дБ относительно средних частот (1–4 кГц).
2. Сглаживаем спектр, чтобы убрать узкие тональные пики.
3. Ищем резкий обрыв: перепад между полосами до и после кандидата, локальную
   крутизну и разброс уровня на «полке».
4. Если резкого обрыва нет, ищем мягкий срез: устойчивую низкоуровневую полку
   у верхней границы и заметный спад перед ней (SRC, lossy, lowpass).
5. Для найденного среза оцениваем источник (MP3/AAC или апскейл).
6. Если среза нет, проверяем энергию у частоты Найквиста: есть — TRUE
   LOSSLESS; тихо/темно — INCONCLUSIVE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from .dsp import SpectrogramData, SpectrumData
from .formatting import format_khz
from .loader import AudioMeta

Verdict = Literal["lossless", "fake", "inconclusive", "lossy"]

# Пороговые значения детектора (подобраны экспериментально).
_MIN_CUTOFF_HZ = 9_500.0
_WIDE_DROP_DB = 22.0
_EDGE_DROP_DB = 12.0
_MIN_SHARPNESS = 0.55
_MAX_PLATEAU_STD_DB = 5.0
_MIN_PRE_LEVEL_DB = -46.0
_MIN_POST_WIDTH_HZ = 500.0

# Анализ во времени: порог верхней границы кадра и допуски.
_TIME_MIN_NYQUIST_HZ = 15_000.0
_TIME_MIN_REF_DB = -60.0
_TIME_EDGE_REL_DB = -55.0
_TIME_MIN_FRAMES = 5
_TIME_SMOOTH_HZ = 250.0
_TIME_TOLERANCE_HZ = 2_000.0
_TIME_MARGIN_HZ = 1_500.0

_UPSCALE_ORIGINS: tuple[tuple[float, str], ...] = (
    (22_050.0, "44.1 кГц"),
    (24_000.0, "48 кГц"),
    (32_000.0, "64 кГц"),
    (44_100.0, "88.2 кГц"),
    (48_000.0, "96 кГц"),
)


@dataclass
class DetectionResult:
    """Результат детекции обрыва частот и итоговый вердикт."""

    verdict: Verdict
    headline: str
    reason: str
    cutoff_hz: float | None = None
    drop_db: float | None = None
    plateau_level_db: float | None = None
    transition_width_hz: float | None = None
    confidence: float = 0.0
    hf_level_db: float | None = None
    estimated_source: str | None = None
    effective_bits: int | None = None
    time_frames: int = 0
    time_edge_median_hz: float | None = None
    time_edge_p10_hz: float | None = None
    time_edge_p90_hz: float | None = None
    time_consistency: float | None = None
    time_above_fraction: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class _EdgeMetrics:
    cutoff_hz: float
    wide_drop: float
    edge_drop: float
    sharpness: float
    pre_level: float
    post_level: float
    post_std: float


@dataclass
class _SoftCutoff:
    """Мягкий срез/полка: SRC-апскейл, lossy-источник или намеренный lowpass."""

    cutoff_hz: float
    pre_level: float
    floor_level: float
    drop_db: float
    floor_std: float
    transition_width_hz: float
    estimated_source: str
    confidence: float


@dataclass
class _TimeEvidence:
    """Распределение верхней границы спектра по громким кадрам."""

    frames: int
    edge_median_hz: float
    edge_p10_hz: float
    edge_p90_hz: float
    consistency: float | None = None
    above_fraction: float | None = None
    below_fraction: float | None = None


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _median_band(
    freqs: np.ndarray,
    values: np.ndarray,
    f0: float,
    f1: float,
    min_width_hz: float = 200.0,
) -> float | None:
    """Медиана значений в полосе [f0, f1]. None, если полоса слишком узкая."""
    if f1 <= f0 or (f1 - f0) < min_width_hz:
        return None
    i0 = int(np.searchsorted(freqs, f0, side="left"))
    i1 = int(np.searchsorted(freqs, f1, side="right"))
    if i1 - i0 < 3:
        return None
    return float(np.median(values[i0:i1]))


def _smooth_spectrum(
    freqs: np.ndarray, levels_db: np.ndarray, width_hz: float = 60.0
) -> np.ndarray:
    """Медианное + скользящее среднее сглаживание по частоте."""
    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 1.0
    median_size = max(1, int(round(width_hz / df)))
    if median_size % 2 == 0:
        median_size += 1
    smoothed = median_filter(levels_db, size=median_size, mode="nearest")
    average_size = max(1, int(round(width_hz * 2.2 / df)))
    smoothed = uniform_filter1d(smoothed, size=average_size, mode="nearest")
    return smoothed


def _frame_edges(
    spectrogram: SpectrogramData | None,
    cutoff_hz: float | None = None,
) -> _TimeEvidence | None:
    """Считает верхнюю границу спектра для каждого громкого кадра.

    Граница — последняя частота, где уровень кадра не ниже −55 дБ
    относительно его же полосы 1–4 кГц. Медиана и перцентили показывают,
    постоянен ли срез во времени (VBR, монтаж, разные участки).
    """
    if spectrogram is None:
        return None
    data = spectrogram.levels_db
    freqs = spectrogram.freqs
    nyquist = float(spectrogram.nyquist)
    if (
        data.ndim != 2
        or data.shape[1] < _TIME_MIN_FRAMES
        or freqs.size != data.shape[0]
        or nyquist < _TIME_MIN_NYQUIST_HZ
    ):
        return None

    df = float(freqs[1] - freqs[0]) if freqs.size > 1 else 1.0
    smooth_size = max(3, int(round(_TIME_SMOOTH_HZ / df)))
    if smooth_size % 2 == 0:
        smooth_size += 1
    smoothed = uniform_filter1d(data, size=smooth_size, axis=0, mode="nearest")

    ref_lo, ref_hi = 1_000.0, min(4_000.0, nyquist * 0.55)
    if ref_hi - ref_lo < 500.0:
        return None
    r0 = int(np.searchsorted(freqs, ref_lo, side="left"))
    r1 = int(np.searchsorted(freqs, ref_hi, side="right"))
    if r1 - r0 < 3:
        return None
    frame_ref = np.median(smoothed[r0:r1, :], axis=0)
    valid = np.isfinite(frame_ref) & (frame_ref >= _TIME_MIN_REF_DB)
    if int(valid.sum()) < _TIME_MIN_FRAMES:
        return None

    relative = smoothed - frame_ref[None, :]
    search_lo = min(_MIN_CUTOFF_HZ, nyquist * 0.55)
    search_hi = nyquist - max(200.0, 0.01 * nyquist)
    lo = int(np.searchsorted(freqs, search_lo, side="left"))
    hi = int(np.searchsorted(freqs, search_hi, side="right"))
    if hi - lo < 3:
        return None

    band = relative[lo:hi, valid]
    mask = band >= _TIME_EDGE_REL_DB
    has_edge = mask.any(axis=0)
    if int(has_edge.sum()) < _TIME_MIN_FRAMES:
        return None
    last = mask.shape[0] - 1 - np.argmax(mask[::-1, :], axis=0)
    edges = freqs[lo + last[has_edge]]
    edges = edges[edges >= search_lo]
    if edges.size < _TIME_MIN_FRAMES:
        return None

    p10, median, p90 = np.percentile(edges, [10.0, 50.0, 90.0])
    evidence = _TimeEvidence(
        frames=int(edges.size),
        edge_median_hz=float(median),
        edge_p10_hz=float(p10),
        edge_p90_hz=float(p90),
    )
    if cutoff_hz is not None:
        evidence.consistency = float(
            np.mean(np.abs(edges - cutoff_hz) <= _TIME_TOLERANCE_HZ)
        )
        evidence.above_fraction = float(np.mean(edges > cutoff_hz + _TIME_MARGIN_HZ))
        evidence.below_fraction = float(np.mean(edges < cutoff_hz - _TIME_MARGIN_HZ))
    return evidence


def _edge_metrics(
    freqs: np.ndarray,
    relative_db: np.ndarray,
    cutoff: float,
    nyquist: float,
) -> _EdgeMetrics | None:
    """Измеряет перепад и «полку» вокруг предполагаемого среза."""
    pre_local = _median_band(freqs, relative_db, cutoff - 450, cutoff - 80)
    post_local = _median_band(freqs, relative_db, cutoff + 80, cutoff + 450)
    pre_wide = _median_band(freqs, relative_db, cutoff - 2400, cutoff - 500)
    post_wide = _median_band(
        freqs,
        relative_db,
        cutoff + 500,
        min(cutoff + 2500, nyquist - 40),
        _MIN_POST_WIDTH_HZ,
    )
    if None in (pre_local, post_local, pre_wide, post_wide):
        return None

    plateau_hi = min(cutoff + 2500, nyquist - 40)
    if plateau_hi - (cutoff + 500) < 300:
        return None
    i0 = int(np.searchsorted(freqs, cutoff + 500, side="left"))
    i1 = int(np.searchsorted(freqs, plateau_hi, side="right"))
    if i1 - i0 < 3:
        return None
    post_std = float(np.std(relative_db[i0:i1]))

    wide_drop = float(pre_wide - post_wide)
    edge_drop = float(pre_local - post_local)
    sharpness = edge_drop / wide_drop if wide_drop > 1e-6 else 0.0
    return _EdgeMetrics(
        cutoff_hz=cutoff,
        wide_drop=wide_drop,
        edge_drop=edge_drop,
        sharpness=sharpness,
        pre_level=float(pre_wide),
        post_level=float(post_wide),
        post_std=post_std,
    )


def _find_brickwall(
    freqs: np.ndarray,
    relative_db: np.ndarray,
    nyquist: float,
) -> _EdgeMetrics | None:
    """Ищет самый выраженный обрыв выше 10 кГц."""
    if nyquist < 15_000:
        return None
    search_hi = nyquist - max(800.0, 0.02 * nyquist)
    search_lo = min(_MIN_CUTOFF_HZ, nyquist * 0.55)
    if search_hi - search_lo < 800:
        return None

    df = float(freqs[1] - freqs[0])
    step = max(25.0, df * 4.0)
    best: _EdgeMetrics | None = None
    best_score = -np.inf

    candidate = search_lo
    while candidate <= search_hi:
        metrics = _edge_metrics(freqs, relative_db, candidate, nyquist)
        if metrics is not None:
            passed = (
                metrics.wide_drop >= _WIDE_DROP_DB
                and metrics.edge_drop >= _EDGE_DROP_DB
                and metrics.sharpness >= _MIN_SHARPNESS
                and metrics.post_std <= _MAX_PLATEAU_STD_DB
                and metrics.pre_level >= _MIN_PRE_LEVEL_DB
            )
            if passed:
                score = metrics.edge_drop + 0.2 * metrics.wide_drop
                if score > best_score:
                    best_score = score
                    best = metrics
        candidate += step

    if best is not None:
        # Уточняем положение среза по максимальному градиенту рядом с кандидатом.
        best.cutoff_hz = _refine_cutoff(freqs, relative_db, best.cutoff_hz)
    return best


def _refine_cutoff(freqs: np.ndarray, relative_db: np.ndarray, approx: float) -> float:
    """Ищет точку максимального спада в окрестности ±300 Гц."""
    half_window = max(60.0, 0.004 * freqs[-1])
    grid = np.arange(approx - 300.0, approx + 301.0, max(freqs[1] - freqs[0], 5.0))
    if grid.size == 0:
        return approx
    before = np.interp(grid - half_window, freqs, relative_db)
    after = np.interp(grid + half_window, freqs, relative_db)
    return float(grid[int(np.argmin(after - before))])


def _match_upscale_source(edge_hz: float) -> tuple[str, float] | None:
    """Сопоставляет верхнюю границу спектра со стандартной частотой источника."""
    if edge_hz <= 23_000.0:
        label, reference = "44.1 кГц", 22_050.0
    elif edge_hz <= 25_500.0:
        label, reference = "44.1/48 кГц", 24_000.0
    elif edge_hz <= 33_500.0:
        label, reference = "64 кГц", 32_000.0
    elif edge_hz <= 46_000.0:
        label, reference = "88.2/96 кГц", 45_000.0
    elif edge_hz <= 49_000.0:
        label, reference = "88.2/96 кГц", 48_000.0
    else:
        return None
    if abs(edge_hz - reference) > 2_500.0:
        return None
    return label, reference


def _describe_soft_source(cutoff_hz: float, meta: AudioMeta) -> str:
    """Описывает вероятный источник для мягкого среза."""
    sample_rate = meta.sample_rate
    if sample_rate > 48_000:
        match = _match_upscale_source(cutoff_hz)
        if match is not None:
            label, reference = match
            if sample_rate > reference * 2.05:
                return f"апскейл с {label} (спектр ограничен ~{format_khz(cutoff_hz)})"
        if cutoff_hz <= 22_000.0:
            return (
                f"апскейл с lossy-источника ({_guess_bitrate(cutoff_hz, sample_rate)})"
            )
        return f"срез спектра в верхней части диапазона (~{format_khz(cutoff_hz)})"

    if sample_rate >= 47_000:
        if cutoff_hz >= 21_800.0:
            return "возможный апскейл с 44.1 кГц"
        return _guess_bitrate(cutoff_hz, sample_rate)
    return _guess_bitrate(cutoff_hz, sample_rate)


def _find_soft_cutoff(
    freqs: np.ndarray,
    relative_db: np.ndarray,
    meta: AudioMeta,
) -> _SoftCutoff | None:
    """Ищет мягкий срез и ровную низкоуровневую полку выше него.

    В отличие от «кирпичной стены», здесь переход может занимать несколько
    кГц (SRC, намеренный lowpass, некоторые lossy-кодеки). Признак среза —
    устойчивая полка, которая заметно ниже опорного уровня и не имеет
    наклона, а не мгновенный скачок.
    """
    nyquist = float(freqs[-1])
    if nyquist < 16_000.0:
        return None

    df = float(freqs[1] - freqs[0])
    window = max(3, int(round(350.0 / df)))
    if window % 2 == 0:
        window += 1
    smoothed = median_filter(relative_db, size=window, mode="nearest")

    pre = _median_band(
        freqs,
        smoothed,
        9_000.0,
        min(14_000.0, nyquist * 0.70),
        500.0,
    )
    if pre is None or pre < _MIN_PRE_LEVEL_DB:
        return None

    # Полка у верхней границы диапазона.
    floor_lo = max(nyquist - 700.0, nyquist * 0.90)
    floor_hi = nyquist - 80.0
    floor = _median_band(freqs, smoothed, floor_lo, floor_hi, 250.0)
    if floor is None:
        return None

    i0 = int(np.searchsorted(freqs, floor_lo, side="left"))
    i1 = int(np.searchsorted(freqs, floor_hi, side="right"))
    if i1 - i0 < 3:
        return None
    floor_std = float(np.std(smoothed[i0:i1]))
    if floor > pre - 30.0 or floor_std > 5.0:
        return None

    if i1 - i0 >= 8:
        slope = float(np.polyfit(freqs[i0:i1], smoothed[i0:i1], 1)[0] * 1_000.0)
        if abs(slope) > 2.5:
            return None
    else:
        slope = 0.0

    # Конец перехода: последняя частота, где ещё есть энергия над полкой.
    threshold = floor + max(6.0, 3.0 * floor_std)
    above = np.where(smoothed >= threshold)[0]
    if above.size < 3:
        return None
    cutoff = float(freqs[int(above[-1])])
    if cutoff < 14_500.0 or cutoff > nyquist - 300.0:
        return None

    # Начало перехода — последняя точка в пределах 3 дБ от опорного уровня.
    pre_idx = np.where((freqs < cutoff) & (smoothed >= pre - 3.0))[0]
    start = float(freqs[int(pre_idx[-1])]) if pre_idx.size else cutoff
    transition = max(0.0, cutoff - start)
    if transition > 6_000.0:
        return None

    drop = pre - floor
    if drop < 30.0:
        return None

    estimated_source = _describe_soft_source(cutoff, meta)
    confidence = 0.52
    confidence += 0.18 * _clamp((drop - 30.0) / 60.0)
    confidence += 0.10 * _clamp((5.0 - floor_std) / 5.0)
    confidence += 0.08 * _clamp((5.0 - abs(slope)) / 5.0)
    if "апскейл с" in estimated_source and "возможный" not in estimated_source:
        confidence += 0.07
    if transition > 4_000.0:
        confidence -= 0.08
    confidence = _clamp(confidence, 0.50, 0.90)

    return _SoftCutoff(
        cutoff_hz=cutoff,
        pre_level=pre,
        floor_level=floor,
        drop_db=drop,
        floor_std=floor_std,
        transition_width_hz=transition,
        estimated_source=estimated_source,
        confidence=confidence,
    )


def _guess_bitrate(cutoff_hz: float, sample_rate: int) -> str:
    """Грубая оценка битрейта lossy-кодека по частоте среза."""
    if sample_rate < 44_100:
        return "MP3/AAC (битрейт не оценивается на низкой частоте дискретизации)"
    if sample_rate <= 48_000:
        normalized = cutoff_hz * 44_100.0 / sample_rate
    else:
        normalized = cutoff_hz
    if normalized < 15_000:
        label = "96 кбит/с"
    elif normalized < 16_000:
        label = "96–128 кбит/с"
    elif normalized < 17_000:
        label = "128 кбит/с"
    elif normalized < 17_950:
        label = "160 кбит/с"
    elif normalized < 19_100:
        label = "192 кбит/с"
    elif normalized < 19_850:
        label = "224–256 кбит/с"
    else:
        label = "320 кбит/с"
    return f"MP3/AAC ≈ {label}"


def _guess_source(cutoff_hz: float, meta: AudioMeta) -> str:
    """Пытается определить исходный формат по частоте среза."""
    # Hi-res контейнер: обрыв на границе стандартной частоты = апскейл.
    for reference, label in _UPSCALE_ORIGINS:
        if meta.sample_rate > reference * 2.05 and abs(cutoff_hz - reference) <= max(
            300.0, 0.012 * reference
        ):
            return f"апскейл с {label} (срез на границе исходного диапазона)"
    if meta.sample_rate > 48_000 and cutoff_hz <= 22_000:
        return (
            f"апскейл с lossy-источника ({_guess_bitrate(cutoff_hz, meta.sample_rate)})"
        )
    if 47_000 <= meta.sample_rate <= 48_000 and cutoff_hz >= 21_800:
        return "возможный апскейл с 44.1 кГц"
    return _guess_bitrate(cutoff_hz, meta.sample_rate)


def _apply_lossy_override(result: DetectionResult, meta: AudioMeta) -> DetectionResult:
    """Если контейнер сам по себе lossy — итоговый вердикт всегда LOSSY SOURCE."""
    if not meta.is_lossy:
        return result
    detail = f"Формат {meta.codec} использует сжатие с потерями."
    if result.cutoff_hz is not None:
        detail += f" Найденный срез: {format_khz(result.cutoff_hz)}."
    result.verdict = "lossy"
    result.headline = "LOSSY SOURCE"
    result.reason = detail
    result.confidence = 1.0
    result.estimated_source = result.estimated_source or "MP3/AAC/иной lossy-кодек"
    result.notes.insert(
        0, "Файл изначально сжат с потерями — это не lossless-источник."
    )
    return result


def _floor_text(level_db: float) -> str:
    """Короткое описание уровня полки."""
    if level_db < -120.0:
        return "ниже −120 дБ"
    return f"≈ {level_db:.0f} дБ"


def _source_sentence(source: str) -> str:
    """Согласует оценку источника с фразой в вердикте."""
    if source.startswith("возможный "):
        return f"Возможно, это {source.removeprefix('возможный ')}."
    if "апскейл" in source:
        return f"Похоже на {source}."
    return f"Вероятно, исходник — {source}."


def _time_note(evidence: _TimeEvidence, cutoff_hz: float | None) -> str:
    """Описывает стабильность верхней границы по кадрам."""
    span = (
        f"{format_khz(evidence.edge_p10_hz, 1)}–{format_khz(evidence.edge_p90_hz, 1)}"
    )
    if cutoff_hz is not None and evidence.consistency is not None:
        if evidence.consistency >= 0.75:
            text = (
                f"Срез подтверждён во времени: {evidence.consistency:.0%} "
                f"из {evidence.frames} громких кадров (медиана "
                f"{format_khz(evidence.edge_median_hz, 1)}, p10–p90 {span})."
            )
        else:
            text = (
                f"Срез непостоянен: с найденной частотой совпало "
                f"{evidence.consistency:.0%} из {evidence.frames} громких кадров "
                f"(медиана {format_khz(evidence.edge_median_hz, 1)}, "
                f"p10–p90 {span})."
            )
        if (evidence.above_fraction or 0.0) > 0.15:
            text += (
                f" В {evidence.above_fraction:.0%} кадров есть энергия выше "
                "найденного среза."
            )
        if (evidence.below_fraction or 0.0) > 0.25:
            text += (
                f" В {evidence.below_fraction:.0%} кадров верхняя граница заметно ниже."
            )
        return text
    return (
        f"По кадрам верхняя граница: медиана "
        f"{format_khz(evidence.edge_median_hz, 1)}, p10–p90 {span} "
        f"({evidence.frames} громких кадров)."
    )


def _time_confidence_delta(evidence: _TimeEvidence | None) -> float:
    """Небольшая поправка уверенности по временной согласованности."""
    if evidence is None or evidence.consistency is None:
        return 0.0
    delta = 0.08 * (evidence.consistency - 0.5)
    spread = evidence.edge_p90_hz - evidence.edge_p10_hz
    if spread > _TIME_TOLERANCE_HZ:
        delta -= 0.05 * _clamp((spread - _TIME_TOLERANCE_HZ) / 3_000.0)
    if evidence.above_fraction is not None:
        delta -= 0.12 * _clamp((evidence.above_fraction - 0.15) / 0.5)
    if evidence.below_fraction is not None:
        delta -= 0.06 * _clamp((evidence.below_fraction - 0.25) / 0.5)
    return delta


def _apply_time_evidence(
    result: DetectionResult, evidence: _TimeEvidence | None
) -> DetectionResult:
    """Переносит временные метрики в результат для GUI и отчёта."""
    if evidence is None:
        return result
    result.time_frames = evidence.frames
    result.time_edge_median_hz = evidence.edge_median_hz
    result.time_edge_p10_hz = evidence.edge_p10_hz
    result.time_edge_p90_hz = evidence.edge_p90_hz
    result.time_consistency = evidence.consistency
    result.time_above_fraction = evidence.above_fraction
    return result


def _apply_effective_bits(
    result: DetectionResult, spectrum: SpectrumData, meta: AudioMeta
) -> DetectionResult:
    """Отмечает недозаполненную разрядность целочисленного PCM-контейнера."""
    if (
        spectrum.grid_bits is None
        or meta.bit_depth is None
        or meta.bit_depth <= spectrum.grid_bits
        or not meta.subtype.upper().startswith("PCM")
    ):
        return result
    result.effective_bits = spectrum.grid_bits
    result.notes.append(
        f"Эффективная разрядность ≈{spectrum.grid_bits} бит: контейнер "
        f"{meta.bit_depth} бит, но младшие биты не используются."
    )
    return result


def detect(
    spectrum: SpectrumData,
    meta: AudioMeta,
    spectrogram: SpectrogramData | None = None,
) -> DetectionResult:
    """Главная функция детекции. Возвращает вердикт и подробности."""
    freqs = spectrum.freqs
    levels = spectrum.levels_db
    nyquist = spectrum.nyquist
    notes: list[str] = []

    # --- Проверка пригодности файла --------------------------------------
    if spectrum.frames_used == 0 or spectrum.peak_db < -70:
        result = DetectionResult(
            verdict="inconclusive",
            headline="INCONCLUSIVE",
            reason="Файл практически беззвучен — анализировать нечего.",
        )
        return _apply_lossy_override(result, meta)

    if spectrum.rms_db < -62:
        result = DetectionResult(
            verdict="inconclusive",
            headline="INCONCLUSIVE",
            reason="Файл слишком тихий: высокочастотный диапазон уходит в шум.",
        )
        return _apply_lossy_override(result, meta)

    # --- Опорный уровень и относительный спектр --------------------------
    ref_lo, ref_hi = 1000.0, min(4000.0, 0.55 * nyquist)
    if ref_hi - ref_lo < 500.0:
        ref_lo, ref_hi = 300.0, max(1000.0, 0.5 * nyquist)
    reference = _median_band(freqs, levels, ref_lo, ref_hi, 300.0)
    if reference is None:
        result = DetectionResult(
            verdict="inconclusive",
            headline="INCONCLUSIVE",
            reason="Недостаточный диапазон частот для анализа.",
        )
        return _apply_lossy_override(result, meta)

    smoothed = _smooth_spectrum(freqs, levels)
    relative = smoothed - reference

    # --- Поиск обрыва -----------------------------------------------------
    edge = _find_brickwall(freqs, relative, nyquist)
    top_level = _median_band(
        freqs,
        relative,
        max(15_000.0, 0.80 * nyquist),
        nyquist * 0.995,
        300.0,
    )
    soft = None if edge is not None else _find_soft_cutoff(freqs, relative, meta)
    time_evidence = _frame_edges(spectrogram) if edge is None and soft is None else None

    if edge is not None:
        cutoff = edge.cutoff_hz
        time_evidence = _frame_edges(spectrogram, cutoff)
        confidence = _clamp(
            0.45 * _clamp((edge.wide_drop - 20.0) / 30.0)
            + 0.30 * _clamp((edge.edge_drop - 10.0) / 25.0)
            + 0.15 * _clamp((0.9 - edge.post_std / 6.0) / 0.9)
            + 0.10 * _clamp((edge.pre_level + 45.0) / 20.0),
            0.30,
            0.99,
        )
        confidence = _clamp(
            confidence + _time_confidence_delta(time_evidence), 0.25, 0.99
        )
        plateau_text = _floor_text(edge.post_level)
        notes.append(
            f"Выше среза энергия падает на {edge.wide_drop:.0f} дБ "
            f"(полка {plateau_text} относительно средних частот)."
        )
        notes.append(f"Крутизна обрыва: {edge.edge_drop:.0f} дБ на участке ±0.4 кГц.")
        if cutoff < nyquist - 1_000.0:
            notes.append(
                f"Для {meta.sample_rate / 1000.0:.1f} кГц честный спектр должен "
                f"доходить до {nyquist / 1000.0:.1f} кГц."
            )
        if time_evidence is not None:
            notes.append(_time_note(time_evidence, cutoff))
        estimated = _guess_source(cutoff, meta)

        result = DetectionResult(
            verdict="fake",
            headline="FAKE LOSSLESS (UPSCALE)",
            reason=(
                f"Обнаружен срез на частоте {format_khz(cutoff)}. "
                f"{_source_sentence(estimated)} Спектр выше среза содержит только "
                f"низкоуровневую полку ({plateau_text} относительно средних частот)."
            ),
            cutoff_hz=cutoff,
            drop_db=edge.wide_drop,
            plateau_level_db=edge.post_level,
            confidence=confidence,
            hf_level_db=top_level,
            estimated_source=estimated,
            notes=notes,
        )
        _apply_time_evidence(result, time_evidence)
    elif soft is not None:
        cutoff = soft.cutoff_hz
        nyquist_khz = nyquist / 1000.0
        time_evidence = _frame_edges(spectrogram, cutoff)
        notes.append(
            f"Выше среза уровень стабилен: {soft.floor_level:.0f} дБ "
            f"(σ {soft.floor_std:.1f} дБ), переход ~"
            f"{soft.transition_width_hz / 1000.0:.1f} кГц."
        )
        if cutoff < nyquist - 1_000.0:
            notes.append(
                f"Для {meta.sample_rate / 1000.0:.1f} кГц честный спектр должен "
                f"доходить до {nyquist_khz:.1f} кГц; выше {cutoff / 1000.0:.1f} кГц "
                "энергии нет."
            )
        notes.append(
            "Мягкая полка может быть следствием ресемплера (SRC) или "
            "намеренного lowpass/мастеринга."
        )
        if time_evidence is not None:
            notes.append(_time_note(time_evidence, cutoff))
        confidence = _clamp(
            soft.confidence + _time_confidence_delta(time_evidence), 0.45, 0.95
        )
        result = DetectionResult(
            verdict="fake",
            headline="FAKE LOSSLESS (UPSCALE)",
            reason=(
                f"Спектр уходит в низкоуровневую полку около {format_khz(cutoff)} "
                f"({_floor_text(soft.floor_level)} относительно средних частот), "
                f"хотя частота Найквиста — {nyquist_khz:.1f} кГц. "
                f"{_source_sentence(soft.estimated_source)}"
            ),
            cutoff_hz=cutoff,
            drop_db=soft.drop_db,
            plateau_level_db=soft.floor_level,
            transition_width_hz=soft.transition_width_hz,
            confidence=confidence,
            hf_level_db=top_level,
            estimated_source=soft.estimated_source,
            notes=notes,
        )
        _apply_time_evidence(result, time_evidence)
    elif top_level is None:
        result = DetectionResult(
            verdict="inconclusive",
            headline="INCONCLUSIVE",
            reason="Не удалось измерить высокочастотную область спектра.",
        )
        _apply_time_evidence(result, time_evidence)
    elif top_level >= -45.0:
        confidence = _clamp(0.55 + (top_level + 45.0) / 45.0 * 0.4, 0.55, 0.98)
        notes.append(
            f"Уровень у частоты Найквиста: {top_level:.0f} дБ относительно средних частот."
        )
        notes.append("Плавного спада или резкого обрыва в верхней полосе не найдено.")
        mixed_bandwidth = (
            time_evidence is not None and time_evidence.edge_p10_hz < nyquist - 3_000.0
        )
        if time_evidence is not None:
            notes.append(_time_note(time_evidence, None))
        if mixed_bandwidth:
            notes.append(
                "Средний спектр доходит до верха, но в части кадров граница "
                "заметно ниже — возможен фильтрованный фрагмент, монтаж или VBR."
            )
            result = DetectionResult(
                verdict="inconclusive",
                headline="INCONCLUSIVE",
                reason=(
                    "Спектр неоднороден: в части файла верхняя граница заметно "
                    f"ниже Найквиста (p10 {format_khz(time_evidence.edge_p10_hz, 1)}). "
                    "Средний спектр скрывает этот участок, поэтому однозначный "
                    "вердикт невозможен."
                ),
                confidence=0.35,
                hf_level_db=top_level,
                notes=notes,
            )
        else:
            result = DetectionResult(
                verdict="lossless",
                headline="TRUE LOSSLESS",
                reason=(
                    "Резких срезов не найдено: спектр сохраняет энергию "
                    "вплоть до частоты Найквиста."
                ),
                confidence=confidence,
                hf_level_db=top_level,
                notes=notes,
            )
        _apply_time_evidence(result, time_evidence)
    else:
        if spectrum.rms_db < -45:
            reason = (
                "Тихий материал: высокочастотная часть спектра теряется в шуме, "
                "однозначный вывод невозможен."
            )
        elif top_level < -70:
            reason = (
                "Высокочастотный диапазон почти пуст, но чёткого обрыва нет. "
                "Возможен естественный спад или сильная фильтрация."
            )
        else:
            reason = (
                "Верхние частоты слабые, но резкого среза не обнаружено. "
                "Данных для однозначного вердикта недостаточно."
            )
        notes.append(
            f"Уровень у частоты Найквиста: {top_level:.0f} дБ относительно средних частот."
        )
        if time_evidence is not None:
            notes.append(_time_note(time_evidence, None))
        result = DetectionResult(
            verdict="inconclusive",
            headline="INCONCLUSIVE",
            reason=reason,
            confidence=0.2,
            hf_level_db=top_level,
            notes=notes,
        )
        _apply_time_evidence(result, time_evidence)

    result = _apply_effective_bits(result, spectrum, meta)
    return _apply_lossy_override(result, meta)
