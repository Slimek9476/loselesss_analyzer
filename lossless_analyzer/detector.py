"""Ядро анализа: поиск «кирпичной стены» (brickwall cutoff) в спектре.

Идея метода
-----------
Lossy-кодеки (MP3, AAC, Vorbis и т.п.) физически не передают самые высокие
частоты. Спектр такого файла имеет почти вертикальный обрыв: выше некоторой
частоты энергия падает на десятки дБ и остаётся на уровне шума. Если такой
файл пересохранить во FLAC/WAV, обрыв никуда не денется.

Алгоритм:
1. Переводим спектр в дБ относительно средних частот (1–8 кГц).
2. Сглаживаем спектр, чтобы убрать узкие тональные пики.
3. Для каждой частоты-кандидата измеряем перепад между полосами до и после
   неё, «локальную» крутизну обрыва и разброс уровня на «полке» после среза.
4. Лучший кандидат проверяется по порогам. Если обрыв найден — вердикт FAKE.
5. Если обрыва нет, оцениваем наличие энергии у частоты Найквиста. Есть
   энергия — TRUE LOSSLESS; тихо/темно — INCONCLUSIVE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d

from .dsp import SpectrumData
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
_BAND_LIMIT_THRESHOLD_DB = -50.0

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
    confidence: float = 0.0
    hf_level_db: float | None = None
    estimated_source: str | None = None
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
class _BandLimit:
    """Мягкое ограничение полосы (типично для апскейла с SRC-фильтром)."""

    edge_hz: float
    pre_level: float
    empty_level: float
    source_label: str
    confidence: float


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
    else:
        return None
    if abs(edge_hz - reference) > 2_500.0:
        return None
    return label, reference


def _find_band_limit(
    freqs: np.ndarray,
    relative_db: np.ndarray,
    meta: AudioMeta,
) -> _BandLimit | None:
    """Ищет «мягкий» апскейл: контейнер hi-res, а спектр обрывается заметно ниже.

    Некоторые конвертеры (SRC) делают срез не мгновенным, а растянутым на
    несколько кГц. Детектор «кирпичной стены» такой случай пропускает, поэтому
    дополнительно проверяем, где заканчивается полоса с реальной энергией.
    """
    sample_rate = meta.sample_rate
    if sample_rate <= 48_000:
        return None
    nyquist = sample_rate / 2.0
    df = float(freqs[1] - freqs[0])
    window = max(3, int(round(1000.0 / df)))
    if window % 2 == 0:
        window += 1
    band_median = median_filter(relative_db, size=window, mode="nearest")

    # Верхняя граница полосы: последняя точка, где медиана ещё выше порога.
    index = band_median.size - 1
    while index > 0 and band_median[index] < _BAND_LIMIT_THRESHOLD_DB:
        index -= 1
    edge = float(freqs[index])
    if edge < 15_000.0:
        return None

    empty = _median_band(
        freqs,
        band_median,
        edge + 2000.0,
        min(nyquist * 0.98, edge + 12_000.0),
        1000.0,
    )
    if empty is None or empty > -55.0:
        return None

    pre = _median_band(freqs, band_median, edge - 3000.0, edge - 500.0, 800.0)
    if pre is None or pre < -40.0:
        return None

    # Контейнер должен быть существенно шире исходного диапазона.
    if sample_rate < 3.2 * edge:
        return None

    match = _match_upscale_source(edge)
    if match is None:
        return None
    label, reference = match

    confidence = _clamp(
        0.55
        + 0.20 * _clamp((-55.0 - empty) / 25.0)
        + 0.15 * _clamp(1.0 - abs(edge - reference) / 2500.0),
        0.50,
        0.88,
    )
    return _BandLimit(
        edge_hz=edge,
        pre_level=pre,
        empty_level=empty,
        source_label=label,
        confidence=confidence,
    )


def _guess_bitrate(cutoff_hz: float, sample_rate: int) -> str:
    """Грубая оценка битрейта lossy-кодека по частоте среза."""
    if sample_rate <= 48_000:
        normalized = cutoff_hz * 44_100.0 / sample_rate
    else:
        normalized = cutoff_hz
    if normalized < 15_000:
        label = "~96 кбит/с"
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


def detect(spectrum: SpectrumData, meta: AudioMeta) -> DetectionResult:
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
    band_limit = None if edge is not None else _find_band_limit(freqs, relative, meta)

    if edge is not None:
        cutoff = edge.cutoff_hz
        confidence = _clamp(
            0.45 * _clamp((edge.wide_drop - 20.0) / 30.0)
            + 0.30 * _clamp((edge.edge_drop - 10.0) / 25.0)
            + 0.15 * _clamp((0.9 - edge.post_std / 6.0) / 0.9)
            + 0.10 * _clamp((edge.pre_level + 45.0) / 20.0),
            0.30,
            0.99,
        )
        if edge.post_level < -120:
            plateau_text = "ниже −120 дБ"
        else:
            plateau_text = f"≈ {edge.post_level:.0f} дБ"
        notes.append(
            f"Выше среза энергия падает на {edge.wide_drop:.0f} дБ "
            f"(полка {plateau_text} относительно средних частот)."
        )
        notes.append(f"Крутизна обрыва: {edge.edge_drop:.0f} дБ на участке ±0.4 кГц.")
        estimated = _guess_source(cutoff, meta)

        result = DetectionResult(
            verdict="fake",
            headline="FAKE LOSSLESS (UPSCALE)",
            reason=(
                f"Обнаружен резкий срез на {format_khz(cutoff)} "
                f"(перепад {edge.wide_drop:.0f} дБ)."
            ),
            cutoff_hz=cutoff,
            drop_db=edge.wide_drop,
            plateau_level_db=edge.post_level,
            confidence=confidence,
            hf_level_db=top_level,
            estimated_source=estimated,
            notes=notes,
        )
    elif band_limit is not None:
        edge_khz = band_limit.edge_hz / 1000.0
        nyquist_khz = nyquist / 1000.0
        estimated = (
            f"апскейл с {band_limit.source_label} "
            f"(спектр ограничен ~{edge_khz:.1f} кГц)"
        )
        notes.append(
            f"Выше {edge_khz:.1f} кГц уровень падает до "
            f"{band_limit.empty_level:.0f} дБ относительно средних частот, "
            f"хотя частота Найквиста контейнера — {nyquist_khz:.1f} кГц."
        )
        notes.append(
            "Обрыв не резкий, поэтому, вероятно, применён ресемплер (SRC), "
            "но высокочастотный диапазон исходника пуст."
        )
        result = DetectionResult(
            verdict="fake",
            headline="FAKE LOSSLESS (UPSCALE)",
            reason=(
                f"Спектр обрывается около {format_khz(band_limit.edge_hz)} "
                f"при частоте Найквиста {nyquist_khz:.1f} кГц — признак апскейла."
            ),
            cutoff_hz=band_limit.edge_hz,
            drop_db=band_limit.pre_level - band_limit.empty_level,
            plateau_level_db=band_limit.empty_level,
            confidence=band_limit.confidence,
            hf_level_db=top_level,
            estimated_source=estimated,
            notes=notes,
        )
    elif top_level is None:
        result = DetectionResult(
            verdict="inconclusive",
            headline="INCONCLUSIVE",
            reason="Не удалось измерить высокочастотную область спектра.",
        )
    elif top_level >= -45.0:
        confidence = _clamp(0.55 + (top_level + 45.0) / 45.0 * 0.4, 0.55, 0.98)
        notes.append(
            f"Уровень у частоты Найквиста: {top_level:.0f} дБ относительно средних частот."
        )
        notes.append("Плавного спада или резкого обрыва в верхней полосе не найдено.")
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
        result = DetectionResult(
            verdict="inconclusive",
            headline="INCONCLUSIVE",
            reason=reason,
            confidence=0.2,
            hf_level_db=top_level,
            notes=notes,
        )

    return _apply_lossy_override(result, meta)
