"""Спектральный анализ: усреднённый спектр и спектрограмма.

Файл читается потоково блоками, поэтому анализ не требует загрузки всего
аудио в память. Кадры для усреднённого спектра и для спектрограммы
рассчитываются за один проход.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import soundfile as sf

ProgressFn = Callable[[float], None]

# Бюджеты количества кадров: ограничивают время анализа длинных файлов.
MAX_SPECTRUM_FRAMES = 4000
MAX_SPECTROGRAM_FRAMES = 2600
_BLOCK_FRAMES = 262_144
_SILENCE_RMS = 1e-5  # кадры тише -100 dBFS не участвуют в усреднении

# Проверка эффективной разрядности: сколько сэмплов достаточно посмотреть.
_QUANT_SAMPLE_LIMIT = 1_500_000
_QUANT_MIN_SAMPLES = 10_000
_QUANT_16_TOLERANCE = 1e-3
_QUANT_24_TOLERANCE = 0.75


@dataclass
class SpectrumData:
    """Усреднённый по файлу спектр."""

    freqs: np.ndarray
    levels_db: np.ndarray
    n_fft: int
    frames_used: int
    nyquist: float
    rms_db: float
    peak_db: float
    grid_bits: int | None = None


@dataclass
class SpectrogramData:
    """Спектрограмма (матрица уровень/частота/время)."""

    times: np.ndarray
    freqs: np.ndarray
    levels_db: np.ndarray
    max_db: float
    nyquist: float


def _next_pow2(value: float) -> int:
    return int(2 ** math.ceil(math.log2(max(2.0, value))))


def _prev_pow2(value: float) -> int:
    return int(2 ** math.floor(math.log2(max(2.0, value))))


def _choose_spectrum_fft(sample_rate: int, total_frames: int) -> int:
    """Окно БПФ с разрешением около 6 Гц."""
    n_fft = _next_pow2(sample_rate / 6.0)
    n_fft = int(np.clip(n_fft, 4096, 32768))
    if total_frames < n_fft:
        n_fft = max(256, _prev_pow2(total_frames))
    return n_fft


def _choose_spectrogram_fft(sample_rate: int, total_frames: int) -> int:
    """Окно для спектрограммы: компромисс между временем и частотой."""
    if sample_rate <= 48_000:
        n_fft = 2048
    elif sample_rate <= 96_000:
        n_fft = 4096
    else:
        n_fft = 8192
    if total_frames < n_fft:
        n_fft = max(256, _prev_pow2(total_frames))
    return n_fft


def _db(value: np.ndarray | float) -> np.ndarray | float:
    """Переводит амплитуду в дБ с защитой от log(0)."""
    return 20.0 * np.log10(np.maximum(value, 1e-12))


def compute_spectra(
    file: sf.SoundFile,
    sample_rate: int,
    total_frames: int,
    progress: ProgressFn | None = None,
) -> tuple[SpectrumData, SpectrogramData]:
    """Один проход по файлу: средний спектр + спектрограмма.

    Для спектра используется окно высокого разрешения, для спектрограммы —
    более короткое окно. Шаг кадров подбирается так, чтобы уложиться в
    бюджеты ``MAX_SPECTRUM_FRAMES`` и ``MAX_SPECTROGRAM_FRAMES``.
    """
    if total_frames <= 0:
        raise ValueError("Пустой аудиопоток")

    file.seek(0)

    # --- Параметры кадров -------------------------------------------------
    n_fft_spec = _choose_spectrum_fft(sample_rate, total_frames)
    hop_spec = max(n_fft_spec // 2, int(math.ceil(total_frames / MAX_SPECTRUM_FRAMES)))

    n_fft_sg = _choose_spectrogram_fft(sample_rate, total_frames)
    hop_sg = max(1, int(math.ceil(total_frames / MAX_SPECTROGRAM_FRAMES)))

    win_spec = np.hanning(n_fft_spec)
    win_sg = np.hanning(n_fft_sg)
    norm_spec = max(win_spec.sum() / 2.0, 1e-12)
    norm_sg = max(win_sg.sum() / 2.0, 1e-12)

    n_bins_spec = n_fft_spec // 2 + 1
    n_bins_sg = n_fft_sg // 2 + 1
    power_sum = np.zeros(n_bins_spec, dtype=np.float64)
    power_sum_all = np.zeros(n_bins_spec, dtype=np.float64)
    frames_loud = 0
    frames_all = 0

    # Матрица спектрограммы заполняется по мере чтения.
    sg_matrix = np.empty((n_bins_sg, MAX_SPECTROGRAM_FRAMES + 8), dtype=np.float32)
    sg_times: list[float] = []
    sg_max = -np.inf

    # --- Состояние потока -------------------------------------------------
    buffer = np.empty((0, file.channels), dtype=np.float32)
    buffer_start = 0  # абсолютный номер первого сэмпла в buffer
    next_spec = 0
    next_sg = 0
    read_frames = 0
    peak = 0.0
    sum_squares = 0.0
    quant_samples = 0
    quant_16_hits = 0
    quant_24_hits = 0

    while True:
        block = file.read(_BLOCK_FRAMES, dtype="float32", always_2d=True)
        if block.size == 0:
            break

        # Сумма мощностей каналов: не теряет противофазный ВЧ-контент.
        if buffer.shape[0] == 0:
            buffer = block.copy()
        else:
            buffer = np.concatenate((buffer, block), axis=0)
        buffer_end = buffer_start + buffer.shape[0]

        # Кадры усреднённого спектра.
        while next_spec + n_fft_spec <= buffer_end:
            offset = next_spec - buffer_start
            frame = buffer[offset : offset + n_fft_spec, :]
            rms = (
                float(np.sqrt(np.mean(frame * frame, dtype=np.float64)))
                if frame.size
                else 0.0
            )
            spectra = np.fft.rfft(frame * win_spec[:, None], axis=0)
            power = np.mean(np.abs(spectra) ** 2, axis=1) / (norm_spec**2)
            power_sum_all += power
            frames_all += 1
            if rms >= _SILENCE_RMS:
                power_sum += power
                frames_loud += 1
            next_spec += hop_spec

        # Кадры спектрограммы.
        while (
            next_sg + n_fft_sg <= buffer_end and len(sg_times) < MAX_SPECTROGRAM_FRAMES
        ):
            offset = next_sg - buffer_start
            frame = buffer[offset : offset + n_fft_sg, :]
            spectra = np.fft.rfft(frame * win_sg[:, None], axis=0)
            power = np.mean(np.abs(spectra) ** 2, axis=1) / (norm_sg**2)
            levels = (10.0 * np.log10(np.maximum(power, 1e-24))).astype(np.float32)
            sg_matrix[:, len(sg_times)] = levels
            sg_times.append(next_sg / sample_rate)
            sg_max = max(sg_max, float(levels.max()))
            next_sg += hop_sg

        # Оценка эффективной разрядности по сетке квантования.
        if quant_samples < _QUANT_SAMPLE_LIMIT:
            values = block.reshape(-1)
            values = values[np.abs(values) > 1e-4]
            if values.size:
                stride = max(1, values.size // 150_000)
                values = values[::stride].astype(np.float64)
                scaled16 = values * 32_768.0
                quant_16_hits += int(
                    np.count_nonzero(
                        np.abs(scaled16 - np.round(scaled16)) <= _QUANT_16_TOLERANCE
                    )
                )
                scaled24 = values * 8_388_608.0
                quant_24_hits += int(
                    np.count_nonzero(
                        np.abs(scaled24 - np.round(scaled24)) <= _QUANT_24_TOLERANCE
                    )
                )
                quant_samples += values.size

        # Освобождаем уже прочитанный буфер (но не «заглядывая» в будущее).
        cut = min(next_spec, next_sg) - buffer_start
        cut = min(cut, buffer.shape[0])
        if cut > 0:
            buffer = buffer[cut:, :]
            buffer_start += cut

        read_frames += block.shape[0]
        peak = max(peak, float(np.max(np.abs(block))))
        sum_squares += float(np.sum(block.astype(np.float64) ** 2))
        if progress is not None:
            progress(min(1.0, read_frames / total_frames))

    # --- Итоговый усреднённый спектр -------------------------------------
    if frames_loud >= 5:
        mean_power = power_sum / frames_loud
        frames_used = frames_loud
    elif frames_all:
        mean_power = power_sum_all / frames_all
        frames_used = frames_all
    else:
        mean_power = np.zeros(n_bins_spec, dtype=np.float64)
        frames_used = 0

    freqs_spec = np.fft.rfftfreq(n_fft_spec, d=1.0 / sample_rate)
    levels_db = 10.0 * np.log10(np.maximum(mean_power, 1e-24))

    grid_bits = None
    if quant_samples >= _QUANT_MIN_SAMPLES:
        fraction_16 = quant_16_hits / quant_samples
        fraction_24 = quant_24_hits / quant_samples
        if fraction_16 >= 0.995 and fraction_24 >= 0.995:
            grid_bits = 16
        elif fraction_24 >= 0.995:
            grid_bits = 24

    rms = math.sqrt(sum_squares / max(1, read_frames * max(1, file.channels)))
    spectrum = SpectrumData(
        freqs=freqs_spec,
        levels_db=levels_db,
        n_fft=n_fft_spec,
        frames_used=frames_used,
        nyquist=sample_rate / 2.0,
        rms_db=float(_db(max(rms, 1e-12))),
        peak_db=float(_db(max(peak, 1e-12))),
        grid_bits=grid_bits,
    )

    # --- Итоговая спектрограмма ------------------------------------------
    frame_count = len(sg_times)
    if frame_count:
        matrix = sg_matrix[:, :frame_count].copy()
        max_db = float(np.max(matrix))
        freqs_sg = np.fft.rfftfreq(n_fft_sg, d=1.0 / sample_rate)
        times = np.asarray(sg_times, dtype=np.float64)
    else:
        matrix = np.zeros((n_bins_sg, 0), dtype=np.float32)
        max_db = -np.inf
        freqs_sg = np.fft.rfftfreq(n_fft_sg, d=1.0 / sample_rate)
        times = np.empty(0, dtype=np.float64)

    spectrogram = SpectrogramData(
        times=times,
        freqs=freqs_sg,
        levels_db=matrix,
        max_db=max_db,
        nyquist=sample_rate / 2.0,
    )
    if progress is not None:
        progress(1.0)
    return spectrum, spectrogram
