"""Оркестрация анализа одного файла."""

from __future__ import annotations

import time
from dataclasses import dataclass

from .detector import DetectionResult, detect
from .dsp import ProgressFn, SpectrogramData, SpectrumData, compute_spectra
from .loader import AudioLoadError, AudioMeta, open_audio


@dataclass
class AnalysisResult:
    """Всё, что нужно интерфейсу для отображения одного файла."""

    meta: AudioMeta
    spectrum: SpectrumData
    spectrogram: SpectrogramData
    detection: DetectionResult
    elapsed_seconds: float


def _analyze_once(
    path: str, progress: ProgressFn | None, force_ffmpeg: bool
) -> AnalysisResult:
    started = time.perf_counter()
    with open_audio(path, force_ffmpeg=force_ffmpeg) as handle:
        spectrum, spectrogram = compute_spectra(
            handle.file,
            handle.meta.sample_rate,
            handle.meta.frames,
            progress=progress,
        )
        detection = detect(spectrum, handle.meta, spectrogram)
        meta = handle.meta
    return AnalysisResult(
        meta=meta,
        spectrum=spectrum,
        spectrogram=spectrogram,
        detection=detection,
        elapsed_seconds=time.perf_counter() - started,
    )


def analyze_file(path: str, progress: ProgressFn | None = None) -> AnalysisResult:
    """Читает файл, считает спектр/спектрограмму и выносит вердикт.

    При ошибке чтения через libsndfile (например, ``psf_fseek()`` на путях
    с эмодзи) анализ один раз повторяется с принудительным декодированием
    через ffmpeg.
    """
    try:
        return _analyze_once(path, progress, force_ffmpeg=False)
    except AudioLoadError:
        raise
    except (RuntimeError, OSError):
        # Не бесконечный retry: вторая попытка идёт сразу через ffmpeg.
        return _analyze_once(path, progress, force_ffmpeg=True)
