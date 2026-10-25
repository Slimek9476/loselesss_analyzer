"""Оркестрация анализа одного файла."""

from __future__ import annotations

import time
from dataclasses import dataclass

from .detector import DetectionResult, detect
from .dsp import ProgressFn, SpectrogramData, SpectrumData, compute_spectra
from .loader import AudioMeta, open_audio


@dataclass
class AnalysisResult:
    """Всё, что нужно интерфейсу для отображения одного файла."""

    meta: AudioMeta
    spectrum: SpectrumData
    spectrogram: SpectrogramData
    detection: DetectionResult
    elapsed_seconds: float


def analyze_file(path: str, progress: ProgressFn | None = None) -> AnalysisResult:
    """Читает файл, считает спектр/спектрограмму и выносит вердикт."""
    started = time.perf_counter()
    with open_audio(path) as handle:
        spectrum, spectrogram = compute_spectra(
            handle.file,
            handle.meta.sample_rate,
            handle.meta.frames,
            progress=progress,
        )
        detection = detect(spectrum, handle.meta)
        meta = handle.meta
    return AnalysisResult(
        meta=meta,
        spectrum=spectrum,
        spectrogram=spectrogram,
        detection=detection,
        elapsed_seconds=time.perf_counter() - started,
    )
