"""Юнит-тесты детектора на синтетических спектрах."""

from __future__ import annotations

import unittest

import numpy as np

from lossless_analyzer.detector import detect
from lossless_analyzer.dsp import SpectrogramData, SpectrumData
from lossless_analyzer.loader import AudioMeta


def _meta(sample_rate: int, lossy: bool = False, codec: str = "FLAC") -> AudioMeta:
    return AudioMeta(
        path="synthetic.flac",
        sample_rate=sample_rate,
        channels=2,
        frames=sample_rate * 10,
        duration=10.0,
        container="FLAC",
        subtype="PCM_24",
        bit_depth=24,
        codec=codec,
        bitrate_kbps=None,
        is_lossy=lossy,
    )


def _spectrum(
    sample_rate: int = 44_100,
    cutoff: float | None = None,
    slope_db_per_decade: float = 6.0,
    floor_db: float = -105.0,
    seed: int = 7,
) -> SpectrumData:
    """Строит правдоподобный усреднённый спектр музыкального шума."""
    n_fft = 16_384 if sample_rate > 48_000 else 8_192
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    levels = -28.0 - slope_db_per_decade * np.log10(np.maximum(freqs, 80.0) / 1000.0)
    if cutoff is not None:
        levels[freqs > cutoff] = floor_db
    rng = np.random.default_rng(seed)
    levels = levels + rng.normal(0.0, 1.2, levels.size)
    return SpectrumData(
        freqs=freqs,
        levels_db=levels,
        n_fft=n_fft,
        frames_used=1200,
        nyquist=sample_rate / 2.0,
        rms_db=-18.0,
        peak_db=-1.0,
    )


def _soft_spectrum(
    sample_rate: int,
    start_hz: float,
    end_hz: float,
    floor_db: float = -120.0,
) -> SpectrumData:
    """Строит спектр с плавным срезом и ровной полкой выше него."""
    spectrum = _spectrum(sample_rate=sample_rate)
    freqs = spectrum.freqs
    levels = np.full_like(freqs, -30.0)
    mask = (freqs >= start_hz) & (freqs <= end_hz)
    levels[mask] = -30.0 - 90.0 * 0.5 * (
        1.0 - np.cos(np.pi * (freqs[mask] - start_hz) / (end_hz - start_hz))
    )
    levels[freqs > end_hz] = floor_db
    spectrum.levels_db = levels
    return spectrum


def _time_spectrogram(
    sample_rate: int,
    cutoffs: list[float | None],
    floor_db: float = -100.0,
    seed: int = 13,
) -> SpectrogramData:
    """Спектрограмма с разным срезом в разных кадрах."""
    n_fft = 2048
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    base = -30.0 - 6.0 * np.log10(np.maximum(freqs, 80.0) / 1000.0)
    rng = np.random.default_rng(seed)
    columns = []
    for cutoff in cutoffs:
        levels = base.copy()
        if cutoff is not None:
            levels[freqs > cutoff] = floor_db
        columns.append(levels + rng.normal(0.0, 0.8, levels.size))
    matrix = np.column_stack(columns).astype(np.float32)
    return SpectrogramData(
        times=np.arange(len(cutoffs), dtype=np.float64) * 0.05,
        freqs=freqs,
        levels_db=matrix,
        max_db=float(matrix.max()),
        nyquist=sample_rate / 2.0,
    )


class DetectorTests(unittest.TestCase):
    def test_brickwall_is_detected(self) -> None:
        spectrum = _spectrum(cutoff=18_000.0)
        result = detect(spectrum, _meta(44_100))
        self.assertEqual(result.verdict, "fake")
        self.assertIsNotNone(result.cutoff_hz)
        self.assertAlmostEqual(result.cutoff_hz / 1000.0, 18.0, delta=0.25)
        self.assertGreater(result.drop_db or 0, 40)
        self.assertIn("192", result.estimated_source or "")

    def test_full_band_is_lossless(self) -> None:
        spectrum = _spectrum()
        result = detect(spectrum, _meta(44_100))
        self.assertEqual(result.verdict, "lossless")
        self.assertIsNone(result.cutoff_hz)

    def test_dark_spectrum_is_inconclusive(self) -> None:
        spectrum = _spectrum(slope_db_per_decade=60.0)
        result = detect(spectrum, _meta(44_100))
        self.assertEqual(result.verdict, "inconclusive")

    def test_quiet_file_is_inconclusive(self) -> None:
        spectrum = _spectrum()
        spectrum.rms_db = -75.0
        spectrum.peak_db = -55.0
        result = detect(spectrum, _meta(44_100))
        self.assertEqual(result.verdict, "inconclusive")

    def test_upscaled_hi_res_is_detected(self) -> None:
        spectrum = _spectrum(sample_rate=96_000, cutoff=22_050.0, floor_db=-140.0)
        result = detect(spectrum, _meta(96_000))
        self.assertEqual(result.verdict, "fake")
        self.assertAlmostEqual(result.cutoff_hz / 1000.0, 22.05, delta=0.3)
        self.assertIn("44.1", result.estimated_source or "")

    def test_soft_48k_cutoff_is_detected(self) -> None:
        spectrum = _soft_spectrum(48_000, 20_000.0, 23_000.0)
        result = detect(spectrum, _meta(48_000))
        self.assertEqual(result.verdict, "fake", result)
        self.assertIsNotNone(result.cutoff_hz)
        self.assertGreater(result.cutoff_hz, 20_500.0)
        self.assertLess(result.cutoff_hz, 23_500.0)
        self.assertIsNotNone(result.estimated_source)

    def test_soft_44k_cutoff_is_detected(self) -> None:
        spectrum = _soft_spectrum(44_100, 16_000.0, 18_500.0)
        result = detect(spectrum, _meta(44_100))
        self.assertEqual(result.verdict, "fake", result)
        self.assertIsNotNone(result.cutoff_hz)
        self.assertGreater(result.cutoff_hz, 16_000.0)
        self.assertLess(result.cutoff_hz, 19_000.0)
        self.assertIn("MP3", result.estimated_source or "")

    def test_natural_rolloff_is_not_a_cutoff(self) -> None:
        spectrum = _spectrum(sample_rate=48_000)
        freqs = spectrum.freqs
        spectrum.levels_db = np.interp(
            freqs,
            [0.0, 10_000.0, 24_000.0],
            [-30.0, -30.0, -80.0],
        )
        result = detect(spectrum, _meta(48_000))
        self.assertNotEqual(result.verdict, "fake")

    def test_stable_cutoff_is_confirmed_over_time(self) -> None:
        spectrum = _spectrum(cutoff=18_000.0)
        spectrogram = _time_spectrogram(44_100, [18_000.0] * 40)
        result = detect(spectrum, _meta(44_100), spectrogram)
        self.assertEqual(result.verdict, "fake")
        self.assertGreaterEqual(result.time_consistency or 0.0, 0.9)
        self.assertLess(abs((result.time_edge_p90_hz or 0.0) - 18_000.0), 800.0)
        self.assertTrue(any("времени" in note for note in result.notes))

    def test_variable_cutoff_gets_temporal_warning(self) -> None:
        spectrum = _spectrum(cutoff=18_000.0)
        cutoffs = [15_500.0 if i % 2 else 21_000.0 for i in range(40)]
        spectrogram = _time_spectrogram(44_100, cutoffs)
        result = detect(spectrum, _meta(44_100), spectrogram)
        self.assertEqual(result.verdict, "fake")
        self.assertIsNotNone(result.time_consistency)
        self.assertLess(result.time_consistency, 0.75)
        self.assertGreater(result.time_above_fraction or 0.0, 0.25)
        self.assertTrue(any("непостоян" in note for note in result.notes))

    def test_partially_full_band_file_is_inconclusive(self) -> None:
        spectrum = _spectrum()
        spectrogram = _time_spectrogram(44_100, [None] * 25 + [18_000.0] * 15)
        result = detect(spectrum, _meta(44_100), spectrogram)
        self.assertEqual(result.verdict, "inconclusive", result)
        self.assertLess(result.time_edge_p10_hz or 0.0, 19_000.0)
        self.assertIn("неоднород", result.reason)

    def test_lossy_container_overrides_verdict(self) -> None:
        spectrum = _spectrum(cutoff=18_000.0)
        result = detect(spectrum, _meta(44_100, lossy=True, codec="MP3"))
        self.assertEqual(result.verdict, "lossy")
        self.assertEqual(result.headline, "LOSSY SOURCE")

    def test_gentle_rolloff_is_lossless(self) -> None:
        spectrum = _spectrum()
        freqs = spectrum.freqs
        levels = np.full_like(freqs, -30.0)
        mask = freqs > 20_000.0
        levels[mask] = -30.0 - 22.0 * np.log2(np.maximum(freqs[mask] / 20_000.0, 1.0))
        spectrum.levels_db = levels
        result = detect(spectrum, _meta(44_100))
        self.assertEqual(result.verdict, "lossless")

    def test_narrow_notch_is_not_a_cutoff(self) -> None:
        spectrum = _spectrum()
        freqs = spectrum.freqs
        levels = np.full_like(freqs, -30.0)
        levels[(freqs > 15_000.0) & (freqs < 15_500.0)] = -90.0
        spectrum.levels_db = levels
        result = detect(spectrum, _meta(44_100))
        self.assertEqual(result.verdict, "lossless")


if __name__ == "__main__":
    unittest.main()
