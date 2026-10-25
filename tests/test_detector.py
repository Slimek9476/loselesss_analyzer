"""Юнит-тесты детектора на синтетических спектрах."""

from __future__ import annotations

import unittest

import numpy as np

from lossless_analyzer.detector import detect
from lossless_analyzer.dsp import SpectrumData
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
