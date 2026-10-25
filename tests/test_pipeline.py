"""Интеграционные тесты: генерация аудио → чтение → спектр → вердикт."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample

from lossless_analyzer.analysis import analyze_file

SR = 44_100
SECONDS = 4.0


def _noise(
    seconds: float = SECONDS, sample_rate: int = SR, seed: int = 42
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = int(seconds * sample_rate)
    noise = rng.normal(0.0, 0.11, samples)
    t = np.arange(samples) / sample_rate
    return (noise + 0.03 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float64)


def _write(path: Path, data: np.ndarray, sample_rate: int) -> None:
    sf.write(str(path), data, sample_rate, subtype="PCM_24")


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="lossless_tests_")
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_true_lossless_audio(self) -> None:
        path = self.tmp / "true_lossless.flac"
        _write(path, _noise(), SR)
        result = analyze_file(str(path))
        self.assertEqual(result.detection.verdict, "lossless", result.detection)
        self.assertIsNone(result.detection.cutoff_hz)
        self.assertEqual(result.meta.sample_rate, SR)

    def test_brickwalled_flac_is_detected(self) -> None:
        path = self.tmp / "fake_lossless.flac"
        data = _noise()
        spectrum = np.fft.rfft(data)
        freqs = np.fft.rfftfreq(data.size, d=1.0 / SR)
        spectrum[freqs > 18_000.0] = 0.0
        filtered = np.fft.irfft(spectrum, n=data.size)
        _write(path, filtered, SR)

        result = analyze_file(str(path))
        self.assertEqual(result.detection.verdict, "fake", result.detection)
        self.assertIsNotNone(result.detection.cutoff_hz)
        self.assertAlmostEqual(result.detection.cutoff_hz / 1000.0, 18.0, delta=0.35)
        self.assertGreater(result.detection.drop_db or 0.0, 30.0)
        self.assertEqual(result.spectrogram.levels_db.ndim, 2)

    def test_upscaled_cd_to_96k_is_detected(self) -> None:
        path = self.tmp / "upscaled_96k.wav"
        source = _noise()
        upsampled = resample(source, int(round(source.size * 96_000 / SR)))
        _write(path, upsampled, 96_000)

        result = analyze_file(str(path))
        self.assertEqual(result.detection.verdict, "fake", result.detection)
        self.assertAlmostEqual(result.detection.cutoff_hz / 1000.0, 22.05, delta=0.45)
        self.assertIn("44.1", result.detection.estimated_source or "")

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg не найден")
    def test_real_mp3_upscaled_to_flac(self) -> None:
        source = self.tmp / "source.wav"
        mp3 = self.tmp / "lossy.mp3"
        flac = self.tmp / "mp3_inside.flac"
        _write(source, _noise(seed=9), SR)

        encode = subprocess.run(
            [
                shutil.which("ffmpeg"),
                "-y",
                "-v",
                "error",
                "-i",
                str(source),
                "-c:a",
                "libmp3lame",
                "-b:a",
                "192k",
                str(mp3),
            ],
            capture_output=True,
        )
        if encode.returncode != 0:
            self.skipTest("libmp3lame недоступен в ffmpeg")

        data, sample_rate = sf.read(str(mp3), always_2d=True)
        sf.write(str(flac), data, sample_rate, subtype="PCM_24")

        result = analyze_file(str(flac))
        self.assertFalse(result.meta.is_lossy)
        self.assertEqual(result.detection.verdict, "fake", result.detection)
        cutoff_khz = (result.detection.cutoff_hz or 0.0) / 1000.0
        self.assertGreater(cutoff_khz, 15.0)
        self.assertLess(cutoff_khz, 21.0)


if __name__ == "__main__":
    unittest.main()
