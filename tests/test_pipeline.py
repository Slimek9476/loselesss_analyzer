"""Интеграционные тесты: генерация аудио → чтение → спектр → вердикт."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf
from scipy.signal import resample

from lossless_analyzer.analysis import analyze_file
from lossless_analyzer.loader import open_audio

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
        self.assertGreater(result.detection.time_frames, 0)
        self.assertGreater(result.detection.time_edge_p10_hz or 0.0, SR / 2.0 - 1_000.0)
        self.assertEqual(result.spectrum.grid_bits, 24)
        self.assertIsNone(result.detection.effective_bits)

    def test_antiphase_stereo_is_analyzed(self) -> None:
        base = _noise(seed=12)
        stereo = np.column_stack((base, -base))
        path = self.tmp / "antiphase.wav"
        sf.write(str(path), stereo, SR, subtype="PCM_24")
        result = analyze_file(str(path))
        self.assertEqual(result.detection.verdict, "lossless", result.detection)
        self.assertGreater(result.spectrum.rms_db, -30.0)

    def test_effective_bit_depth_is_reported(self) -> None:
        base = _noise(seed=14)
        quantized_16 = np.round(base * 32_768) / 32_768
        path = self.tmp / "16bit_in_24bit.flac"
        sf.write(str(path), quantized_16, SR, subtype="PCM_24")
        result = analyze_file(str(path))
        self.assertEqual(result.spectrum.grid_bits, 16)
        self.assertEqual(result.detection.effective_bits, 16)
        self.assertTrue(
            any("Эффективная разрядность" in note for note in result.detection.notes)
        )

    def test_partially_filtered_file_is_inconclusive(self) -> None:
        base = _noise(seed=15)
        mixed = base.copy()
        second = mixed[base.size // 2 :]
        spectrum = np.fft.rfft(second)
        freqs = np.fft.rfftfreq(second.size, d=1.0 / SR)
        spectrum[freqs > 18_000.0] = 0.0
        mixed[base.size // 2 :] = np.fft.irfft(spectrum, n=second.size)
        path = self.tmp / "mixed_bandwidth.wav"
        _write(path, mixed, SR)
        result = analyze_file(str(path))
        self.assertEqual(result.detection.verdict, "inconclusive", result.detection)
        self.assertLess(result.detection.time_edge_p10_hz or 0.0, 19_000.0)
        self.assertIn("неоднород", result.detection.reason)

    def test_unicode_filename_is_analyzed(self) -> None:
        folder = self.tmp / "Любимое 💙"
        folder.mkdir()
        path = folder / "Falling in Love 💙.flac"
        _write(path, _noise(), SR)
        result = analyze_file(str(path))
        self.assertEqual(result.detection.verdict, "lossless", result.detection)
        self.assertEqual(result.meta.sample_rate, SR)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg не найден")
    def test_libsndfile_failure_falls_back_to_ffmpeg(self) -> None:
        path = self.tmp / "Трек 💙.wav"
        _write(path, _noise(), SR)
        real_open = sf.SoundFile

        def fake_open(target, *args, **kwargs):
            if str(target) == str(path):
                raise RuntimeError("Internal psf_fseek() failed.")
            return real_open(target, *args, **kwargs)

        with mock.patch("lossless_analyzer.loader.sf.SoundFile", side_effect=fake_open):
            with open_audio(str(path)) as handle:
                self.assertTrue(handle.used_fallback)
                self.assertEqual(handle.meta.sample_rate, SR)
            result = analyze_file(str(path))

        self.assertEqual(result.detection.verdict, "lossless", result.detection)

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg не найден")
    def test_unicode_m4a_is_decoded_via_ffmpeg(self) -> None:
        source = self.tmp / "source.wav"
        m4a = self.tmp / "Трек 💙.m4a"
        _write(source, _noise(seed=5), SR)
        encode = subprocess.run(
            [
                shutil.which("ffmpeg"),
                "-y",
                "-v",
                "error",
                "-i",
                str(source),
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(m4a),
            ],
            capture_output=True,
        )
        if encode.returncode != 0:
            self.skipTest("AAC-энкодер ffmpeg недоступен")

        with open_audio(str(m4a)) as handle:
            self.assertTrue(handle.used_fallback)
            self.assertTrue(handle.meta.is_lossy)
            self.assertEqual(handle.meta.sample_rate, SR)

        result = analyze_file(str(m4a))
        self.assertEqual(result.detection.verdict, "lossy", result.detection)

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
        self.assertGreaterEqual(result.detection.time_consistency or 0.0, 0.8)
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
