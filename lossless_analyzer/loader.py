"""Чтение аудиофайлов и метаданных.

Основной декодер — libsndfile через пакет ``soundfile``. Если формат не
поддерживается (например, M4A/ALAC/AAC/WMA), файл автоматически
конвертируется во временный WAV при помощи ffmpeg.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import soundfile as sf

# Расширения, которые программа готова принимать.
SUPPORTED_EXTENSIONS = {
    ".wav",
    ".wave",
    ".flac",
    ".aif",
    ".aiff",
    ".aifc",
    ".ogg",
    ".oga",
    ".mp3",
    ".m4a",
    ".mp4",
    ".alac",
    ".aac",
    ".wma",
    ".wv",
    ".ape",
    ".opus",
    ".caf",
    ".w64",
}

# Кодеки, которые заведомо используют сжатие с потерями.
LOSSY_CODECS = (
    "mp3",
    "mp2",
    "aac",
    "vorbis",
    "opus",
    "wmav1",
    "wmav2",
    "wmapro",
    "ac3",
    "eac3",
    "atrac",
    "mpc",
    "cook",
    "qdm2",
    "dts",
    "gsm",
    "musepack",
)

# Соответствие подтипов libsndfile разрядности.
_SUBTYPE_BITS = {
    "PCM_S8": 8,
    "PCM_U8": 8,
    "PCM_16": 16,
    "PCM_24": 24,
    "PCM_32": 32,
    "FLOAT": 32,
    "DOUBLE": 64,
    "ALAW": 8,
    "ULAW": 8,
    "IMA_ADPCM": 4,
    "MS_ADPCM": 4,
}

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0


class AudioLoadError(RuntimeError):
    """Ошибка чтения аудиофайла."""


@dataclass(frozen=True)
class AudioMeta:
    """Технические сведения об аудиофайле."""

    path: str
    sample_rate: int
    channels: int
    frames: int
    duration: float
    container: str
    subtype: str
    bit_depth: int | None
    codec: str
    bitrate_kbps: float | None
    is_lossy: bool

    @property
    def nyquist(self) -> float:
        return self.sample_rate / 2.0


@dataclass
class AudioHandle:
    """Открытый поток libsndfile вместе с метаданными."""

    file: sf.SoundFile
    meta: AudioMeta


def _decode_with_ffmpeg(source: Path, destination: Path) -> None:
    """Декодирует любой поддерживаемый ffmpeg файл в float32 WAV без потерь."""
    exe = shutil.which("ffmpeg")
    if not exe:
        raise AudioLoadError(
            "Формат не поддерживается libsndfile, а ffmpeg не найден. "
            "Установите ffmpeg и добавьте его в PATH."
        )
    command = [
        exe,
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        str(source),
        "-map",
        "0:a:0",
        "-vn",
        "-c:a",
        "pcm_f32le",
        str(destination),
    ]
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600,
            creationflags=_CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AudioLoadError(f"ffmpeg не смог обработать файл: {exc}") from exc
    if process.returncode != 0 or not destination.exists():
        message = (process.stderr or "").strip().splitlines()
        detail = message[-1] if message else "неизвестная ошибка ffmpeg"
        raise AudioLoadError(f"ffmpeg не смог декодировать файл: {detail}")


def _ffprobe(path: Path) -> dict:
    """Возвращает JSON-описание первого аудиопотока (если доступен ffprobe)."""
    exe = shutil.which("ffprobe")
    if not exe:
        return {}
    command = [
        exe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_name,codec_long_name,sample_rate,channels,"
        "bits_per_sample,bits_per_raw_sample,bit_rate,duration,sample_fmt",
        "-show_entries",
        "format=duration,bit_rate,format_name",
        "-of",
        "json",
        str(path),
    ]
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30,
            creationflags=_CREATE_NO_WINDOW,
        )
        if process.returncode != 0:
            return {}
        data = json.loads(process.stdout or "{}")
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {}
    streams = data.get("streams") or []
    result = dict(streams[0]) if streams else {}
    result["_format"] = data.get("format") or {}
    return result


def _to_int(value) -> int | None:
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _to_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _bit_depth_from_subtype(subtype: str) -> int | None:
    if not subtype:
        return None
    if subtype in _SUBTYPE_BITS:
        return _SUBTYPE_BITS[subtype]
    for name, bits in _SUBTYPE_BITS.items():
        if subtype.startswith(name):
            return bits
    return None


def _build_meta(
    original: Path,
    handle: sf.SoundFile,
    info: sf._SoundFileInfo | None,
    probe: dict,
) -> AudioMeta:
    """Собирает метаданные, объединяя данные libsndfile и ffprobe."""
    codec_raw = str(probe.get("codec_name") or "").lower()
    codec_long = str(probe.get("codec_long_name") or "")

    if info is not None:
        container = info.format or original.suffix.lstrip(".").upper()
        subtype = info.subtype or ""
        bit_depth = _bit_depth_from_subtype(subtype)
        codec = codec_raw or (container.lower() if container else "")
    else:
        container = original.suffix.lstrip(".").upper() or "FFMPEG"
        subtype = str(probe.get("sample_fmt") or "")
        bit_depth = _to_int(probe.get("bits_per_raw_sample")) or _to_int(
            probe.get("bits_per_sample")
        )
        codec = codec_raw

    if probe.get("bits_per_raw_sample") and bit_depth is None:
        bit_depth = _to_int(probe.get("bits_per_raw_sample"))

    bitrate_bps = _to_float(probe.get("bit_rate"))
    if bitrate_bps is None:
        bitrate_bps = _to_float((probe.get("_format") or {}).get("bit_rate"))
    bitrate_kbps = bitrate_bps / 1000.0 if bitrate_bps else None

    is_lossy = bool(codec) and any(codec.startswith(item) for item in LOSSY_CODECS)
    if not is_lossy:
        subtype_upper = subtype.upper()
        if (
            "MPEG" in subtype_upper
            or "VORBIS" in subtype_upper
            or "OPUS" in subtype_upper
        ):
            is_lossy = True
    if not is_lossy:
        is_lossy = original.suffix.lower() in {".mp3", ".aac", ".wma", ".opus", ".mp2"}

    duration = handle.frames / handle.samplerate if handle.samplerate else 0.0
    if duration <= 0:
        duration = _to_float(probe.get("duration")) or 0.0

    if not codec:
        codec = codec_long or container

    return AudioMeta(
        path=str(original),
        sample_rate=handle.samplerate,
        channels=handle.channels,
        frames=handle.frames,
        duration=duration,
        container=container.upper(),
        subtype=subtype,
        bit_depth=bit_depth,
        codec=codec.upper() if codec else "UNKNOWN",
        bitrate_kbps=bitrate_kbps,
        is_lossy=is_lossy,
    )


@contextlib.contextmanager
def open_audio(path: str | Path) -> Iterator[AudioHandle]:
    """Открывает файл для чтения, при необходимости декодируя его через ffmpeg.

    Временный WAV удаляется автоматически после выхода из контекста.
    """
    original = Path(path)
    if not original.is_file():
        raise AudioLoadError(f"Файл не найден: {original}")

    probe = _ffprobe(original)
    tempdir: tempfile.TemporaryDirectory | None = None
    info: sf._SoundFileInfo | None = None
    handle: sf.SoundFile | None = None

    try:
        info = sf.info(str(original))
        handle = sf.SoundFile(str(original))
    except Exception:
        # libsndfile не умеет этот формат — пробуем ffmpeg.
        handle = None
        info = None
        tempdir = tempfile.TemporaryDirectory(prefix="lossless_analyzer_")
        decoded = Path(tempdir.name) / "decoded.wav"
        try:
            _decode_with_ffmpeg(original, decoded)
            handle = sf.SoundFile(str(decoded))
        except Exception as exc:
            tempdir.cleanup()
            if isinstance(exc, AudioLoadError):
                raise
            raise AudioLoadError(
                f"Не удалось открыть аудиофайл «{original.name}»: {exc}"
            ) from exc

    try:
        if handle.frames < 256:
            raise AudioLoadError("Файл слишком короткий для анализа.")
        meta = _build_meta(original, handle, info, probe)
        yield AudioHandle(file=handle, meta=meta)
    finally:
        try:
            handle.close()
        finally:
            if tempdir is not None:
                tempdir.cleanup()
