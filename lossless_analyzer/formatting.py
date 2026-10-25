"""Вспомогательные функции форматирования значений для интерфейса и отчётов."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .loader import AudioMeta


def format_duration(seconds: float) -> str:
    """Форматирует длительность в виде ЧЧ:ММ:СС (часы опускаются для коротких файлов)."""
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def format_khz(freq_hz: float | None, digits: int = 2) -> str:
    """Возвращает частоту в килогерцах, например «18.05 кГц»."""
    if freq_hz is None:
        return "—"
    return f"{freq_hz / 1000.0:.{digits}f} кГц"


def format_channels(count: int) -> str:
    """Склоняет слово «канал» для подписи метаданных."""
    if count == 1:
        return "1 канал"
    if 2 <= count <= 4:
        return f"{count} канала"
    return f"{count} каналов"


def format_bitrate(kbps: float | None) -> str:
    """Форматирует битрейт в кбит/с."""
    if not kbps or kbps <= 0:
        return "—"
    if kbps >= 1000:
        return f"{kbps / 1000.0:.2f} Мбит/с"
    return f"{kbps:.0f} кбит/с"


def format_audio_meta(meta: "AudioMeta") -> str:
    """Компактная строка технических параметров файла (без дублей)."""
    parts = [meta.container]
    if meta.is_lossy:
        if meta.codec and meta.codec.upper() != meta.container.upper():
            parts.append(meta.codec)
    elif meta.bit_depth:
        parts.append(f"{meta.bit_depth} бит")
        codec = (meta.codec or "").upper()
        if codec and codec != meta.container.upper() and not codec.startswith("PCM"):
            parts.append(meta.codec)
    elif meta.subtype:
        parts.append(meta.subtype.replace("_", " "))
    parts.extend(
        [
            format_channels(meta.channels),
            f"{meta.sample_rate / 1000.0:.1f} кГц",
            f"Найквист {meta.nyquist / 1000.0:.2f} кГц",
            format_duration(meta.duration),
        ]
    )
    if meta.is_lossy and meta.bitrate_kbps:
        parts.append(format_bitrate(meta.bitrate_kbps))
    return " · ".join(part for part in parts if part)
