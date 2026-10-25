"""Отрисовка графиков и экспорт отчёта.

Используется только matplotlib (без pyplot): фигуры встраиваются в Qt
напрямую через FigureCanvasQTAgg. Цвета согласованы с тёмной темой GUI.
"""

from __future__ import annotations

import os
import textwrap
from datetime import datetime
from pathlib import Path

os.environ.setdefault("QT_API", "pyside6")

import matplotlib

matplotlib.use("QtAgg")

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QSizePolicy

from .analysis import AnalysisResult
from .formatting import format_audio_meta

# --- Палитра -------------------------------------------------------------
BG = "#0e1318"
PANEL = "#131a22"
GRID = "#2a3745"
FG = "#e6eef7"
MUTED = "#8a9bad"
ACCENT = "#4cc2ff"
GOOD = "#46d98b"
FAKE = "#ff5c6c"
WARN = "#ffc857"
ORANGE = "#ff9f43"
CUTOFF_LINE = "#7df9ff"

VERDICT_COLORS = {
    "lossless": GOOD,
    "fake": FAKE,
    "inconclusive": WARN,
    "lossy": ORANGE,
}

matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.edgecolor": GRID,
        "axes.labelcolor": FG,
        "text.color": FG,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "savefig.facecolor": BG,
    }
)


def style_axes(ax) -> None:
    """Приводит оси к тёмному оформлению приложения."""
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=9, width=0.6, length=3)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(color=GRID, linewidth=0.5, alpha=0.45)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)
    ax.title.set_color(FG)


def style_colorbar(colorbar) -> None:
    colorbar.outline.set_edgecolor(GRID)
    colorbar.ax.yaxis.set_tick_params(color=GRID, labelcolor=MUTED, labelsize=8)
    colorbar.set_label("дБ", color=MUTED, fontsize=8)


def _spectrum_bounds(levels: np.ndarray) -> tuple[float, float]:
    finite = levels[np.isfinite(levels)]
    if finite.size == 0:
        return -130.0, 5.0
    floor = float(np.percentile(finite, 0.5))
    ymin = float(np.clip(floor - 8.0, -140.0, -70.0))
    return ymin, 6.0


def draw_spectrum(
    ax, result: AnalysisResult, show_title: bool = False, focus_high: bool = False
) -> None:
    """Рисует усреднённый спектр с отметками Найквиста и среза."""
    spectrum = result.spectrum
    freqs_khz = spectrum.freqs / 1000.0
    levels = spectrum.levels_db
    ymin, ymax = _spectrum_bounds(levels)
    clipped = np.maximum(levels, ymin)

    ax.plot(freqs_khz, clipped, color=ACCENT, linewidth=0.85, zorder=2)
    ax.fill_between(freqs_khz, clipped, ymin, color=ACCENT, alpha=0.10, linewidth=0)

    nyquist_khz = spectrum.nyquist / 1000.0
    if focus_high:
        ax.set_xlim(min(10.0, nyquist_khz * 0.5), nyquist_khz)
    else:
        ax.set_xlim(0, nyquist_khz)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Частота, кГц")
    ax.set_ylabel("Уровень, дБ")
    style_axes(ax)

    ax.text(
        0.995,
        0.97,
        f"Найквист {nyquist_khz:.2f} кГц",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=MUTED,
        fontsize=8,
    )

    cutoff = result.detection.cutoff_hz
    if cutoff is not None:
        cutoff_khz = cutoff / 1000.0
        ax.axvline(
            cutoff_khz, color=FAKE, linewidth=1.2, linestyle=(0, (4, 3)), zorder=3
        )
        ax.annotate(
            f"срез {cutoff_khz:.2f} кГц",
            xy=(cutoff_khz, 0.70),
            xycoords=("data", "axes fraction"),
            xytext=(-7, 0),
            textcoords="offset points",
            color=FAKE,
            fontsize=8,
            rotation=90,
            ha="right",
            va="center",
        )

    if show_title:
        ax.set_title("Усреднённый спектр", loc="left", fontsize=10, pad=7)


def draw_spectrogram(ax, result: AnalysisResult, show_title: bool = False):
    """Рисует спектрограмму. Возвращает объект изображения для colorbar."""
    spectrogram = result.spectrogram
    data = spectrogram.levels_db
    if data.size == 0:
        ax.text(
            0.5,
            0.5,
            "Нет данных",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=MUTED,
        )
        style_axes(ax)
        return None

    finite = data[np.isfinite(data)]
    vmax = float(finite.max()) if finite.size else 0.0
    vmin = max(vmax - 90.0, -140.0)
    duration = max(result.meta.duration, 1e-3)

    image = ax.imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=(0.0, duration, 0.0, spectrogram.nyquist / 1000.0),
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        rasterized=True,
    )
    ax.set_xlim(0, duration)
    ax.set_ylim(0, spectrogram.nyquist / 1000.0)
    ax.set_xlabel("Время, с")
    ax.set_ylabel("Частота, кГц")
    style_axes(ax)
    ax.grid(False)

    cutoff = result.detection.cutoff_hz
    if cutoff is not None:
        cutoff_khz = cutoff / 1000.0
        ax.axhline(
            cutoff_khz,
            color=CUTOFF_LINE,
            linewidth=1.0,
            linestyle=(0, (4, 3)),
            alpha=0.9,
        )
        ax.text(
            duration * 0.004,
            cutoff_khz + spectrogram.nyquist * 0.008 / 1000.0,
            f"срез {cutoff_khz:.2f} кГц",
            color=CUTOFF_LINE,
            fontsize=8,
            va="bottom",
        )

    if show_title:
        ax.set_title("Спектрограмма", loc="left", fontsize=10, pad=7)
    return image


class _BaseCanvas(FigureCanvasQTAgg):
    """Базовый холст: фигура, сброс осей, заглушка."""

    def __init__(self, parent=None, figsize=(7, 3.0)):
        self.figure = Figure(figsize=figsize, dpi=100, facecolor=BG)
        super().__init__(self.figure)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(f"background: {PANEL};")
        self.result: AnalysisResult | None = None

    def _reset(self):
        self.figure.clear()
        self.figure.subplots_adjust(left=0.075, right=0.985, top=0.94, bottom=0.17)
        ax = self.figure.add_subplot(111)
        style_axes(ax)
        return ax

    def show_placeholder(self, text: str) -> None:
        self.result = None
        ax = self._reset()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        ax.text(
            0.5,
            0.5,
            text,
            transform=ax.transAxes,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=10,
        )
        self.draw_idle()


class SpectrumCanvas(_BaseCanvas):
    """Холст усреднённого спектра."""

    def __init__(self, parent=None):
        super().__init__(parent, figsize=(7, 2.8))
        self.focus_high = False
        self.show_placeholder("Спектр появится после анализа файла")

    def show_result(self, result: AnalysisResult) -> None:
        self.result = result
        ax = self._reset()
        draw_spectrum(ax, result, focus_high=self.focus_high)
        self.draw_idle()

    def set_focus_high(self, enabled: bool) -> None:
        self.focus_high = enabled
        if self.result is not None:
            self.show_result(self.result)


class SpectrogramCanvas(_BaseCanvas):
    """Холст спектрограммы с colorbar."""

    def __init__(self, parent=None):
        super().__init__(parent, figsize=(7, 3.6))
        self.show_placeholder("Спектрограмма появится после анализа файла")

    def show_result(self, result: AnalysisResult) -> None:
        self.result = result
        self.figure.clear()
        self.figure.subplots_adjust(left=0.075, right=0.94, top=0.94, bottom=0.17)
        ax = self.figure.add_subplot(111)
        image = draw_spectrogram(ax, result)
        if image is not None:
            colorbar = self.figure.colorbar(image, ax=ax, pad=0.012, fraction=0.035)
            style_colorbar(colorbar)
        self.draw_idle()


def build_report(result: AnalysisResult, destination: str | Path) -> None:
    """Сохраняет PNG/PDF-отчёт: вердикт + спектр + спектрограмма."""
    detection = result.detection
    color = VERDICT_COLORS.get(detection.verdict, FG)

    figure = Figure(figsize=(14, 9.5), dpi=150, facecolor=BG)
    grid = figure.add_gridspec(
        2,
        1,
        height_ratios=[1.0, 1.35],
        hspace=0.33,
        left=0.075,
        right=0.92,
        top=0.775,
        bottom=0.065,
    )
    ax_spectrum = figure.add_subplot(grid[0])
    ax_spectrogram = figure.add_subplot(grid[1])
    draw_spectrum(ax_spectrum, result, show_title=True)
    image = draw_spectrogram(ax_spectrogram, result, show_title=True)
    if image is not None:
        colorbar = figure.colorbar(image, ax=ax_spectrogram, pad=0.01, fraction=0.035)
        style_colorbar(colorbar)

    name = Path(result.meta.path).name
    if len(name) > 80:
        name = name[:77] + "…"

    figure.text(
        0.075,
        0.972,
        "LOSSLESS ANALYZER · ОТЧЁТ",
        color=MUTED,
        fontsize=9,
        weight="bold",
    )
    figure.text(
        0.075, 0.930, detection.headline, color=color, fontsize=25, weight="bold"
    )
    figure.text(
        0.075,
        0.892,
        textwrap.fill(detection.reason, width=105),
        color=FG,
        fontsize=10,
        va="top",
        linespacing=1.4,
    )
    figure.text(
        0.075,
        0.840,
        f"{name} · {format_audio_meta(result.meta)}",
        color=MUTED,
        fontsize=8.5,
    )
    if detection.estimated_source:
        figure.text(
            0.075,
            0.815,
            f"Вероятный исходный формат: {detection.estimated_source}",
            color=FG,
            fontsize=8.5,
        )
    if detection.cutoff_hz is not None:
        confidence = (
            f" · уверенность {detection.confidence * 100:.0f}%"
            if detection.confidence
            else ""
        )
        figure.text(
            0.92,
            0.930,
            f"Срез {detection.cutoff_hz / 1000.0:.2f} кГц{confidence}",
            color=color,
            fontsize=10,
            ha="right",
            va="center",
        )

    stamp = datetime.now().strftime("%d.%m.%Y %H:%M")
    figure.text(
        0.075,
        0.028,
        f"Сформировано {stamp} · Lossless Analyzer",
        color=MUTED,
        fontsize=8,
    )

    figure.savefig(destination, facecolor=BG)
    figure.clear()
