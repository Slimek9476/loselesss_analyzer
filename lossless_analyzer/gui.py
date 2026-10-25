"""Главное окно приложения на PySide6."""

from __future__ import annotations

import os
from html import escape
from pathlib import Path

os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .analysis import AnalysisResult, analyze_file
from .formatting import format_audio_meta, format_khz
from .loader import SUPPORTED_EXTENSIONS
from .plots import (
    ACCENT,
    BG,
    FAKE,
    FG,
    MUTED,
    PANEL,
    VERDICT_COLORS,
    SpectrumCanvas,
    SpectrogramCanvas,
    build_report,
)

APP_STYLE = f"""
QWidget {{
    background: {BG};
    color: {FG};
    font-family: "Segoe UI";
    font-size: 12px;
}}
QWidget#sidebar {{
    background: #111820;
    border-right: 1px solid #223040;
}}
QLabel#appTitle {{
    font-size: 21px;
    font-weight: 700;
    color: {FG};
}}
QLabel#appSubtitle, QLabel#hint {{
    color: {MUTED};
    font-size: 11px;
}}
QLabel#hint {{
    padding: 8px;
}}
QLabel#fileTitle {{
    font-size: 16px;
    font-weight: 600;
    color: {FG};
}}
QLabel#metaLabel {{
    color: {MUTED};
    font-size: 11px;
}}
QLabel#verdictLabel {{
    font-size: 23px;
    font-weight: 800;
    letter-spacing: 0.3px;
}}
QLabel#confidence {{
    font-size: 11px;
    font-weight: 600;
}}
QLabel#verdictReason {{
    color: #cbd8e5;
    font-size: 12px;
}}
QLabel#details {{
    color: #a6b6c6;
    font-size: 11px;
}}
QLabel#plotCaption {{
    color: {MUTED};
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.6px;
}}
QFrame#verdictCard {{
    background: #141b23;
    border: 1px solid #2b3947;
    border-radius: 10px;
}}
QFrame#plotPanel {{
    background: {PANEL};
    border: 1px solid #223040;
    border-radius: 10px;
}}
QPushButton {{
    background: #1b2632;
    border: 1px solid #2b3b4b;
    border-radius: 7px;
    padding: 7px 12px;
    color: {FG};
}}
QPushButton:hover {{
    background: #223140;
    border-color: #3b5165;
}}
QPushButton:pressed {{
    background: #18222d;
}}
QPushButton:disabled {{
    color: #5c6b7a;
    border-color: #232e39;
    background: #151c24;
}}
QPushButton#primary {{
    background: #1c6fa3;
    border: 1px solid #2a8ac8;
    color: #ffffff;
    font-weight: 600;
}}
QPushButton#primary:hover {{
    background: #2382bf;
}}
QCheckBox {{
    color: {MUTED};
    font-size: 10px;
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 13px;
    height: 13px;
    border-radius: 3px;
    border: 1px solid #3a4b5d;
    background: #0f151c;
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}
QCheckBox:disabled {{
    color: #4c5a68;
}}
QListWidget#fileList {{
    background: #0f161d;
    border: 1px solid #223040;
    border-radius: 8px;
    padding: 5px;
    outline: none;
}}
QListWidget#fileList::item {{
    padding: 8px 9px;
    margin: 1px 0;
    border-radius: 6px;
    color: {FG};
}}
QListWidget#fileList::item:hover {{
    background: #18222c;
}}
QListWidget#fileList::item:selected {{
    background: #1c2b39;
    color: #ffffff;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: #2c3b4b;
    border-radius: 4px;
    min-height: 24px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QSplitter::handle {{
    background: transparent;
}}
QStatusBar {{
    background: #0b1117;
    color: {MUTED};
    border-top: 1px solid #1d2833;
}}
QProgressBar {{
    background: #17212b;
    border: none;
    border-radius: 3px;
    height: 6px;
}}
QProgressBar::chunk {{
    background: {ACCENT};
    border-radius: 3px;
}}
"""


class _Cancelled(Exception):
    """Внутреннее исключение для остановки анализа при закрытии окна."""


class AnalysisWorker(QThread):
    """Последовательно анализирует файлы в фоновом потоке."""

    file_started = Signal(str)
    file_progress = Signal(str, int)
    file_done = Signal(str, object)
    file_failed = Signal(str, str)

    def __init__(self, paths: list[str], parent=None):
        super().__init__(parent)
        self._paths = paths

    def run(self) -> None:  # noqa: D401 - метод Qt
        for path in self._paths:
            if self.isInterruptionRequested():
                break
            self.file_started.emit(path)

            def report(fraction: float, _path: str = path) -> None:
                if self.isInterruptionRequested():
                    raise _Cancelled()
                self.file_progress.emit(_path, int(max(0.0, min(1.0, fraction)) * 100))

            try:
                result = analyze_file(path, report)
            except _Cancelled:
                break
            except Exception as exc:  # noqa: BLE001 - показываем текст ошибки в UI
                message = str(exc).strip() or "Неизвестная ошибка анализа"
                self.file_failed.emit(path, message)
            else:
                self.file_done.emit(path, result)


def _blend(color: str, background: str, alpha: float) -> str:
    """Смешивает два цвета (для мягкой рамки карточки вердикта)."""
    front = QColor(color)
    back = QColor(background)
    red = int(front.red() * alpha + back.red() * (1 - alpha))
    green = int(front.green() * alpha + back.green() * (1 - alpha))
    blue = int(front.blue() * alpha + back.blue() * (1 - alpha))
    return QColor(red, green, blue).name()


class MainWindow(QMainWindow):
    """Окно с очередью файлов, графиками и карточкой вердикта."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lossless Analyzer — проверка апскейлов")
        self.resize(1420, 920)
        self.setMinimumSize(1080, 720)
        self.setAcceptDrops(True)
        self.setStyleSheet(APP_STYLE)

        self._items: dict[str, QListWidgetItem] = {}
        self._results: dict[str, AnalysisResult] = {}
        self._errors: dict[str, str] = {}
        self._pending: list[str] = []
        self._analyzing: set[str] = set()
        self._worker: AnalysisWorker | None = None
        self._current_path: str | None = None
        self._last_dir = str(Path.home())

        self._build_ui()
        self._build_shortcuts()
        self._show_empty()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        central = QWidget()
        central_layout = QHBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(1)
        splitter.addWidget(self._build_sidebar())
        splitter.addWidget(self._build_content())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([330, 1080])
        central_layout.addWidget(splitter)
        self.setCentralWidget(central)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setFixedWidth(220)
        self.progress.setTextVisible(False)
        self.progress.hide()
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage("Готово")

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setMinimumWidth(300)
        sidebar.setMaximumWidth(430)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(18, 20, 18, 16)
        layout.setSpacing(11)

        title = QLabel("Lossless Analyzer")
        title.setObjectName("appTitle")
        subtitle = QLabel("Поиск апскейлов и фейковых FLAC/WAV")
        subtitle.setObjectName("appSubtitle")
        subtitle.setWordWrap(True)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self.add_button = QPushButton("Добавить файлы")
        self.add_button.setObjectName("primary")
        self.add_button.clicked.connect(self._choose_files)
        self.remove_button = QPushButton("Убрать")
        self.remove_button.clicked.connect(self._remove_current)
        self.remove_button.setEnabled(False)
        buttons.addWidget(self.add_button, 1)
        buttons.addWidget(self.remove_button)

        self.file_list = QListWidget()
        self.file_list.setObjectName("fileList")
        self.file_list.setSelectionMode(QListWidget.SingleSelection)
        self.file_list.setTextElideMode(Qt.ElideMiddle)
        self.file_list.setVerticalScrollMode(QListWidget.ScrollPerPixel)
        self.file_list.currentItemChanged.connect(self._on_selection_changed)

        hint = QLabel("Перетащите файлы сюда\nWAV · FLAC · ALAC · MP3 · M4A")
        hint.setObjectName("hint")
        hint.setAlignment(Qt.AlignCenter)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addSpacing(2)
        layout.addLayout(buttons)
        layout.addWidget(self.file_list, 1)
        layout.addWidget(hint)
        return sidebar

    def _build_content(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(22, 18, 22, 14)
        layout.setSpacing(12)

        # --- Заголовок с именем файла ------------------------------------
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(12)
        titles = QVBoxLayout()
        titles.setSpacing(2)
        self.file_label = QLabel("Файл не выбран")
        self.file_label.setObjectName("fileTitle")
        self.meta_label = QLabel("")
        self.meta_label.setObjectName("metaLabel")
        self.meta_label.setWordWrap(True)
        titles.addWidget(self.file_label)
        titles.addWidget(self.meta_label)
        header_layout.addLayout(titles, 1)

        self.export_button = QPushButton("Экспорт отчёта")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export_report)
        header_layout.addWidget(self.export_button, 0, Qt.AlignTop)
        layout.addWidget(header)

        # --- Карточка вердикта -------------------------------------------
        self.verdict_card = QFrame()
        self.verdict_card.setObjectName("verdictCard")
        card_layout = QVBoxLayout(self.verdict_card)
        card_layout.setContentsMargins(18, 14, 18, 14)
        card_layout.setSpacing(5)

        top_row = QHBoxLayout()
        top_row.setSpacing(10)
        self.verdict_label = QLabel("ОЖИДАНИЕ")
        self.verdict_label.setObjectName("verdictLabel")
        self.confidence_label = QLabel("")
        self.confidence_label.setObjectName("confidence")
        top_row.addWidget(self.verdict_label)
        top_row.addStretch(1)
        top_row.addWidget(self.confidence_label, 0, Qt.AlignVCenter)
        card_layout.addLayout(top_row)

        self.reason_label = QLabel("")
        self.reason_label.setObjectName("verdictReason")
        self.reason_label.setWordWrap(True)
        card_layout.addWidget(self.reason_label)

        self.details_label = QLabel("")
        self.details_label.setObjectName("details")
        self.details_label.setWordWrap(True)
        self.details_label.setTextFormat(Qt.RichText)
        self.details_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        card_layout.addWidget(self.details_label)
        layout.addWidget(self.verdict_card)

        # --- Графики ------------------------------------------------------
        plots_splitter = QSplitter(Qt.Vertical)
        plots_splitter.setChildrenCollapsible(False)
        plots_splitter.setHandleWidth(10)

        spectrum_panel = QFrame()
        spectrum_panel.setObjectName("plotPanel")
        spectrum_layout = QVBoxLayout(spectrum_panel)
        spectrum_layout.setContentsMargins(10, 8, 10, 8)
        spectrum_layout.setSpacing(4)
        spectrum_head = QHBoxLayout()
        spectrum_caption = QLabel("УСРЕДНЁННЫЙ СПЕКТР")
        spectrum_caption.setObjectName("plotCaption")
        self.focus_check = QCheckBox("Фокус 10 кГц+")
        self.focus_check.setEnabled(False)
        self.focus_check.setToolTip("Показать только высокочастотную область спектра")
        spectrum_head.addWidget(spectrum_caption)
        spectrum_head.addStretch(1)
        spectrum_head.addWidget(self.focus_check)
        self.spectrum_canvas = SpectrumCanvas()
        self.focus_check.toggled.connect(self.spectrum_canvas.set_focus_high)
        spectrum_layout.addLayout(spectrum_head)
        spectrum_layout.addWidget(self.spectrum_canvas, 1)

        spectrogram_panel = QFrame()
        spectrogram_panel.setObjectName("plotPanel")
        spectrogram_layout = QVBoxLayout(spectrogram_panel)
        spectrogram_layout.setContentsMargins(10, 8, 10, 8)
        spectrogram_layout.setSpacing(4)
        spectrogram_caption = QLabel("СПЕКТРОГРАММА")
        spectrogram_caption.setObjectName("plotCaption")
        self.spectrogram_canvas = SpectrogramCanvas()
        spectrogram_layout.addWidget(spectrogram_caption)
        spectrogram_layout.addWidget(self.spectrogram_canvas, 1)

        plots_splitter.addWidget(spectrum_panel)
        plots_splitter.addWidget(spectrogram_panel)
        plots_splitter.setStretchFactor(0, 4)
        plots_splitter.setStretchFactor(1, 6)
        plots_splitter.setSizes([340, 520])
        layout.addWidget(plots_splitter, 1)
        return content

    def _build_shortcuts(self) -> None:
        open_action = QAction(self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._choose_files)
        self.addAction(open_action)

        export_action = QAction(self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self._export_report)
        self.addAction(export_action)

        remove_action = QAction(self)
        remove_action.setShortcut(QKeySequence("Delete"))
        remove_action.triggered.connect(self._remove_current)
        self.addAction(remove_action)

    # ------------------------------------------------------------- Файлы
    def _choose_files(self) -> None:
        patterns = " ".join(f"*{ext}" for ext in sorted(SUPPORTED_EXTENSIONS))
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Добавить аудиофайлы",
            self._last_dir,
            f"Аудиофайлы ({patterns});;Все файлы (*)",
        )
        if files:
            self._last_dir = str(Path(files[0]).parent)
            self._add_paths(files)

    def _add_paths(self, paths: list[str]) -> None:
        added: list[str] = []
        for raw in paths:
            try:
                path = str(Path(raw).resolve())
            except OSError:
                continue
            suffix = Path(path).suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS or path in self._items:
                continue
            item = QListWidgetItem(Path(path).name)
            item.setData(Qt.UserRole, path)
            item.setToolTip(path)
            item.setForeground(QBrush(QColor(MUTED)))
            self.file_list.addItem(item)
            self._items[path] = item
            self._set_item_status(path, "queued")
            added.append(path)

        if not added:
            return
        self._pending.extend(added)
        self.statusBar().showMessage(f"В очередь добавлено файлов: {len(added)}")
        if self.file_list.currentItem() is None:
            self.file_list.setCurrentItem(self._items[added[0]])
        self._pump_worker()

    def _remove_current(self) -> None:
        item = self.file_list.currentItem()
        if item is None:
            return
        path = item.data(Qt.UserRole)
        row = self.file_list.row(item)
        self.file_list.takeItem(row)
        self._items.pop(path, None)
        self._results.pop(path, None)
        self._errors.pop(path, None)
        self._pending = [p for p in self._pending if p != path]
        self._current_path = None
        if self.file_list.count():
            self.file_list.setCurrentRow(min(row, self.file_list.count() - 1))
        else:
            self.remove_button.setEnabled(False)
            self._show_empty()
            self.statusBar().showMessage("Готово")

    # ---------------------------------------------------------- Очередь
    def _pump_worker(self) -> None:
        if self._worker is not None:
            return
        self._pending = [
            p for p in dict.fromkeys(self._pending) if p not in self._results
        ]
        if not self._pending:
            return
        paths = self._pending
        self._pending = []
        self._worker = AnalysisWorker(paths, self)
        self._worker.file_started.connect(self._on_file_started)
        self._worker.file_progress.connect(self._on_file_progress)
        self._worker.file_done.connect(self._on_file_done)
        self._worker.file_failed.connect(self._on_file_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_file_started(self, path: str) -> None:
        self._analyzing.add(path)
        self._set_item_status(path, "analyzing")
        if self._is_current(path):
            self._show_loading(path)
        self.progress.setValue(0)
        self.progress.show()
        self.statusBar().showMessage(f"Анализ: {Path(path).name}")

    def _is_current(self, path: str) -> bool:
        item = self.file_list.currentItem()
        return item is not None and item.data(Qt.UserRole) == path

    def _on_file_progress(self, path: str, value: int) -> None:
        self.progress.setValue(value)
        self.statusBar().showMessage(f"Анализ: {Path(path).name} — {value}%")

    def _on_file_done(self, path: str, result: AnalysisResult) -> None:
        self._analyzing.discard(path)
        self._results[path] = result
        self._errors.pop(path, None)
        self._set_item_status(path, "done")
        if self._is_current(path):
            self._display_result(result)
        self.statusBar().showMessage(
            f"Готово: {Path(path).name} — {result.detection.headline}", 8000
        )

    def _on_file_failed(self, path: str, message: str) -> None:
        self._analyzing.discard(path)
        self._errors[path] = message
        self._set_item_status(path, "error")
        if self._is_current(path):
            self._display_error(path, message)
        self.statusBar().showMessage(f"Ошибка: {Path(path).name} — {message}", 10000)

    def _on_worker_finished(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        self.progress.hide()
        if self._pending:
            self._pump_worker()
        elif not self._analyzing:
            self.statusBar().showMessage("Готово", 5000)

    def _set_item_status(self, path: str, status: str) -> None:
        item = self._items.get(path)
        if item is None:
            return
        name = Path(path).name
        if status == "analyzing":
            marker, color = "◌", ACCENT
        elif status == "error":
            marker, color = "!", FAKE
        elif status == "done":
            result = self._results.get(path)
            marker = "●"
            color = VERDICT_COLORS.get(result.detection.verdict, FG) if result else FG
        else:
            marker, color = "·", MUTED
        item.setText(f"{marker}  {name}")
        item.setForeground(QBrush(QColor(color)))
        if status == "done":
            result = self._results.get(path)
            detail = result.detection.headline if result else "анализ завершён"
        else:
            detail = self._status_text(path, status)
        item.setToolTip(f"{path}\n{detail}")

    @staticmethod
    def _status_text(path: str, status: str) -> str:
        names = {
            "queued": "в очереди",
            "analyzing": "идёт анализ…",
            "error": "ошибка чтения/анализа",
        }
        return names.get(status, "")

    # --------------------------------------------------------- Отображение
    def _on_selection_changed(
        self, current: QListWidgetItem | None, _previous=None
    ) -> None:
        if current is None:
            self._current_path = None
            self._show_empty()
            self.remove_button.setEnabled(False)
            return
        self.remove_button.setEnabled(True)
        path = current.data(Qt.UserRole)
        self._current_path = path
        if path in self._results:
            self._display_result(self._results[path])
        elif path in self._errors:
            self._display_error(path, self._errors[path])
        elif path in self._analyzing:
            self._show_loading(path)
        else:
            self._show_pending(path)

    def _display_result(self, result: AnalysisResult) -> None:
        meta = result.meta
        detection = result.detection
        self._current_path = meta.path
        name = Path(meta.path).name
        self.file_label.setText(name if len(name) <= 96 else name[:93] + "…")
        self.file_label.setToolTip(meta.path)
        self.meta_label.setText(format_audio_meta(meta))

        color = VERDICT_COLORS.get(detection.verdict, FG)
        self.verdict_label.setText(detection.headline)
        self.verdict_label.setStyleSheet(f"color: {color};")

        self.confidence_label.setText(
            f"уверенность {detection.confidence * 100:.0f}%"
            if detection.confidence >= 0.3
            else ""
        )
        self.confidence_label.setStyleSheet(f"color: {color};")

        self.reason_label.setText(detection.reason)
        self.details_label.setText(self._details_html(result))
        self.verdict_card.setStyleSheet(
            "QFrame#verdictCard {"
            f" background: #141b23; border: 1px solid {_blend(color, '#141b23', 0.55)};"
            " border-radius: 10px; }"
        )

        self.spectrum_canvas.show_result(result)
        self.spectrogram_canvas.show_result(result)
        self.focus_check.setEnabled(True)
        self.export_button.setEnabled(True)
        if self._worker is None:
            self.progress.hide()

    def _details_html(self, result: AnalysisResult) -> str:
        detection = result.detection
        parts: list[str] = []
        if detection.cutoff_hz is not None:
            drop = (
                f" · перепад {detection.drop_db:.0f} дБ"
                if detection.drop_db is not None
                else ""
            )
            parts.append(f"<b>Срез:</b> {format_khz(detection.cutoff_hz)}{drop}")
        if detection.estimated_source:
            parts.append(f"<b>Источник:</b> {escape(detection.estimated_source)}")
        if result.meta.is_lossy:
            parts.append("<b>Контейнер:</b> с потерями")
        line = " &nbsp;·&nbsp; ".join(parts)

        notes = "".join(
            f"<div style='color:{MUTED}; margin-top:3px'>{escape(note)}</div>"
            for note in detection.notes
        )
        return (line + notes) if (line or notes) else "&nbsp;"

    def _display_error(self, path: str, message: str) -> None:
        self._current_path = path
        self.file_label.setText(Path(path).name)
        self.file_label.setToolTip(path)
        self.meta_label.setText(str(Path(path).parent))
        self.verdict_label.setText("ОШИБКА АНАЛИЗА")
        self.verdict_label.setStyleSheet(f"color: {FAKE};")
        self.confidence_label.setText("")
        self.reason_label.setText("Не удалось прочитать или проанализировать файл.")
        self.details_label.setText(
            f"<span style='color:{MUTED}'>{escape(message)}</span>"
        )
        self.verdict_card.setStyleSheet(
            "QFrame#verdictCard {"
            f" background: #141b23; border: 1px solid {_blend(FAKE, '#141b23', 0.55)};"
            " border-radius: 10px; }"
        )
        self.spectrum_canvas.show_placeholder("Нет данных для отображения")
        self.spectrogram_canvas.show_placeholder("Нет данных для отображения")
        self.focus_check.setEnabled(False)
        self.export_button.setEnabled(False)
        if self._worker is None:
            self.progress.hide()

    def _show_loading(self, path: str) -> None:
        self._current_path = path
        self.file_label.setText(Path(path).name)
        self.meta_label.setText("Чтение файла и расчёт спектра…")
        self.verdict_label.setText("АНАЛИЗ…")
        self.verdict_label.setStyleSheet(f"color: {ACCENT};")
        self.confidence_label.setText("")
        self.reason_label.setText(
            "Файл читается потоково, поэтому анализ длинных записей занимает время."
        )
        self.details_label.setText(f"<span style='color:{MUTED}'>{escape(path)}</span>")
        self.verdict_card.setStyleSheet(
            "QFrame#verdictCard {"
            f" background: #141b23; border: 1px solid {_blend(ACCENT, '#141b23', 0.45)};"
            " border-radius: 10px; }"
        )
        self.spectrum_canvas.show_placeholder("Идёт анализ…")
        self.spectrogram_canvas.show_placeholder("Идёт анализ…")
        self.focus_check.setEnabled(False)
        self.export_button.setEnabled(False)

    def _show_pending(self, path: str) -> None:
        self._current_path = path
        self.file_label.setText(Path(path).name)
        self.meta_label.setText(str(Path(path).parent))
        self.verdict_label.setText("В ОЧЕРЕДИ")
        self.verdict_label.setStyleSheet(f"color: {MUTED};")
        self.confidence_label.setText("")
        self.reason_label.setText("Файл ожидает анализа.")
        self.details_label.setText(f"<span style='color:{MUTED}'>{escape(path)}</span>")
        self.verdict_card.setStyleSheet("")
        self.spectrum_canvas.show_placeholder("Ожидание анализа")
        self.spectrogram_canvas.show_placeholder("Ожидание анализа")
        self.focus_check.setEnabled(False)
        self.export_button.setEnabled(False)
        self.progress.hide()

    def _show_empty(self) -> None:
        self.file_label.setText("Файл не выбран")
        self.meta_label.setText("Добавьте аудиофайл или перетащите его в окно")
        self.verdict_label.setText("ОЖИДАНИЕ")
        self.verdict_label.setStyleSheet(f"color: {MUTED};")
        self.confidence_label.setText("")
        self.reason_label.setText(
            "Программа найдёт «кирпичную стену» в спектре и покажет, "
            "не является ли lossless-файл пересохранённым MP3/AAC."
        )
        self.details_label.setText(
            f"<span style='color:{MUTED}'>Поддерживаются WAV, FLAC, ALAC, MP3, M4A, "
            "OGG и другие форматы. Для M4A/ALAC нужен ffmpeg.</span>"
        )
        self.verdict_card.setStyleSheet("")
        self.spectrum_canvas.show_placeholder("Спектр появится после анализа файла")
        self.spectrogram_canvas.show_placeholder(
            "Спектрограмма появится после анализа файла"
        )
        self.focus_check.setEnabled(False)
        self.export_button.setEnabled(False)
        self.progress.hide()

    # ------------------------------------------------------------- Экспорт
    def _export_report(self) -> None:
        path = self._current_path
        result = self._results.get(path) if path else None
        if result is None:
            QMessageBox.information(
                self, "Экспорт отчёта", "Сначала проанализируйте файл."
            )
            return
        stem = Path(path).stem
        default_path = str(Path(path).with_name(f"{stem}_lossless_report.png"))
        filename, selected = QFileDialog.getSaveFileName(
            self,
            "Экспорт отчёта",
            default_path,
            "PNG (*.png);;PDF (*.pdf)",
        )
        if not filename:
            return
        if Path(filename).suffix.lower() not in {".png", ".pdf"}:
            filename += ".pdf" if "PDF" in selected else ".png"
        try:
            build_report(result, filename)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Ошибка экспорта", str(exc))
            return
        self.statusBar().showMessage(f"Отчёт сохранён: {filename}", 10000)

    # ------------------------------------------------------- Drag and drop
    @staticmethod
    def _paths_from_mime(mime) -> list[str]:
        if not mime.hasUrls():
            return []
        result = []
        for url in mime.urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                result.append(str(path))
        return result

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._paths_from_mime(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802
        if self._paths_from_mime(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802
        paths = self._paths_from_mime(event.mimeData())
        if paths:
            self._add_paths(paths)
            event.acceptProposedAction()
        else:
            event.ignore()

    # ------------------------------------------------------------- Закрытие
    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            if not worker.wait(8000):
                worker.terminate()
                worker.wait(1500)
            self._worker = None
        super().closeEvent(event)
