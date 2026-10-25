"""Точка входа Lossless Analyzer."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from lossless_analyzer.gui import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Lossless Analyzer")
    app.setOrganizationName("Lossless Analyzer")
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
