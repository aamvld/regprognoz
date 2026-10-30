"""Точка входа программной системы «РегПрогноз».

Запуск:  python main.py
"""

import sys
import traceback

# Библиотеки ОБЯЗАТЕЛЬНО загружаются раньше PySide6 (причина — в preload.py)
import preload  # noqa: F401

from PySide6.QtWidgets import QApplication, QMessageBox

from ui.main_window import MainWindow


def excepthook(exc_type, exc, tb):
    """Последний рубеж защиты: любая неперехваченная ошибка показывается
    в окне сообщения, а программа продолжает работу (п. 4.2.3 ТЗ)."""
    traceback.print_exception(exc_type, exc, tb)
    QMessageBox.critical(None, "Ошибка",
                         f"Непредвиденная ошибка: {exc}\n\n"
                         "Работа программы продолжается.")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")          # одинаковый вид на Windows и macOS
    sys.excepthook = excepthook
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
