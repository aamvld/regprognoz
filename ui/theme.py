"""Оформление интерфейса: светлая тема в стиле государственных порталов
(белые карточки на светло-сером фоне, синий акцентный цвет, скруглённые
кнопки и поля ввода).

Палитра задаётся явно, поэтому вид программы не зависит от того, включена
ли в Windows тёмная тема.
"""

import os
import tempfile

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import QApplication

# ---- цвета ----
BG = "#F5F7FA"          # фон окна
CARD = "#FFFFFF"        # фон карточек, полей, таблиц
TEXT = "#0B1F33"        # основной текст
TEXT2 = "#66727F"       # второстепенный текст
BORDER = "#C9D1DA"      # рамки полей
LINE = "#E1E6EE"        # тонкие разделители, сетка таблиц
BLUE = "#0D4CD3"        # акцентный цвет: кнопки, активная вкладка, ссылки
BLUE_HOVER = "#0B40B3"
BLUE_PRESS = "#08328A"
BLUE_LIGHT = "#E8EFFD"  # подсветка выделения и лучших значений
RED = "#EE3F58"         # линия идеального прогноза на графике
DISABLED = "#A6B0BC"
DISABLED_BG = "#E9EDF2"

QSS = """
* {{ color: {TEXT}; }}
QMainWindow {{ background: {CARD}; }}
QDialog {{ background: {BG}; }}

QMenuBar {{ background: {CARD}; border-bottom: 1px solid {LINE}; padding: 2px 6px; }}
QMenuBar::item {{ padding: 6px 12px; background: transparent; border-radius: 6px; }}
QMenuBar::item:selected {{ background: {BLUE_LIGHT}; color: {BLUE}; }}
QMenu {{ background: {CARD}; border: 1px solid {LINE}; padding: 6px 0; }}
QMenu::item {{ padding: 6px 28px 6px 20px; }}
QMenu::item:selected {{ background: {BLUE_LIGHT}; color: {BLUE}; }}
QMenu::item:disabled {{ color: {DISABLED}; }}
QMenu::separator {{ height: 1px; background: {LINE}; margin: 6px 12px; }}

QTabWidget::pane {{ border: none; border-top: 1px solid {LINE}; background: {BG}; }}
QTabBar {{ background: {CARD}; }}
QTabBar::tab {{ background: {CARD}; color: {TEXT}; padding: 10px 18px 8px 18px;
               border: none; border-bottom: 3px solid transparent; margin-right: 2px; }}
QTabBar::tab:hover {{ color: {BLUE}; }}
QTabBar::tab:selected {{ color: {BLUE}; border-bottom: 3px solid {BLUE}; font-weight: bold; }}
QTabBar::tab:disabled {{ color: {DISABLED}; }}

QGroupBox {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 12px;
            margin-top: 0px; padding: 36px 14px 14px 14px; font-weight: bold; font-size: 11pt; }}
QGroupBox::title {{ subcontrol-origin: padding; subcontrol-position: top left;
                   left: 14px; top: 12px; color: {TEXT}; }}
QGroupBox QWidget {{ font-weight: normal; font-size: 10pt; }}
QGroupBox QPushButton[kind="primary"] {{ font-weight: bold; }}

QPushButton {{ background: {CARD}; color: {BLUE}; border: 1px solid {BLUE};
              border-radius: 8px; padding: 8px 20px; min-height: 20px; }}
QPushButton:hover {{ background: {BLUE_LIGHT}; }}
QPushButton:pressed {{ background: #D3E0FA; }}
QPushButton:disabled {{ color: {DISABLED}; border-color: {BORDER}; background: {CARD}; }}
QPushButton[kind="primary"] {{ background: {BLUE}; color: white; border: 1px solid {BLUE};
                              font-weight: bold; }}
QPushButton[kind="primary"]:hover {{ background: {BLUE_HOVER}; border-color: {BLUE_HOVER}; }}
QPushButton[kind="primary"]:pressed {{ background: {BLUE_PRESS}; border-color: {BLUE_PRESS}; }}
QPushButton[kind="primary"]:disabled {{ background: {DISABLED_BG}; color: {DISABLED};
                                        border-color: {DISABLED_BG}; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{ background: {CARD}; border: 1px solid {BORDER};
    border-radius: 8px; padding: 5px 8px; min-height: 20px;
    selection-background-color: {BLUE}; selection-color: white; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border: 1px solid {BLUE}; }}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    background: {DISABLED_BG}; color: {DISABLED}; }}
QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right;
                        width: 30px; border: none; }}
QComboBox::down-arrow {{ image: url({ARROW}); width: 12px; height: 8px; }}
QComboBox::down-arrow:disabled {{ image: url({ARROW_DIS}); }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ width: 0px; border: none; }}
QComboBox QAbstractItemView {{ background: {CARD}; border: 1px solid {LINE};
    selection-background-color: {BLUE_LIGHT}; selection-color: {BLUE}; }}

QListWidget, QTextEdit, QTextBrowser {{ background: {CARD}; border: 1px solid {BORDER};
    border-radius: 8px; padding: 4px; }}
QListWidget::item:selected {{ background: {BLUE_LIGHT}; color: {TEXT}; }}

QTableWidget, QTableView {{ background: {CARD}; border: 1px solid {LINE}; border-radius: 8px;
    gridline-color: {LINE}; selection-background-color: {BLUE_LIGHT}; selection-color: {TEXT}; }}
QHeaderView::section {{ background: {BG}; color: {TEXT2}; border: none;
    border-right: 1px solid {LINE}; border-bottom: 1px solid {LINE}; padding: 6px 8px; font-weight: bold; }}
QTableCornerButton::section {{ background: {BG}; border: none; border-bottom: 1px solid {LINE}; }}

QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid {BORDER};
    border-radius: 4px; background: {CARD}; }}
QCheckBox::indicator:checked, QListView::indicator:checked {{ background: {BLUE};
    border-color: {BLUE}; image: url({CHECK}); }}
QListView::indicator {{ width: 16px; height: 16px; border: 1px solid {BORDER};
    border-radius: 4px; background: {CARD}; }}
QCheckBox::indicator:disabled {{ background: {DISABLED_BG}; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {BORDER}; border-radius: 4px; min-height: 30px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {BORDER}; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QStatusBar {{ background: {CARD}; color: {TEXT2}; border-top: 1px solid {LINE}; }}
QStatusBar::item {{ border: none; }}
QToolTip {{ background: {TEXT}; color: white; border: none; padding: 4px 8px; }}
QMessageBox {{ background: {CARD}; }}
"""


def _icons() -> dict:
    """Рисует значки (стрелка списка, галочка) в PNG во временной папке:
    таблица стилей Qt принимает картинки только в виде файлов."""
    d = os.path.join(tempfile.gettempdir(), "regprognoz_theme")
    os.makedirs(d, exist_ok=True)
    paths = {}
    for name, color, w, h, pts in (
        ("arrow", TEXT2, 12, 8, [(1.5, 2), (6, 6.5), (10.5, 2)]),
        ("arrow_dis", DISABLED, 12, 8, [(1.5, 2), (6, 6.5), (10.5, 2)]),
        ("check", "#FFFFFF", 16, 16, [(3.5, 8.5), (6.8, 11.5), (12.5, 4.5)]),
    ):
        for scale, suffix in ((1, ""), (2, "@2x")):
            pm = QPixmap(w * scale, h * scale)
            pm.fill(Qt.GlobalColor.transparent)
            p = QPainter(pm)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            pen = QPen(QColor(color), 2 * scale)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            p.setPen(pen)
            p.drawPolyline([QPointF(x * scale, y * scale) for x, y in pts])
            p.end()
            pm.save(os.path.join(d, f"{name}{suffix}.png"))
        paths[name] = os.path.join(d, f"{name}.png").replace("\\", "/")
    return {"ARROW": paths["arrow"], "ARROW_DIS": paths["arrow_dis"], "CHECK": paths["check"]}


def apply_theme(app: QApplication) -> None:
    """Включает светлую тему для всего приложения."""
    app.setStyle("Fusion")
    pal = QPalette()
    for role, color in (
        (QPalette.ColorRole.Window, BG), (QPalette.ColorRole.WindowText, TEXT),
        (QPalette.ColorRole.Base, CARD), (QPalette.ColorRole.AlternateBase, BG),
        (QPalette.ColorRole.Text, TEXT), (QPalette.ColorRole.Button, CARD),
        (QPalette.ColorRole.ButtonText, TEXT), (QPalette.ColorRole.Highlight, BLUE),
        (QPalette.ColorRole.HighlightedText, "#FFFFFF"), (QPalette.ColorRole.ToolTipBase, TEXT),
        (QPalette.ColorRole.ToolTipText, "#FFFFFF"), (QPalette.ColorRole.Link, BLUE),
        (QPalette.ColorRole.PlaceholderText, TEXT2),
    ):
        pal.setColor(role, QColor(color))
    for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text, QPalette.ColorRole.ButtonText):
        pal.setColor(QPalette.ColorGroup.Disabled, role, QColor(DISABLED))
    app.setPalette(pal)
    font = QFont()
    font.setFamilies(["Segoe UI", "Lato", "Arial"])
    font.setPointSize(10)
    app.setFont(font)
    app.setStyleSheet(QSS.format(**globals(), **_icons()))
