"""Генерация Word-файла с таблицами для РПЗ.

Запуск:  python make_tables.py        (≈ 1 минута)
Результат: results/Таблицы_для_РПЗ.docx — четыре готовые таблицы:
  1) характеристики инструментальных средств (п. 5.4.7 ТЗ);
  2) модульные тесты (п. 5.4.8 ТЗ);
  3) функциональные тесты по требованиям ТЗ (п. 5.4.8 ТЗ);
  4) проверка временных характеристик (п. 4.1.1.2 ТЗ).
Все фактические результаты получаются автоматически при запуске,
версии программ — из установленных на этом компьютере.
"""

import os
import platform
import shutil
import subprocess
import sys
import time
import unittest
from importlib import metadata

import preload  # noqa: F401  (библиотеки загружаются до PySide6)

try:
    from docx import Document
    from docx.enum.section import WD_ORIENT
    from docx.enum.table import WD_TABLE_ALIGNMENT
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm, Pt, RGBColor
except ImportError:
    sys.exit("Нужна библиотека python-docx. Установите: pip install python-docx")

import check_performance
import functional_tests

HERE = os.path.dirname(os.path.abspath(__file__))
FONT = "Times New Roman"


# ===========================================================================
# Версии инструментов
# ===========================================================================
def pkg(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "не установлен"


def cmd_version(cmd, args=("--version",)):
    exe = shutil.which(cmd)
    if not exe:
        return None
    try:
        out = subprocess.run([exe, *args], capture_output=True, text=True,
                             timeout=15, shell=sys.platform == "win32" and exe.endswith(".cmd"))
        return out.stdout.strip().splitlines()[0] if out.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):
        return None


def inno_version():
    if sys.platform != "win32":
        return None
    import winreg
    keys = [(winreg.HKEY_LOCAL_MACHINE,
             r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1"),
            (winreg.HKEY_LOCAL_MACHINE,
             r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1"),
            (winreg.HKEY_CURRENT_USER,
             r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1")]
    for root, path in keys:
        try:
            with winreg.OpenKey(root, path) as k:
                return winreg.QueryValueEx(k, "DisplayVersion")[0]
        except OSError:
            continue
    return None


def tools_rows():
    git = cmd_version("git")
    code = cmd_version("code")
    rows = [
        ("Язык программирования", "Python", platform.python_version(),
         "Высокоуровневый язык с развитыми библиотеками анализа данных и "
         "построения графического интерфейса"),
        ("Среда разработки", "Visual Studio Code", code or "—",
         "Редактор кода с подсветкой синтаксиса, отладчиком и встроенным терминалом"),
        ("Библиотека интерфейса", "PySide6 (Qt for Python)", pkg("PySide6"),
         "Графический интерфейс: главное окно, вкладки, таблицы, диалоги выбора файлов"),
        ("Обработка данных", "pandas", pkg("pandas"),
         "Чтение файлов CSV, хранение и обработка табличных данных, обработка пропусков"),
        ("Численные вычисления", "NumPy", pkg("numpy"),
         "Матричные операции; собственная реализация МНК, гребневой и лассо-регрессии"),
        ("Визуализация", "Matplotlib", pkg("matplotlib"),
         "Построение графика «фактические – прогнозные значения», сохранение в PNG"),
        ("Средство автоматизации сборки", "PyInstaller", pkg("pyinstaller"),
         "Сборка исполняемого файла со встроенным интерпретатором и библиотеками "
         "(п. 4.5.4 ТЗ)"),
        ("Средство создания дистрибутива", "Inno Setup", inno_version() or "6",
         "Создание установщика программы для Windows (п. 4.6.1 ТЗ)"),
        ("Контроль версий", "Git", (git or "—").replace("git version ", ""),
         "Хранение истории изменений исходного кода"),
        ("Средство модульного тестирования", "unittest (стандартная библиотека Python)",
         platform.python_version(),
         "Автоматические модульные тесты классов (метод «белого ящика»)"),
        ("Эталон для проверки результатов", "scikit-learn", pkg("scikit-learn"),
         "Сверка результатов собственных алгоритмов регрессии и разбиения выборки "
         "(в программу не входит)"),
        ("Методы тестирования", "—", "—",
         "«Белый ящик»: модульные тесты классов; «чёрный ящик»: функциональные тесты "
         "по требованиям ТЗ с выделением классов эквивалентности и граничных значений; "
         "проверка блокировок интерфейса и устойчивости к ошибкам; сравнение с "
         "эталонной реализацией; замер временных характеристик"),
    ]
    if not git:
        rows = [r for r in rows if r[1] != "Git"]
    return rows


# ===========================================================================
# Модульные тесты
# ===========================================================================
UNIT_DESCR = {
    "TestDataset": "Размеры набора, числовые столбцы, предпросмотр, поиск пропусков",
    "TestCsvReader": "Чтение корректного CSV и CSV из Excel; ошибки формата, кодировки, "
                     "размера; неизменность исходного файла",
    "TestInputValidator": "Граничные значения параметров генерации, моделирования и "
                          "данных для прогноза",
    "TestMissingValueHandler": "Удаление строк, замена средним, сохранение исходного набора",
    "TestSyntheticGenerator": "Параметры по умолчанию, воспроизводимость, точная "
                              "зависимость при нулевом шуме, сохранение в CSV",
    "TestDataPreparer": "Размеры и воспроизводимость разбиения, стандартизация по "
                        "обучающей выборке",
    "TestSolvers": "Точность МНК, аналитическое решение Ridge, обнуление коэффициентов "
                   "и условия оптимальности Lasso",
    "TestMetrics": "MSE, MAE, R² на известных значениях; R² для постоянной выборки",
    "TestExperimentService": "Восстановление истинных коэффициентов, прогноз по уравнению, "
                             "сравнение на одной выборке, пропуски",
    "TestOutput": "График, отчёт TXT, таблица CSV, файл PNG, ошибка записи",
}


def unit_rows():
    suite = unittest.defaultTestLoader.discover(os.path.join(HERE, "tests"),
                                                top_level_dir=HERE)

    def walk(s):
        for x in s:
            yield from (walk(x) if isinstance(x, unittest.TestSuite) else [x])

    # список тестов собирается ДО запуска: после прогона unittest его очищает
    tests = [(type(t).__name__, t.id()) for t in walk(suite)]
    result = unittest.TestResult()
    suite.run(result)
    failed = {t.id() for t, _ in result.failures + result.errors}

    stats = {}
    for cls, test_id in tests:
        total, bad = stats.get(cls, (0, 0))
        stats[cls] = (total + 1, bad + (test_id in failed))
    rows = [(cls, UNIT_DESCR.get(cls, ""), str(n),
             "пройдены" if bad == 0 else f"не пройдено {bad}")
            for cls, (n, bad) in stats.items()]
    return rows, result.testsRun, len(failed)


# ===========================================================================
# Оформление Word
# ===========================================================================
def set_cell(cell, text, size=10, bold=False, align=None, color=None):
    cell.text = ""
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    if align:
        p.alignment = align
    run = p.add_run(str(text))
    run.font.name = FONT
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = color
    run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)


def shade(cell, hex_color):
    tcPr = cell._element.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tcPr.append(shd)


def repeat_header(row):
    trPr = row._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    trPr.append(el)


def para(doc, text, size=14, bold=False, align=None, space_after=6, indent=True):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    if indent:
        p.paragraph_format.first_line_indent = Cm(1.25)
    if align is not None:
        p.alignment = align
    run = p.add_run(text)
    run.font.name = FONT
    run.font.size = Pt(size)
    run.bold = bold
    run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    return p


def caption(doc, text):
    """Подпись таблицы по ГОСТ 7.32: слева, над таблицей."""
    p = para(doc, text, size=14, indent=False, space_after=3)
    p.paragraph_format.keep_with_next = True


def table(doc, headers, rows, widths, size=10, center_cols=(), group_col=None):
    """group_col — индекс столбца с названием раздела: раздел выводится
    отдельной объединённой строкой, сам столбец в таблицу не попадает."""
    shown = [i for i in range(len(headers)) if i != group_col]
    t = doc.add_table(rows=1, cols=len(shown))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0]
    repeat_header(hdr)
    for j, i in enumerate(shown):
        set_cell(hdr.cells[j], headers[i], size, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER)
        shade(hdr.cells[j], "E7E6E6")
    last_group = None
    for r in rows:
        if group_col is not None and r[group_col] != last_group:
            last_group = r[group_col]
            gr = t.add_row()
            merged = gr.cells[0].merge(gr.cells[-1])
            set_cell(merged, last_group, size, bold=True)
            shade(merged, "F2F2F2")
        cells = t.add_row().cells
        for j, i in enumerate(shown):
            v = r[i]
            color = RGBColor(0xC0, 0, 0) if str(v).startswith(("НЕ ", "не пройдено")) else None
            set_cell(cells[j], v, size,
                     align=WD_ALIGN_PARAGRAPH.CENTER if i in center_cols else None,
                     color=color)
    # ширины: и в сетке таблицы (LibreOffice), и в каждой ячейке (Word)
    t.autofit = False
    for j, w in enumerate(widths):
        t.columns[j].width = Cm(w)
    for row in t.rows:
        for j, w in enumerate(widths):
            row.cells[j].width = Cm(w)
    doc.add_paragraph()
    return t


def main():
    started = time.perf_counter()
    print("1/4. Версии инструментов…")
    tools = tools_rows()
    print("2/4. Модульные тесты…")
    units, n_unit, n_unit_failed = unit_rows()
    print(f"     выполнено {n_unit}, не пройдено {n_unit_failed}")
    print("3/4. Функциональные тесты…")
    func = functional_tests.run_all()
    n_func_ok = sum(r["Вывод"] == "пройден" for r in func)
    print(f"     пройдено {n_func_ok} из {len(func)}")
    print("4/4. Замер временных характеристик (около минуты)…")
    env, perf = check_performance.collect()

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.orientation = WD_ORIENT.PORTRAIT
    sec.left_margin, sec.right_margin = Cm(3), Cm(1.5)
    sec.top_margin = sec.bottom_margin = Cm(2)

    para(doc, "Таблицы для расчётно-пояснительной записки «РегПрогноз»", 14, True,
         WD_ALIGN_PARAGRAPH.CENTER, indent=False)
    para(doc, f"Сформировано автоматически {time.strftime('%d.%m.%Y %H:%M')}. "
              f"{env}. Номера таблиц замените на сквозные номера РПЗ.", 12, indent=False)

    caption(doc, "Таблица 1 – Характеристики инструментальных средств разработки")
    table(doc, ["Назначение", "Средство", "Версия", "Обоснование выбора / применение"],
          tools, [3.6, 3.4, 2.2, 7.3], size=11, center_cols=(2,))

    para(doc, f"Модульное тестирование выполнено методом «белого ящика» средствами "
              f"unittest: {n_unit} автоматических тестов для {len(units)} классов, "
              f"не пройдено — {n_unit_failed}.", 14)
    caption(doc, "Таблица 2 – Модульные тесты")
    table(doc, ["Тестовый класс", "Что проверяется", "Тестов", "Результат"],
          units, [4.2, 8.6, 1.6, 2.1], size=11, center_cols=(2, 3))

    para(doc, f"Функциональное тестирование выполнено методом «чёрного ящика» по "
              f"требованиям ТЗ; для числовых параметров проверены граничные значения "
              f"и значения за границами допустимых диапазонов. Пройдено {n_func_ok} "
              f"тестов из {len(func)}.", 14)
    caption(doc, "Таблица 3 – Функциональные тесты")
    f_rows = [(r["Раздел"], str(r["№"]), r["Пункт ТЗ"], r["Исходные данные"],
               r["Ожидаемый результат"], r["Фактический результат"], r["Вывод"])
              for r in func]
    table(doc, ["Раздел", "№", "Пункт ТЗ", "Исходные данные", "Ожидаемый результат",
                "Фактический результат", "Вывод"],
          f_rows, [0.8, 1.7, 3.6, 3.6, 5.2, 1.6], size=9,
          center_cols=(1, 2, 6), group_col=0)

    para(doc, f"Временные характеристики проверены на компьютере: {env.replace('Компьютер: ', '')}. "
              f"Каждая операция выполнялась три раза, в таблице указано наибольшее время.", 14)
    caption(doc, "Таблица 4 – Проверка временных характеристик и объёма памяти")
    table(doc, ["Операция", "Требование", "Факт", "Выполнено"],
          perf, [9.0, 2.6, 3.0, 1.9], size=11, center_cols=(1, 2, 3))

    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "Таблицы_для_РПЗ.docx")
    doc.save(out)
    print(f"\nГотово за {time.perf_counter() - started:.0f} с: {out}")


if __name__ == "__main__":
    main()
