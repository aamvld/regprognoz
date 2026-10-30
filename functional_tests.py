"""Функциональные тесты по требованиям ТЗ (метод «чёрного ящика»).

Запуск:  python functional_tests.py
Каждый тест: № | пункт ТЗ | исходные данные | ожидаемый результат |
фактический результат | вывод. Фактический результат получается
автоматически при прогоне. Таблица сохраняется в
results/functional_tests.csv, а make_tables.py вставляет её в Word.

Использованы методы: классы эквивалентности и граничные значения
(допустимые и недопустимые значения на границах диапазонов ТЗ),
проверка блокировок интерфейса и устойчивости к ошибкам.
"""

import hashlib
import os
import sys
import tempfile
import traceback

import preload  # noqa: F401  (библиотеки загружаются до PySide6)
import numpy as np
import pandas as pd

from core.csv_reader import CsvReader
from core.errors import ValidationError
from core.experiment import ExperimentService
from core.generator import SyntheticGenerator
from core.missing import MissingValueHandler
from core.params import ModelParams
from core.validator import InputValidator
from output.plot_builder import PlotBuilder
from output.report_exporter import ReportExporter

HERE = os.path.dirname(os.path.abspath(__file__))


# ===========================================================================
# Тестовые файлы
# ===========================================================================
def make_test_files(folder: str) -> dict[str, str]:
    """Создаёт набор корректных и ошибочных CSV. Возвращает {ключ: путь}."""
    os.makedirs(folder, exist_ok=True)
    f = {}

    def gen(n, p, seed=1, noise=2.0):
        # отдельная генерация: SyntheticGenerator программы правильно
        # отказывается создавать наборы вне диапазонов ТЗ, а здесь они нужны
        rng = np.random.default_rng(seed)
        X = np.round(rng.normal(size=(n, p)), 4)
        y = np.round(3.0 + X @ rng.uniform(-5, 5, p) + rng.normal(0, noise, n), 4)
        df = pd.DataFrame(X, columns=[f"x{i}" for i in range(1, p + 1)])
        df["y"] = y
        return df

    def save(key, name, frame, **kw):
        path = os.path.join(folder, name)
        frame.to_csv(path, index=False, **{"sep": ";", "encoding": "utf-8", **kw})
        f[key] = path

    save("ok", "01_корректный_500x3.csv", gen(500, 3))
    save("rows20", "02_граница_20_строк.csv", gen(20, 2))
    save("rows19", "03_ошибка_19_строк.csv", gen(19, 2))
    save("rows10000", "04_граница_10000_строк.csv", gen(10_000, 3))
    save("rows10001", "05_ошибка_10001_строка.csv", gen(10_001, 3))
    save("cols21", "06_граница_20_входных_признаков.csv", gen(100, 20))
    save("cols22", "07_ошибка_21_входной_признак.csv", gen(100, 21))
    save("comma", "08_ошибка_разделитель_запятая.csv", gen(100, 3), sep=",")
    df = gen(100, 2).rename(columns={"y": "цена", "x1": "площадь"})
    save("cp1251", "09_ошибка_кодировка_cp1251.csv", df, encoding="cp1251")
    save("deccomma", "10_ошибка_десятичная_запятая.csv", gen(100, 2), decimal=",")
    path = os.path.join(folder, "11_ошибка_пустой_файл.csv")
    open(path, "w", encoding="utf-8").close()
    f["empty"] = path
    path = os.path.join(folder, "12_ошибка_расширение.txt")
    gen(100, 2).to_csv(path, sep=";", index=False)
    f["txt"] = path
    df = gen(100, 2).rename(columns={"y": "цена", "x1": "площадь", "x2": "этаж"})
    save("bom", "13_корректный_Excel_UTF8_BOM.csv", df, encoding="utf-8-sig")
    df = gen(100, 2)
    df.insert(0, "город", ["Москва", "Казань"] * 50)
    save("text_col", "14_корректный_с_текстовым_столбцом.csv", df)
    df = gen(100, 2)
    df.loc[[2, 10, 50], "x1"] = np.nan
    df.loc[5, "y"] = np.nan
    save("missing", "15_корректный_с_пропусками.csv", df)
    df = gen(30, 2)
    df.loc[:14, "x2"] = np.nan
    save("missing_many", "16_пропуски_в_половине_строк.csv", df)
    return f


def file_hash(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


# ===========================================================================
# Вспомогательные функции
# ===========================================================================
def expect_error(action, must_contain: str = ""):
    """Ожидаем сообщение об ошибке (ValidationError), а не аварийное завершение."""
    try:
        action()
    except ValidationError as e:
        ok = must_contain.lower() in str(e).lower()
        return f"Сообщение: «{e}»", ok
    return "Ошибка не обнаружена, данные приняты", False


def fmt(v):
    return f"{v:.4f}"


# ===========================================================================
# Тесты ядра (без интерфейса)
# ===========================================================================
def core_tests(F, tmp):
    T = []  # (раздел, пункт ТЗ, исходные данные, ожидаемый, функция)

    def t(section, req, data, expected):
        def deco(fn):
            T.append((section, req, data, expected, fn))
            return fn
        return deco

    S = "Загрузка данных"

    @t(S, "4.1.1.1, 4.1.2.1", "Корректный файл: 500 строк, 3 входных + целевой",
       "Набор загружен; 500 строк, 4 столбца; предпросмотр 100 строк")
    def _():
        ds = CsvReader().read(F["ok"])
        return (f"Загружено строк: {ds.row_count()}, столбцов: {ds.column_count()}, "
                f"предпросмотр {len(ds.preview(100))} строк",
                (ds.row_count(), ds.column_count(), len(ds.preview(100))) == (500, 4, 100))

    @t(S, "4.1.2.1", "Граница: 20 строк", "Набор загружен")
    def _():
        ds = CsvReader().read(F["rows20"])
        return f"Загружено {ds.row_count()} строк", ds.row_count() == 20

    @t(S, "4.1.2.1, 4.2.1", "За границей: 19 строк",
       "Сообщение: допустимо от 20 до 10 000 строк")
    def _():
        return expect_error(lambda: CsvReader().read(F["rows19"]), "от 20")

    @t(S, "4.1.2.1", "Граница: 10 000 строк", "Набор загружен")
    def _():
        ds = CsvReader().read(F["rows10000"])
        return f"Загружено {ds.row_count()} строк", ds.row_count() == 10_000

    @t(S, "4.1.2.1, 4.2.1", "За границей: 10 001 строка",
       "Сообщение: допустимо от 20 до 10 000 строк")
    def _():
        return expect_error(lambda: CsvReader().read(F["rows10001"]), "10000")

    @t(S, "4.1.2.1", "Граница: 20 входных + 1 целевой столбец", "Набор загружен")
    def _():
        ds = CsvReader().read(F["cols21"])
        return (f"Загружено, числовых столбцов: {len(ds.numeric_columns())}",
                len(ds.numeric_columns()) == 21)

    @t(S, "4.1.2.1, 4.2.1", "За границей: 21 входной + 1 целевой",
       "Сообщение о допустимом числе столбцов")
    def _():
        return expect_error(lambda: CsvReader().read(F["cols22"]), "числовых столбцов")

    @t(S, "4.1.2.1, 4.2.1", "Разделитель полей «,» вместо «;»",
       "Сообщение: требуется разделитель «;»")
    def _():
        return expect_error(lambda: CsvReader().read(F["comma"]), "разделитель")

    @t(S, "4.1.2.1, 4.2.1", "Кодировка Windows-1251 вместо UTF-8",
       "Сообщение: файл не в кодировке UTF-8")
    def _():
        return expect_error(lambda: CsvReader().read(F["cp1251"]), "UTF-8")

    @t(S, "4.1.2.1, 4.2.1", "Десятичный разделитель — запятая",
       "Сообщение с подсказкой про десятичную точку")
    def _():
        return expect_error(lambda: CsvReader().read(F["deccomma"]), "точка")

    @t(S, "4.2.1, 4.2.3", "Пустой файл", "Сообщение: файл пустой")
    def _():
        return expect_error(lambda: CsvReader().read(F["empty"]), "пустой")

    @t(S, "4.2.1", "Файл с расширением .txt", "Сообщение: требуется расширение .csv")
    def _():
        return expect_error(lambda: CsvReader().read(F["txt"]), ".csv")

    @t(S, "4.1.2.1", "Файл из Excel «CSV UTF-8» (с меткой BOM), русские заголовки",
       "Набор загружен, заголовки прочитаны верно")
    def _():
        ds = CsvReader().read(F["bom"])
        cols = list(ds.frame.columns)
        return f"Загружено; столбцы: {', '.join(cols)}", cols[0] == "площадь"

    @t(S, "4.1.1.1", "Файл с текстовым столбцом «город»",
       "Набор загружен; текстовый столбец не предлагается как признак")
    def _():
        ds = CsvReader().read(F["text_col"])
        num = ds.numeric_columns()
        return f"Числовые столбцы: {', '.join(num)}", "город" not in num and len(num) == 3

    @t(S, "4.2.4", "Загрузка файла, обучение и сохранение результатов",
       "Исходный файл не изменился (контрольная сумма совпадает)")
    def _():
        before = file_hash(F["ok"])
        ds = CsvReader().read(F["ok"])
        r = ExperimentService().train(ds, ModelParams(target="y", features=["x1", "x2", "x3"]))
        ReportExporter.save_txt(r, os.path.join(tmp, "r.txt"))
        after = file_hash(F["ok"])
        return f"MD5 до: {before[:10]}…, после: {after[:10]}…", before == after

    S = "Обработка пропусков"

    @t(S, "4.1.1.1", "Файл с 4 пропусками (в 4 строках), способ «удаление строк»",
       "Удалены 4 строки, осталось 96, пропусков нет")
    def _():
        ds = CsvReader().read(F["missing"])
        out = MissingValueHandler.apply(ds, ["x1", "x2", "y"], "drop")
        return (f"Строк: {out.row_count()}, пропусков: {out.missing_count()}",
                out.row_count() == 96 and out.missing_count() == 0)

    @t(S, "4.1.1.1", "Тот же файл, способ «замена средним»",
       "100 строк; пропуски заменены средним столбца")
    def _():
        ds = CsvReader().read(F["missing"])
        mean = ds.frame["x1"].mean()
        out = MissingValueHandler.apply(ds, ["x1", "x2", "y"], "mean")
        v = out.frame.loc[2, "x1"]
        return (f"Строк: {out.row_count()}, пропусков: {out.missing_count()}, "
                f"x1[2] = {fmt(v)} (среднее {fmt(mean)})",
                out.row_count() == 100 and out.missing_count() == 0 and abs(v - mean) < 1e-12)

    @t(S, "4.2.1", "30 строк, пропуски в 15; способ «удаление строк»",
       "Сообщение: после удаления осталось менее 20 строк")
    def _():
        ds = CsvReader().read(F["missing_many"])
        return expect_error(lambda: MissingValueHandler.apply(ds, ["x2", "y"], "drop"),
                            "не менее 20")

    S = "Генерация данных"

    @t(S, "4.1.2.2", "Параметры по умолчанию: 500, 3, шум 1,0, seed 42",
       "Набор 500 × 4; показаны коэффициенты зависимости")
    def _():
        ds = SyntheticGenerator().generate()
        return (f"{ds.row_count()} × {ds.column_count()}; {ds.coefficients_text()}",
                (ds.row_count(), ds.column_count()) == (500, 4))

    @t(S, "4.1.2.2", "Границы: n = 20, p = 1, шум 0; обучение линейной модели",
       "Модель находит истинные коэффициенты, R² = 1")
    def _():
        ds = SyntheticGenerator(20, 1, 0.0, 42).generate()
        r = ExperimentService().train(ds, ModelParams(target="y", features=["x1"]))
        ok = (abs(r.coefficients[0] - ds.true_coefficients[0]) < 1e-3
              and abs(r.metrics.r2 - 1) < 1e-6)  # данные округлены до 4 знаков
        return (f"Истинный b1 = {ds.true_coefficients[0]}, найден {fmt(r.coefficients[0])}; "
                f"R² = {fmt(r.metrics.r2)}", ok)

    @t(S, "4.1.2.2", "Границы: n = 10 000, p = 20", "Набор 10 000 × 21 сгенерирован")
    def _():
        ds = SyntheticGenerator(10_000, 20, 1.0, 1).generate()
        return f"{ds.row_count()} × {ds.column_count()}", ds.frame.shape == (10_000, 21)

    for n, p, noise, what in ((19, 3, 1.0, "n = 19"), (10_001, 3, 1.0, "n = 10 001"),
                              (500, 0, 1.0, "p = 0"), (500, 21, 1.0, "p = 21"),
                              (500, 3, -1.0, "шум = −1")):
        @t(S, "4.1.2.2, 4.2.1", f"Недопустимое значение: {what}",
           "Сообщение о допустимом диапазоне")
        def _(n=n, p=p, noise=noise):
            return expect_error(lambda: SyntheticGenerator(n, p, noise, 42).generate())

    @t(S, "4.1.2.2", "Два запуска с seed 7 и один с seed 8",
       "При одинаковом seed данные совпадают, при разном — различаются")
    def _():
        a, b, c = (SyntheticGenerator(100, 2, 1.0, s).generate().frame for s in (7, 7, 8))
        same, diff = a.equals(b), not a.equals(c)
        return f"seed 7 = seed 7: {'да' if same else 'нет'}; seed 7 ≠ seed 8: " \
               f"{'да' if diff else 'нет'}", same and diff

    @t(S, "4.1.1.1", "Сохранение сгенерированного набора в CSV и повторная загрузка",
       "Загруженные данные совпадают со сгенерированными")
    def _():
        ds = SyntheticGenerator(300, 4, 1.0, 5).generate()
        path = os.path.join(tmp, "gen.csv")
        SyntheticGenerator.save_csv(ds, path)
        back = CsvReader().read(path)
        ok = back.frame.equals(ds.frame)
        return f"Файл {os.path.basename(path)}: совпадение {'полное' if ok else 'нет'}", ok

    S = "Построение и оценка модели"
    ds = SyntheticGenerator(500, 3, 5.0, 7).generate()
    feats = ["x1", "x2", "x3"]

    for algo, name in (("linear", "Линейная регрессия"), ("ridge", "Ridge, α = 1"),
                       ("lasso", "Lasso, α = 1")):
        @t(S, "4.1.1.1, 4.1.2.3", f"{name}; набор 500 × 3, шум 5; параметры по умолчанию",
           "Выведены свободный член, 3 коэффициента, MSE, MAE, R²")
        def _(algo=algo):
            r = ExperimentService().train(ds, ModelParams(algorithm=algo, target="y",
                                                          features=feats))
            m = r.metrics
            return (f"b0 = {fmt(r.intercept)}; b = ({', '.join(fmt(b) for b in r.coefficients)}); "
                    f"MSE = {fmt(m.mse)}, MAE = {fmt(m.mae)}, R² = {fmt(m.r2)}",
                    len(r.coefficients) == 3 and all(np.isfinite([m.mse, m.mae, m.r2])))

    for alg, a in (("ridge", 0.0), ("lasso", -1.0)):
        @t(S, "4.1.2.3, 4.2.1", f"{'Ridge' if alg == 'ridge' else 'Lasso'}, α = {a:g}",
           "Сообщение: α должен быть больше 0")
        def _(alg=alg, a=a):
            return expect_error(lambda: ExperimentService().train(ds, ModelParams(
                algorithm=alg, alpha=a, target="y", features=feats)), "больше 0")

    @t(S, "4.1.2.3", "Доля тестовой выборки 10 % и 40 % (границы), 500 строк",
       "Тестовая выборка 50 и 200 наблюдений")
    def _():
        n = [len(ExperimentService().train(ds, ModelParams(
            target="y", features=feats, test_size=ts)).y_test) for ts in (0.10, 0.40)]
        return f"Размер тестовой выборки: {n[0]} и {n[1]}", n == [50, 200]

    for ts in (0.09, 0.41):
        @t(S, "4.1.2.3, 4.2.1", f"Доля тестовой выборки {ts * 100:.0f} %",
           "Сообщение о диапазоне 10–40 %")
        def _(ts=ts):
            return expect_error(lambda: ExperimentService().train(ds, ModelParams(
                target="y", features=feats, test_size=ts)), "10 %")

    @t(S, "4.2.2", "Целевой признак y выбран также входным",
       "Сообщение: целевой не может быть входным")
    def _():
        return expect_error(lambda: ExperimentService().train(ds, ModelParams(
            target="y", features=["y", "x1"])), "одновременно")

    @t(S, "4.2.2", "Не выбран ни один входной признак", "Сообщение: не выбран входной признак")
    def _():
        return expect_error(lambda: ExperimentService().train(ds, ModelParams(
            target="y", features=[])), "входной")

    @t(S, "4.1.1.1", "Линейная регрессия со стандартизацией и без",
       "Метрики совпадают (стандартизация не меняет прогноз МНК)")
    def _():
        a, b = (ExperimentService().train(ds, ModelParams(target="y", features=feats,
                                                          standardize=s)) for s in (True, False))
        return (f"R² = {a.metrics.r2:.10f} и {b.metrics.r2:.10f}",
                abs(a.metrics.r2 - b.metrics.r2) < 1e-10)

    @t(S, "4.1.1.1", "Обучение дважды с seed 42 и один раз с seed 1",
       "При одинаковом seed результаты совпадают, при разном — различаются")
    def _():
        r = [ExperimentService().train(ds, ModelParams(target="y", features=feats, seed=s))
             .metrics.mse for s in (42, 42, 1)]
        return f"MSE: {fmt(r[0])}, {fmt(r[1])}, {fmt(r[2])}", r[0] == r[1] != r[2]

    @t(S, "4.1.1.1", "Сравнение алгоритмов на наборе 500 × 3",
       "Сводная таблица из 3 строк, одинаковая тестовая выборка")
    def _():
        rs = ExperimentService().compare(ds, ModelParams(target="y", features=feats))
        same = all(np.array_equal(r.y_test, rs[0].y_test) for r in rs)
        txt = "; ".join(f"{r.summary_row()[0].split(' (')[0]}: R² = {fmt(r.metrics.r2)}"
                        for r in rs)
        return txt, len(rs) == 3 and same

    S = "Прогнозирование"

    @t(S, "4.1.1.1, 4.1.2.4", "Обученная модель; x1 = 1, x2 = 0,5, x3 = −2",
       "Прогноз равен b0 + b1·x1 + b2·x2 + b3·x3")
    def _():
        svc = ExperimentService()
        r = svc.train(ds, ModelParams(target="y", features=feats))
        vals = InputValidator.validate_prediction_input(
            {"x1": "1", "x2": "0,5", "x3": "-2"}, feats)
        y1 = svc.predict_one(vals)
        y2 = r.intercept + sum(b * vals[f] for b, f in zip(r.coefficients, feats))
        return f"Прогноз {fmt(y1)}, расчёт по уравнению {fmt(y2)}", abs(y1 - y2) < 1e-9

    @t(S, "4.2.1", "Поле x2 не заполнено", "Сообщение: не задано значение признака")
    def _():
        return expect_error(lambda: InputValidator.validate_prediction_input(
            {"x1": "1", "x2": "", "x3": "2"}, feats), "не задано")

    @t(S, "4.2.1", "В поле x1 введено «abc»", "Сообщение: значение не является числом")
    def _():
        return expect_error(lambda: InputValidator.validate_prediction_input(
            {"x1": "abc", "x2": "1", "x3": "2"}, feats), "не является числом")

    @t(S, "4.2.2", "Прогноз без обучения модели", "Сообщение: модель не обучена")
    def _():
        return expect_error(lambda: ExperimentService().predict_one({"x1": 1}), "не обучена")

    S = "Визуализация и экспорт"
    r = ExperimentService().train(ds, ModelParams(algorithm="ridge", target="y", features=feats))

    @t(S, "4.1.1.1", "Обученная модель; построение графика",
       "График: точки тестовой выборки и линия ŷ = y")
    def _():
        ax = PlotBuilder.actual_vs_predicted(r).axes[0]
        n = len(ax.collections[0].get_offsets())
        return f"Точек: {n}, линий: {len(ax.lines)}", n == len(r.y_test) and len(ax.lines) == 1

    @t(S, "4.1.1.1, 4.5.3", "Сохранение отчёта TXT",
       "Файл содержит параметры, коэффициенты и метрики")
    def _():
        p = os.path.join(tmp, "отчёт.txt")
        ReportExporter.save_txt(r, p, "data.csv", 500)
        with open(p, encoding="utf-8") as fh:
            text = fh.read()
        need = ["Алгоритм", "α", "Свободный член", "x1", "MSE", "MAE", "R²"]
        miss = [s for s in need if s not in text]
        return ("Все разделы присутствуют" if not miss else f"Нет: {miss}"), not miss

    @t(S, "4.1.1.1, 4.5.3", "Сохранение таблицы прогнозов CSV",
       "Файл: 100 строк «факт; прогноз; остаток»")
    def _():
        p = os.path.join(tmp, "прогнозы.csv")
        ReportExporter.save_predictions_csv(r, p)
        df = pd.read_csv(p, sep=";")
        return f"{len(df)} строк, столбцы: {', '.join(df.columns)}", len(df) == 100

    @t(S, "4.1.1.1, 4.5.3", "Сохранение графика PNG", "Создан корректный файл PNG")
    def _():
        p = os.path.join(tmp, "график.png")
        ReportExporter.save_plot_png(PlotBuilder.actual_vs_predicted(r), p)
        with open(p, "rb") as fh:
            ok = fh.read(8) == b"\x89PNG\r\n\x1a\n"
        return f"Файл {os.path.getsize(p) // 1024} КБ, формат PNG: {'да' if ok else 'нет'}", ok

    @t(S, "4.2.3", "Сохранение в несуществующую папку", "Сообщение об ошибке сохранения")
    def _():
        return expect_error(lambda: ReportExporter.save_txt(
            r, os.path.join(tmp, "нет_папки", "x.txt")), "папка не существует")

    return T


# ===========================================================================
# Тесты интерфейса (блокировки и устойчивость)
# ===========================================================================
def gui_tests(F):
    T = []
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
        from ui.help_window import HelpWindow
        from ui.main_window import TAB_PLOT, TAB_PREDICT, MainWindow
    except ImportError as e:
        reason = f"не выполнен: {e}"
        return [("Интерфейс", "4.2.2", "—", "—", lambda: (reason, None))]

    messages = []
    QMessageBox.warning = lambda *a: messages.append(a[2])
    QMessageBox.critical = lambda *a: messages.append(a[2])
    app = QApplication.instance() or QApplication(sys.argv)
    state = {}

    def window():
        if "w" not in state:
            state["w"] = MainWindow()
        return state["w"]

    def load(key):
        QFileDialog.getOpenFileName = lambda *a, **k: (F[key], "")
        window().on_load_csv()

    def t(req, data, expected):
        def deco(fn):
            T.append(("Интерфейс", req, data, expected, fn))
            return fn
        return deco

    def yn(x):
        return "доступна" if x else "недоступна"

    @t("4.2.2", "Запуск программы, данные не загружены",
       "«Обучить», «Прогноз», «График», сохранение недоступны")
    def _():
        w = window()
        st = [w.btn_train.isEnabled(), w.tabs.isTabEnabled(TAB_PREDICT),
              w.tabs.isTabEnabled(TAB_PLOT), w.act_save_txt.isEnabled()]
        return (f"«Обучить» {yn(st[0])}, вкладки прогноза/графика "
                f"{'доступны' if st[1] or st[2] else 'недоступны'}, сохранение {yn(st[3])}",
                not any(st))

    @t("4.1.1.1, 4.2.2", "Загрузка корректного файла через диалог",
       "Показаны 100 строк, число строк и столбцов; «Обучить» доступна")
    def _():
        load("ok")
        w = window()
        return (f"{w.lbl_data_info.text()} Строк в таблице: {w.tbl_preview.rowCount()}; "
                f"«Обучить» {yn(w.btn_train.isEnabled())}",
                w.tbl_preview.rowCount() == 100 and w.btn_train.isEnabled())

    @t("4.2.2", "Сняты все входные признаки", "«Обучить» недоступна")
    def _():
        w = window()
        w._check_all(False)
        res = not w.btn_train.isEnabled()
        w._check_all(True)
        return f"«Обучить» {yn(not res)}", res

    @t("4.2.2", "Выбран целевой признак x1",
       "x1 исключён из входных и недоступен для отметки")
    def _():
        w = window()
        w.cmb_target.setCurrentText("x1")
        feats = w._checked_features()
        item = [w.lst_features.item(i) for i in range(w.lst_features.count())
                if w.lst_features.item(i).text() == "x1"][0]
        from PySide6.QtCore import Qt
        enabled = bool(item.flags() & Qt.ItemFlag.ItemIsEnabled)
        w.cmb_target.setCurrentText("y")
        w._check_all(True)
        return (f"Входные: {', '.join(feats)}; пункт x1 {'доступен' if enabled else 'недоступен'}",
                "x1" not in feats and not enabled)

    @t("4.1.2.3", "Выбор алгоритма: линейная, затем Ridge",
       "Поле α недоступно для линейной и доступно для Ridge")
    def _():
        w = window()
        w.cmb_algo.setCurrentIndex(w.cmb_algo.findData("linear"))
        a = w.sp_alpha.isEnabled()
        w.cmb_algo.setCurrentIndex(w.cmb_algo.findData("ridge"))
        b = w.sp_alpha.isEnabled()
        return f"Линейная: α {'доступно' if a else 'недоступно'}; Ridge: α " \
               f"{'доступно' if b else 'недоступно'}", (not a) and b

    @t("4.2.2", "Нажата «Обучить модель»",
       "Выведены результаты; прогноз, график и сохранение стали доступны")
    def _():
        w = window()
        messages.clear()
        w.on_train()
        st = [w.trained, w.tabs.isTabEnabled(TAB_PREDICT), w.tabs.isTabEnabled(TAB_PLOT),
              w.act_save_txt.isEnabled(), w.tbl_coef.rowCount() == 4]
        return (f"Коэффициентов в таблице: {w.tbl_coef.rowCount() - 1} + свободный член; "
                f"прогноз/график/сохранение {'доступны' if all(st[1:4]) else 'недоступны'}",
                all(st) and not messages)

    @t("4.1.1.1", "Нажата «Сравнить алгоритмы»", "Таблица сравнения из 3 строк")
    def _():
        w = window()
        w.on_compare()
        return f"Строк в таблице: {w.tbl_compare.rowCount()}", w.tbl_compare.rowCount() == 3

    @t("4.2.1, 4.2.3", "Прогноз: в поле признака введено «abc»",
       "Сообщение об ошибке, программа продолжает работу")
    def _():
        w = window()
        messages.clear()
        for e in w.prediction_inputs.values():
            e.setText("1")
        list(w.prediction_inputs.values())[0].setText("abc")
        w.on_predict()
        return (f"Сообщение: «{messages[-1]}»" if messages else "Сообщения нет",
                bool(messages) and w.isEnabled())

    @t("4.2.3", "Загрузка файла с разделителем «,» после работы с данными",
       "Сообщение об ошибке; прежний набор и модель сохранены")
    def _():
        w = window()
        messages.clear()
        load("comma")
        return (f"Сообщение: «{messages[-1]}»; набор: {w.dataset.row_count()} строк, "
                f"модель {'обучена' if w.trained else 'сброшена'}",
                bool(messages) and w.dataset.row_count() == 500 and w.trained)

    @t("4.2.2", "Загрузка нового корректного набора после обучения",
       "Модель сброшена; прогноз, график и сохранение снова недоступны")
    def _():
        w = window()
        load("rows20")
        st = [w.trained, w.tabs.isTabEnabled(TAB_PREDICT), w.act_save_txt.isEnabled()]
        return (f"Модель {'обучена' if st[0] else 'сброшена'}; прогноз "
                f"{yn(st[1])}; сохранение {yn(st[2])}", not any(st))

    @t("4.1.1.1, 5.2", "Меню «Справка» (F1)",
       "Окно справки с описанием функций и формата входного файла")
    def _():
        h = HelpWindow()
        heads = [s[0] for s in h.sections]
        ok = "Функции программы" in heads and "Формат входного файла" in heads
        return f"Разделы: {', '.join(heads)}", ok

    return T


# ===========================================================================
# Прогон
# ===========================================================================
def run_all() -> list[dict]:
    tmp = tempfile.mkdtemp()
    F = make_test_files(os.path.join(tmp, "files"))
    cases = core_tests(F, tmp) + gui_tests(F)
    rows = []
    for i, (section, req, data, expected, fn) in enumerate(cases, start=1):
        try:
            actual, ok = fn()
        except Exception as e:  # noqa: BLE001
            actual, ok = f"Непредвиденная ошибка: {e!r}", False
            traceback.print_exc()
        rows.append({"№": i, "Раздел": section, "Пункт ТЗ": req,
                     "Исходные данные": data, "Ожидаемый результат": expected,
                     "Фактический результат": actual,
                     "Вывод": "пройден" if ok else ("не выполнен" if ok is None
                                                    else "НЕ ПРОЙДЕН")})
    return rows


def main():
    rows = run_all()
    for r in rows:
        mark = {"пройден": "ok", "не выполнен": "--"}.get(r["Вывод"], "!!")
        print(f"[{mark}] {r['№']:2}. {r['Раздел']}: {r['Исходные данные']}\n"
              f"        → {r['Фактический результат']}")
    passed = sum(r["Вывод"] == "пройден" for r in rows)
    print(f"\nПройдено {passed} из {len(rows)}")
    out = os.path.join(HERE, "results")
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out, "functional_tests.csv"),
                              sep=";", index=False, encoding="utf-8-sig")
    print(f"Таблица: {os.path.join(out, 'functional_tests.csv')}")


if __name__ == "__main__":
    main()
