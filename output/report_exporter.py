"""Класс ReportExporter — сохранение результатов в TXT, CSV и PNG.

Файлы записываются только по пути, который выбрал пользователь
(п. 4.2.4 ТЗ). Исходный файл данных не изменяется.
"""

from datetime import datetime

import pandas as pd
from matplotlib.figure import Figure

from core.errors import save_error
from core.params import ALGORITHM_NAMES
from core.result import ModelResult


def _fmt(v: float) -> str:
    return "не определён" if v != v else f"{v:.6f}"



class ReportExporter:

    @staticmethod
    def save_txt(r: ModelResult, path: str, source: str = "",
                 n_rows: int | None = None) -> None:
        """Текстовый отчёт: параметры модели, коэффициенты, метрики."""
        p = r.params
        lines = [
            "ОТЧЁТ ПРОГРАММНОЙ СИСТЕМЫ «РегПрогноз»",
            f"Дата и время: {datetime.now():%d.%m.%Y %H:%M:%S}",
            "",
            "1. Данные",
            f"   Источник: {source or 'синтетический набор данных'}",
        ]
        if n_rows is not None:
            lines.append(f"   Число строк: {n_rows}")
        lines += [
            f"   Целевой признак: {p.target}",
            f"   Входные признаки: {', '.join(p.features)}",
            "   Обработка пропусков: " +
            ("удаление строк" if p.missing_method == "drop"
             else "замена средним значением"),
            "",
            "2. Параметры модели",
            f"   Алгоритм: {ALGORITHM_NAMES[p.algorithm]}",
        ]
        if p.algorithm in ("ridge", "lasso"):
            lines.append(f"   Параметр регуляризации α: {p.alpha}")
        lines += [
            f"   Стандартизация признаков: {'да' if p.standardize else 'нет'}",
            f"   Доля тестовой выборки: {p.test_size * 100:.0f} %",
            f"   Начальное значение генератора: {p.seed}",
            f"   Размер тестовой выборки: {len(r.y_test)}",
            "",
            "3. Коэффициенты модели (в исходной шкале признаков)",
            f"   Свободный член: {r.intercept:.6f}",
        ]
        for name, b in zip(p.features, r.coefficients):
            lines.append(f"   {name}: {b:.6f}")
        lines += [
            f"   Уравнение: {r.equation_text()}",
            "",
            "4. Показатели качества на тестовой выборке",
            f"   MSE: {_fmt(r.metrics.mse)}",
            f"   MAE: {_fmt(r.metrics.mae)}",
            f"   R²:  {_fmt(r.metrics.r2)}",
        ]
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except OSError as e:
            raise save_error("отчёт", path, e)

    @staticmethod
    def save_predictions_csv(r: ModelResult, path: str) -> None:
        """Таблица фактических и прогнозных значений тестовой выборки."""
        frame = pd.DataFrame({
            "№": range(1, len(r.y_test) + 1),
            "Фактическое": r.y_test,
            "Прогнозное": r.y_pred,
            "Остаток": r.y_test - r.y_pred,
        })
        try:
            frame.to_csv(path, sep=";", index=False, encoding="utf-8")
        except OSError as e:
            raise save_error("таблицу", path, e)

    @staticmethod
    def save_plot_png(fig: Figure, path: str) -> None:
        """Сохранение графика в PNG."""
        try:
            fig.savefig(path, dpi=150, bbox_inches="tight")
        except OSError as e:
            raise save_error("график", path, e)
