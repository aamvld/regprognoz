"""Класс PlotBuilder — график «фактические – прогнозные значения»."""

import numpy as np
from matplotlib.figure import Figure

from core.params import ALGORITHM_NAMES
from core.result import ModelResult


class PlotBuilder:
    """Строит диаграмму рассеяния для тестовой выборки.

    Используется matplotlib.figure.Figure (без pyplot), чтобы график
    можно было встроить в окно программы.
    """

    @staticmethod
    def actual_vs_predicted(r: ModelResult, fig: Figure | None = None) -> Figure:
        if fig is None:
            fig = Figure(figsize=(7, 5), dpi=100)
        fig.clear()
        ax = fig.add_subplot(111)

        y, y_hat = np.asarray(r.y_test), np.asarray(r.y_pred)
        ax.scatter(y, y_hat, s=18, alpha=0.7, label="Наблюдения тестовой выборки")

        # Линия идеального прогноза: точки на ней — прогноз точно равен факту
        lo = float(min(y.min(), y_hat.min()))
        hi = float(max(y.max(), y_hat.max()))
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5,
                label="Идеальный прогноз (ŷ = y)")

        ax.set_xlabel(f"Фактическое значение «{r.params.target}»")
        ax.set_ylabel(f"Прогнозное значение «{r.params.target}»")
        r2 = r.metrics.r2
        r2_text = "не определён" if r2 != r2 else f"{r2:.4f}"
        ax.set_title(f"{ALGORITHM_NAMES[r.params.algorithm]}: R² = {r2_text}")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
        fig.tight_layout()
        return fig
