"""Класс ModelResult — результат обучения одной модели."""

from dataclasses import dataclass

import numpy as np

from .metrics import Metrics
from .params import ALGORITHM_NAMES, ModelParams


@dataclass
class ModelResult:
    params: ModelParams
    intercept: float            # свободный член (в исходной шкале признаков)
    coefficients: list[float]   # коэффициенты при входных признаках
    metrics: Metrics            # MSE, MAE, R² по тестовой выборке
    y_test: np.ndarray          # фактические значения тестовой выборки
    y_pred: np.ndarray          # прогнозные значения тестовой выборки

    def summary_row(self) -> list:
        """Строка сводной таблицы сравнения: алгоритм, MSE, MAE, R²."""
        return [ALGORITHM_NAMES[self.params.algorithm],
                self.metrics.mse, self.metrics.mae, self.metrics.r2]

    def equation_text(self) -> str:
        """Уравнение модели в виде текста."""
        s = f"{self.params.target} = {self.intercept:.4f}"
        for name, b in zip(self.params.features, self.coefficients):
            sign = "+" if b >= 0 else "-"
            s += f" {sign} {abs(b):.4f}·{name}"
        return s
