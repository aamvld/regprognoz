"""Классы Metrics и MetricsCalculator — показатели качества."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Metrics:
    mse: float  # среднеквадратичная ошибка
    mae: float  # средняя абсолютная ошибка
    r2: float   # коэффициент детерминации


class MetricsCalculator:
    """Формулы считаются вручную (их же приводим в РПЗ)."""

    @staticmethod
    def mse(y, y_hat) -> float:
        # MSE = (1/n) · Σ (yᵢ - ŷᵢ)²
        y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
        return float(np.mean((y - y_hat) ** 2))

    @staticmethod
    def mae(y, y_hat) -> float:
        # MAE = (1/n) · Σ |yᵢ - ŷᵢ|
        y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
        return float(np.mean(np.abs(y - y_hat)))

    @staticmethod
    def r2(y, y_hat) -> float:
        # R² = 1 - Σ(yᵢ - ŷᵢ)² / Σ(yᵢ - ȳ)²
        y, y_hat = np.asarray(y, float), np.asarray(y_hat, float)
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        if ss_tot == 0:  # все y одинаковые — R² не определён
            return float("nan")
        return float(1 - ss_res / ss_tot)

    def compute(self, y, y_hat) -> Metrics:
        return Metrics(self.mse(y, y_hat), self.mae(y, y_hat),
                       self.r2(y, y_hat))
