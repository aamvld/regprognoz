"""Регрессионные модели (рис. 4б): абстрактный класс и три алгоритма."""

from abc import ABC, abstractmethod

import numpy as np

from .solvers import LeastSquaresEstimator


class RegressionModel(ABC):
    """Общий интерфейс модели: обучение, прогноз, коэффициенты."""

    def __init__(self):
        self._estimator = None
        self.intercept_ = 0.0
        self.coef_ = []
        self.fitted = False

    @abstractmethod
    def _build_estimator(self):
        """Создаёт решатель для конкретного алгоритма (переопределяется)."""

    def fit(self, X, y) -> None:
        self._estimator = self._build_estimator()
        self._estimator.fit(X, y)
        self.intercept_ = float(self._estimator.intercept_)
        self.coef_ = [float(c) for c in np.ravel(self._estimator.coef_)]
        self.fitted = True

    def predict(self, X):
        return self._estimator.predict(X)

    def intercept(self) -> float:
        return self.intercept_

    def coefficients(self) -> list[float]:
        return self.coef_


class LinearModel(RegressionModel):
    """Линейная регрессия (метод наименьших квадратов)."""

    def _build_estimator(self):
        return LeastSquaresEstimator(penalty=None)


class RidgeModel(RegressionModel):
    """Гребневая регрессия: штраф α·Σb² (L2)."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def _build_estimator(self):
        return LeastSquaresEstimator(penalty="l2", alpha=self.alpha)


class LassoModel(RegressionModel):
    """Лассо-регрессия: штраф α·Σ|b| (L1), может обнулять коэффициенты."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def _build_estimator(self):
        return LeastSquaresEstimator(penalty="l1", alpha=self.alpha)


def create_model(algorithm: str, alpha: float) -> RegressionModel:
    """Фабрика: по названию алгоритма возвращает объект модели."""
    if algorithm == "linear":
        return LinearModel()
    if algorithm == "ridge":
        return RidgeModel(alpha)
    if algorithm == "lasso":
        return LassoModel(alpha)
    raise ValueError(f"Неизвестный алгоритм: {algorithm}")
