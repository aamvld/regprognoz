"""Класс DataPreparer — разбиение выборки и стандартизация."""

import math

import numpy as np


class DataPreparer:
    """Разбивает данные и стандартизирует признаки: z = (x - mean) / scale.

    Среднее и стандартное отклонение считаются ТОЛЬКО по обучающей выборке,
    чтобы информация из тестовой выборки не попадала в обучение.
    """

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    @staticmethod
    def split(X, y, test_size: float, seed: int):
        """Случайное разбиение. Возвращает X_train, X_test, y_train, y_test.

        Строки перемешиваются генератором с заданным начальным значением,
        первые ⌈test_size·n⌉ из перемешанного порядка идут в тестовую
        выборку, остальные — в обучающую. При одном и том же seed разбиение
        всегда одинаковое (воспроизводимость, п. 4.1.1.1 ТЗ).
        """
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        n = len(y)
        n_test = math.ceil(test_size * n)
        perm = np.random.RandomState(seed).permutation(n)
        test_idx, train_idx = perm[:n_test], perm[n_test:]
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    def fit_scaler(self, X_train) -> None:
        X_train = np.asarray(X_train, dtype=float)
        self.mean_ = X_train.mean(axis=0)
        scale = X_train.std(axis=0)
        scale[scale == 0] = 1.0  # постоянный признак: делить на 0 нельзя
        self.scale_ = scale

    def scale(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.scale_
