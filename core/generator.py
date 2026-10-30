"""Класс SyntheticGenerator — генерация синтетических данных (рис. 14)."""

import numpy as np
import pandas as pd

from .dataset import Dataset, SyntheticDataset
from .errors import save_error
from .validator import InputValidator


class SyntheticGenerator:
    """Генерирует y = b0 + b1·x1 + ... + bp·xp + ε, ε ~ N(0, noise²)."""

    def __init__(self, n_samples: int = 500, n_features: int = 3,
                 noise: float = 1.0, seed: int = 42):
        # значения по умолчанию — из п. 4.1.2.2 ТЗ
        self.n_samples = n_samples
        self.n_features = n_features
        self.noise = noise
        self.seed = seed

    def generate(self) -> SyntheticDataset:
        InputValidator.validate_gen_params(
            self.n_samples, self.n_features, self.noise, self.seed)

        rng = np.random.default_rng(self.seed)          # генератор с seed
        b0 = float(np.round(rng.uniform(-10, 10), 3))    # свободный член
        b = np.round(rng.uniform(-5, 5, self.n_features), 3)  # коэффициенты
        X = rng.normal(0.0, 1.0, (self.n_samples, self.n_features))
        eps = rng.normal(0.0, self.noise, self.n_samples)  # шум уровня σ
        y = b0 + X @ b + eps

        cols = [f"x{i}" for i in range(1, self.n_features + 1)]
        frame = pd.DataFrame(np.round(X, 4), columns=cols)
        frame["y"] = np.round(y, 4)
        return SyntheticDataset(frame, b0, [float(v) for v in b])

    @staticmethod
    def save_csv(ds: Dataset, path: str) -> None:
        """Сохраняет набор в формате входного файла (п. 4.1.2.1 ТЗ)."""
        try:
            ds.frame.to_csv(path, sep=";", index=False, encoding="utf-8")
        except OSError as e:
            raise save_error("данные", path, e)
