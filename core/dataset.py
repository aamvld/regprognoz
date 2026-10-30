"""Классы Dataset и SyntheticDataset (диаграмма классов, рис. 4а)."""

import pandas as pd


class Dataset:
    """Табличный набор данных: обёртка над pandas.DataFrame."""

    def __init__(self, frame: pd.DataFrame, source_path: str = ""):
        self.frame = frame              # сами данные
        self.source_path = source_path  # путь к исходному файлу ("" для синтетики)

    def row_count(self) -> int:
        """Число строк."""
        return len(self.frame)

    def column_count(self) -> int:
        """Число столбцов."""
        return self.frame.shape[1]

    def numeric_columns(self) -> list[str]:
        """Имена числовых столбцов: только их можно выбрать как признаки."""
        return list(self.frame.select_dtypes(include="number").columns)

    def preview(self, n: int = 100) -> pd.DataFrame:
        """Первые n строк для показа в таблице (п. 4.1.1.1 ТЗ — 100 строк)."""
        return self.frame.head(n)

    def has_missing(self, cols: list[str] | None = None) -> bool:
        """Есть ли пропуски (во всём наборе или в указанных столбцах)."""
        part = self.frame if cols is None else self.frame[cols]
        return bool(part.isna().any().any())

    def missing_count(self, cols: list[str] | None = None) -> int:
        """Общее число пропущенных значений."""
        part = self.frame if cols is None else self.frame[cols]
        return int(part.isna().sum().sum())


class SyntheticDataset(Dataset):
    """Сгенерированный набор данных + истинные коэффициенты зависимости."""

    def __init__(self, frame: pd.DataFrame, true_intercept: float,
                 true_coefficients: list[float]):
        super().__init__(frame, source_path="")
        self.true_intercept = true_intercept
        self.true_coefficients = true_coefficients

    def coefficients_text(self) -> str:
        """Формула зависимости для показа пользователю (п. 4.1.2.2 ТЗ)."""
        parts = [f"{self.true_intercept:.4f}"]
        for i, b in enumerate(self.true_coefficients, start=1):
            sign = "+" if b >= 0 else "-"
            parts.append(f" {sign} {abs(b):.4f}·x{i}")
        return "y = " + "".join(parts) + " + ε"
