"""Класс InputValidator — контроль вводимой информации (п. 4.2.1 ТЗ)."""

import os

from .dataset import Dataset
from .errors import ValidationError


class InputValidator:
    """Проверки файла, столбцов и параметров. При ошибке — ValidationError."""

    MIN_ROWS = 20
    MAX_ROWS = 10_000
    MAX_FEATURES = 20

    # ---------- файл ----------
    @staticmethod
    def validate_file(path: str) -> None:
        """Проверка до чтения: файл существует и имеет расширение .csv."""
        if not path:
            raise ValidationError("Файл не выбран.")
        if not os.path.isfile(path):
            raise ValidationError(f"Файл не найден: {path}")
        if not path.lower().endswith(".csv"):
            raise ValidationError("Файл должен иметь расширение .csv.")

    @classmethod
    def validate_dataset(cls, ds: Dataset) -> None:
        """Проверка после чтения: число строк и числовых столбцов (рис. 12)."""
        n = ds.row_count()
        if not (cls.MIN_ROWS <= n <= cls.MAX_ROWS):
            raise ValidationError(
                f"Число строк в файле — {n}. Допустимо от {cls.MIN_ROWS} "
                f"до {cls.MAX_ROWS}.")
        k = len(ds.numeric_columns())
        if not (2 <= k <= cls.MAX_FEATURES + 1):
            raise ValidationError(
                f"Числовых столбцов в файле — {k}. Нужно от 2 до "
                f"{cls.MAX_FEATURES + 1} (целевой + от 1 до {cls.MAX_FEATURES} "
                f"входных). Проверьте, что десятичный разделитель — точка.")

    # ---------- выбор признаков ----------
    @classmethod
    def validate_columns(cls, ds: Dataset, target: str,
                         features: list[str]) -> None:
        """Целевой и входные признаки выбраны, числовые и не пересекаются."""
        numeric = ds.numeric_columns()
        if not target:
            raise ValidationError("Не выбран целевой признак.")
        if target not in numeric:
            raise ValidationError(f"Целевой признак «{target}» не числовой.")
        if not features:
            raise ValidationError("Не выбран ни один входной признак.")
        if len(features) > cls.MAX_FEATURES:
            raise ValidationError(
                f"Входных признаков не может быть больше {cls.MAX_FEATURES}.")
        if target in features:
            raise ValidationError(
                "Целевой признак не может одновременно быть входным.")
        bad = [f for f in features if f not in numeric]
        if bad:
            raise ValidationError(
                "Нечисловые входные признаки: " + ", ".join(bad))

    # ---------- генерация ----------
    @classmethod
    def validate_gen_params(cls, n_samples, n_features, noise, seed) -> None:
        """Параметры генерации из п. 4.1.2.2 ТЗ."""
        if not isinstance(n_samples, int) or not (
                cls.MIN_ROWS <= n_samples <= cls.MAX_ROWS):
            raise ValidationError(
                f"Число наблюдений должно быть целым от {cls.MIN_ROWS} "
                f"до {cls.MAX_ROWS}.")
        if not isinstance(n_features, int) or not (
                1 <= n_features <= cls.MAX_FEATURES):
            raise ValidationError(
                f"Число признаков должно быть целым от 1 до {cls.MAX_FEATURES}.")
        if noise is None or noise < 0:
            raise ValidationError("Уровень шума должен быть неотрицательным.")
        if not isinstance(seed, int):
            raise ValidationError("Начальное значение генератора — целое число.")

    # ---------- моделирование ----------
    @staticmethod
    def validate_model_params(p) -> None:
        """Параметры моделирования из п. 4.1.2.3 ТЗ (p — ModelParams)."""
        if p.algorithm not in ("linear", "ridge", "lasso"):
            raise ValidationError("Неизвестный алгоритм.")
        if p.algorithm in ("ridge", "lasso") and not (p.alpha > 0):
            raise ValidationError("Параметр α должен быть больше 0.")
        if not (0.10 <= p.test_size <= 0.40):
            raise ValidationError(
                "Доля тестовой выборки должна быть от 10 % до 40 %.")
        if not isinstance(p.seed, int):
            raise ValidationError("Начальное значение генератора — целое число.")
        if p.missing_method not in ("drop", "mean"):
            raise ValidationError("Неизвестный способ обработки пропусков.")

    # ---------- прогноз ----------
    @staticmethod
    def validate_prediction_input(raw: dict[str, str],
                                  features: list[str]) -> dict[str, float]:
        """Значения для прогноза: все заданы и являются числами (рис. 17).

        Принимает строки из полей ввода, возвращает словарь чисел.
        Запятая как десятичный разделитель тоже допускается — так удобнее.
        """
        values = {}
        for f in features:
            text = str(raw.get(f, "")).strip().replace(",", ".")
            if text == "":
                raise ValidationError(f"Не задано значение признака «{f}».")
            try:
                values[f] = float(text)
            except ValueError:
                raise ValidationError(
                    f"Значение признака «{f}» не является числом: «{text}».")
        return values
