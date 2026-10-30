"""Класс CsvReader — чтение набора данных из CSV (п. 4.1.2.1 ТЗ, рис. 12)."""

import pandas as pd

from .dataset import Dataset
from .errors import ValidationError
from .validator import InputValidator


class CsvReader:
    """Читает CSV: UTF-8, разделитель «;», заголовки, десятичная точка.

    Исходный файл только читается и никогда не изменяется (п. 4.2.4 ТЗ).
    """

    def __init__(self, sep: str = ";", encoding: str = "utf-8"):
        self.sep = sep
        self.encoding = encoding

    def read(self, path: str) -> Dataset:
        InputValidator.validate_file(path)

        # "utf-8-sig" — тот же UTF-8, но корректно убирает метку BOM,
        # которую добавляет Excel при сохранении «CSV UTF-8».
        enc = "utf-8-sig" if self.encoding.lower() == "utf-8" else self.encoding
        try:
            frame = pd.read_csv(path, sep=self.sep, encoding=enc, decimal=".")
        except UnicodeDecodeError:
            raise ValidationError(
                "Файл не в кодировке UTF-8. Пересохраните его как «CSV UTF-8».")
        except pd.errors.EmptyDataError:
            raise ValidationError("Файл пустой.")
        except pd.errors.ParserError as e:
            raise ValidationError(f"Не удалось разобрать файл: {e}")

        # Типичная ошибка: файл с разделителем «,» читается как один столбец.
        if frame.shape[1] == 1 and "," in str(frame.columns[0]):
            raise ValidationError(
                "Похоже, в файле разделитель «,». Требуется «;».")

        ds = Dataset(frame, source_path=path)
        InputValidator.validate_dataset(ds)
        return ds
