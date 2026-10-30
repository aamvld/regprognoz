"""Класс MissingValueHandler — обработка пропусков (рис. 13)."""

from .dataset import Dataset, SyntheticDataset
from .errors import ValidationError
from .validator import InputValidator


class MissingValueHandler:
    """Удаляет строки с пропусками или заменяет пропуски средним."""

    @staticmethod
    def apply(ds: Dataset, cols: list[str], method: str) -> Dataset:
        """Возвращает НОВЫЙ набор; исходный не меняется.

        cols   — столбцы, в которых ищутся пропуски (целевой + входные);
        method — "drop" (удалить строки) или "mean" (заменить средним).
        """
        if not ds.has_missing(cols):
            return ds  # пропусков нет — набор без изменений

        frame = ds.frame.copy()
        if method == "drop":
            frame = frame.dropna(subset=cols).reset_index(drop=True)
            if len(frame) < InputValidator.MIN_ROWS:
                raise ValidationError(
                    f"После удаления строк с пропусками осталось {len(frame)} "
                    f"строк, нужно не менее {InputValidator.MIN_ROWS}. "
                    f"Выберите замену средним значением.")
        elif method == "mean":
            for c in cols:
                if frame[c].isna().any():
                    mean = frame[c].mean()  # среднее по непропущенным
                    if mean != mean:        # NaN: весь столбец пустой
                        raise ValidationError(
                            f"Столбец «{c}» не содержит ни одного значения.")
                    frame[c] = frame[c].fillna(mean)
        else:
            raise ValidationError("Неизвестный способ обработки пропусков.")

        # у синтетического набора сохраняем истинные коэффициенты
        if isinstance(ds, SyntheticDataset):
            return SyntheticDataset(frame, ds.true_intercept,
                                    ds.true_coefficients)
        return Dataset(frame, ds.source_path)
