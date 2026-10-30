"""Исключение для ошибок ввода (п. 4.2.1 ТЗ).

Любая проверка, которая не прошла, выбрасывает ValidationError с понятным
текстом причины. Интерфейс перехватывает его и показывает сообщение
пользователю, после чего работа продолжается (п. 4.2.3 ТЗ).
"""


class ValidationError(Exception):
    """Ошибка входных данных или параметров с текстом причины на русском."""


def save_error(what: str, path: str, e: OSError) -> ValidationError:
    """Понятное сообщение об ошибке записи файла (п. 4.2.3 ТЗ)."""
    if isinstance(e, FileNotFoundError):
        reason = "папка не существует"
    elif isinstance(e, PermissionError):
        reason = "нет прав на запись или файл открыт в другой программе"
    else:
        reason = str(e)
    return ValidationError(f"Не удалось сохранить {what} в «{path}»: {reason}.")
