"""Совместимость библиотек. Импортируется ПЕРВЫМ в main.py.

Проблема: PySide6 устанавливает свой перехватчик импорта модулей и для
каждого нового модуля пытается прочитать его исходный текст. Для
«виртуальных» модулей библиотеки six (six.moves; её используют dateutil →
pandas и matplotlib) Python при этом обращается к атрибуту _path загрузчика
_SixMetaPathImporter, которого нет, и программа падает с ошибкой
AttributeError: '_SixMetaPathImporter' object has no attribute '_path'.

Решение:
1) добавляем загрузчику six недостающий атрибут _path — тогда перехватчик
   PySide6 получает обычную «ожидаемую» ошибку и спокойно пропускает модуль;
2) заранее загружаем используемые библиотеки (при запуске из исходников).
В собранном .exe PyInstaller сам загружает PySide6 ещё до main.py,
поэтому build.bat подключает этот файл как runtime hook
(--runtime-hook preload.py): такие скрипты выполняются РАНЬШЕ
стандартной настройки PySide6, и все библиотеки успевают загрузиться.
"""

import six

if not hasattr(six._SixMetaPathImporter, "_path"):
    six._SixMetaPathImporter._path = []

# six и все модули dateutil, которые его используют
import six.moves  # noqa: F401,E402
import dateutil.parser  # noqa: F401
import dateutil.relativedelta  # noqa: F401
import dateutil.rrule  # noqa: F401
import dateutil.tz  # noqa: F401

# Научные библиотеки и их части, которые использует программа
import numpy  # noqa: F401
import pandas  # noqa: F401
import matplotlib  # noqa: F401
import matplotlib.dates  # noqa: F401
import matplotlib.figure  # noqa: F401
import matplotlib.backends.backend_agg  # noqa: F401
