# РегПрогноз — курсовая работа по ТРПС

Программная система решения задачи прогнозирования с использованием
регрессионного анализа. Студент: А. О. Кортова, ИУ6-55Б.

## Первый запуск (Windows)
Открыть папку проекта в VS Code, открыть терминал (Terminal → New Terminal):

    py -3.12 -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

В каждом новом терминале сначала: `.venv\Scripts\activate`

## Команды
| Команда | Что делает |
|---|---|
| `python main.py` | запуск программы |
| `python check_core.py` | быстрая проверка ядра, сверка со scikit-learn |
| `python check_gui.py` | автоматическая проверка окна |
| `python -m unittest -v` | модульные тесты (46 шт.) |
| `python functional_tests.py` | функциональные тесты по ТЗ (61 шт.) |
| `python check_performance.py` | замер временных характеристик (п. 4.1.1.2 ТЗ) |
| `python make_tables.py` | **Word-файл с таблицами для РПЗ** → `results\Таблицы_для_РПЗ.docx` |
| `.\build.bat` | сборка `.exe` и установщика |

## Структура (соответствует диаграмме классов)
    main.py                    точка входа
    preload.py                 совместимость PySide6 с библиотекой six
    ui/main_window.py          MainWindow — главное окно, вкладки, блокировки
    ui/help_window.py          HelpWindow — справка
    core/csv_reader.py         CsvReader — чтение CSV
    core/validator.py          InputValidator — проверка ввода
    core/errors.py             ValidationError — ошибка с текстом причины
    core/dataset.py            Dataset, SyntheticDataset
    core/missing.py            MissingValueHandler — обработка пропусков
    core/generator.py          SyntheticGenerator — синтетические данные
    core/params.py             ModelParams — параметры моделирования
    core/preparer.py           DataPreparer — разбиение, стандартизация
    core/models.py             RegressionModel, LinearModel, RidgeModel, LassoModel
    core/solvers.py            LeastSquaresEstimator — МНК, Ridge, Lasso на numpy
    core/metrics.py            MetricsCalculator, Metrics — MSE, MAE, R²
    core/result.py             ModelResult — результат обучения
    core/experiment.py         ExperimentService — обучение, сравнение, прогноз
    output/plot_builder.py     PlotBuilder — график «факт – прогноз»
    output/report_exporter.py  ReportExporter — TXT, CSV, PNG
    tests/                     модульные тесты
    test_files/                CSV для ручной проверки и скриншотов ошибок
    examples/example_data.csv  пример входного файла (входит в установщик)

## Сборка дистрибутива
1. Установить Inno Setup 6 (jrsoftware.org/isdl.php).
2. Удалить папки `build` и `dist` (если есть), запустить `.\build.bat`.
3. Результат: `installer\RegPrognoz_Setup.exe` — установщик;
   `dist\RegPrognoz\RegPrognoz.exe` — программа без установки.
