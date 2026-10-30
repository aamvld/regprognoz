"""Проверка ядра «РегПрогноз» без интерфейса.

Запуск:  python check_core.py
Скрипт создаёт тестовые CSV в папке samples и прогоняет по ним все функции.
"""

import os

import numpy as np

from core.csv_reader import CsvReader
from core.errors import ValidationError
from core.experiment import ExperimentService
from core.generator import SyntheticGenerator
from core.params import ModelParams
from core.validator import InputValidator

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(HERE, "samples")
os.makedirs(SAMPLES, exist_ok=True)


def title(text):
    print("\n" + "=" * 60 + "\n" + text + "\n" + "=" * 60)


# 1. Генерация синтетических данных без шума -------------------------------
title("1. Генерация (шум 0): модель должна найти истинные коэффициенты")
gen = SyntheticGenerator(n_samples=500, n_features=3, noise=0.0, seed=42)
syn = gen.generate()
print("Истинная зависимость:", syn.coefficients_text())

p = ModelParams(algorithm="linear", target="y", features=["x1", "x2", "x3"])
res = ExperimentService().train(syn, p)
print("Найдено моделью:     ", res.equation_text())
print(f"R² = {res.metrics.r2:.6f}  (ожидаем 1.0)")

# 2. Сохранение и загрузка CSV ---------------------------------------------
title("2. Сохранение в CSV и повторная загрузка")
gen_noisy = SyntheticGenerator(n_samples=500, n_features=3, noise=5.0, seed=7)
syn_noisy = gen_noisy.generate()
path = os.path.join(SAMPLES, "synthetic_500x3.csv")
SyntheticGenerator.save_csv(syn_noisy, path)
ds = CsvReader().read(path)
print(f"Файл: {path}")
print(f"Строк: {ds.row_count()}, столбцов: {ds.column_count()}")
print(ds.preview(5))

# 3. Обучение трёх алгоритмов и сравнение ----------------------------------
title("3. Сравнение алгоритмов (шум 5, α = 1, стандартизация вкл.)")
p = ModelParams(target="y", features=["x1", "x2", "x3"], alpha=1.0)
print(f"{'Алгоритм':32}{'MSE':>10}{'MAE':>10}{'R²':>10}")
for r in ExperimentService().compare(ds, p):
    name, mse, mae, r2 = r.summary_row()
    print(f"{name:32}{mse:10.3f}{mae:10.3f}{r2:10.4f}")

# 4. Прогноз для одного наблюдения -----------------------------------------
title("4. Прогноз для одного наблюдения")
svc = ExperimentService()
res = svc.train(ds, ModelParams(target="y", features=["x1", "x2", "x3"]))
raw = {"x1": "1", "x2": "0,5", "x3": "-2"}   # как будто введено в поля
values = InputValidator.validate_prediction_input(raw, p.features)
y_hat = svc.predict_one(values)
manual = res.intercept + sum(b * values[f] for b, f in
                             zip(res.coefficients, p.features))
print(f"Прогноз программы: {y_hat:.4f}")
print(f"Ручной расчёт по уравнению: {manual:.4f}  (должны совпасть)")

# 5. Сверка метрик со scikit-learn -----------------------------------------
title("5. Сверка наших формул метрик со scikit-learn")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
print("MSE совпадает:", np.isclose(res.metrics.mse,
                                   mean_squared_error(res.y_test, res.y_pred)))
print("MAE совпадает:", np.isclose(res.metrics.mae,
                                   mean_absolute_error(res.y_test, res.y_pred)))
print("R²  совпадает:", np.isclose(res.metrics.r2,
                                   r2_score(res.y_test, res.y_pred)))

# 5а. Сверка наших алгоритмов со scikit-learn (эталон) ----------------------
title("5а. Сверка собственных алгоритмов со scikit-learn")
try:
    import warnings
    from sklearn.linear_model import Lasso, LinearRegression, Ridge
    from sklearn.model_selection import train_test_split
    from core.preparer import DataPreparer
    from core.solvers import LeastSquaresEstimator

    warnings.filterwarnings("ignore")
    X = ds.frame[["x1", "x2", "x3"]].to_numpy()
    y = ds.frame["y"].to_numpy()
    pairs = [("Линейная", LeastSquaresEstimator(None), LinearRegression()),
             ("Ridge α=1", LeastSquaresEstimator("l2", 1.0), Ridge(alpha=1.0)),
             ("Lasso α=0.5", LeastSquaresEstimator("l1", 0.5),
              Lasso(alpha=0.5, max_iter=100_000, tol=1e-10))]
    for name, ours, ref in pairs:
        ours.fit(X, y)
        ref.fit(X, y)
        diff = np.max(np.abs(np.append(ours.coef_, ours.intercept_)
                             - np.append(ref.coef_, ref.intercept_)))
        print(f"  {name:12} макс. разница коэффициентов: {diff:.2e}")
    same = all(np.array_equal(a, b) for a, b in zip(
        train_test_split(X, y, test_size=0.2, random_state=42),
        DataPreparer.split(X, y, 0.2, 42)))
    print("  Разбиение выборки совпадает с train_test_split:", same)
except ImportError:
    print("  scikit-learn не установлен — сверка пропущена")

# 6. Пропуски --------------------------------------------------------------
title("6. Обработка пропусков")
frame = syn_noisy.frame.copy()
frame.loc[[0, 5, 10], "x2"] = np.nan
path_nan = os.path.join(SAMPLES, "with_missing.csv")
frame.to_csv(path_nan, sep=";", index=False)
ds_nan = CsvReader().read(path_nan)
print("Пропусков в файле:", ds_nan.missing_count())
for method in ("drop", "mean"):
    r = ExperimentService().train(ds_nan, ModelParams(
        target="y", features=["x1", "x2", "x3"], missing_method=method))
    print(f"  способ {method}: R² = {r.metrics.r2:.4f}")

# 7. Ошибочные ситуации: программа не падает, а сообщает причину ------------
title("7. Ошибки ввода (должны быть сообщения, а не падение)")
bad_comma = os.path.join(SAMPLES, "bad_comma_separator.csv")
syn_noisy.frame.to_csv(bad_comma, sep=",", index=False)
bad_small = os.path.join(SAMPLES, "bad_19_rows.csv")
syn_noisy.frame.head(19).to_csv(bad_small, sep=";", index=False)

cases = [
    ("Разделитель «,»", lambda: CsvReader().read(bad_comma)),
    ("19 строк", lambda: CsvReader().read(bad_small)),
    ("Файл не .csv", lambda: CsvReader().read(__file__)),
    ("α = 0 для Ridge", lambda: ExperimentService().train(ds, ModelParams(
        algorithm="ridge", target="y", features=["x1"], alpha=0))),
    ("Доля теста 50 %", lambda: ExperimentService().train(ds, ModelParams(
        target="y", features=["x1"], test_size=0.5))),
    ("Цель = вход", lambda: ExperimentService().train(ds, ModelParams(
        target="y", features=["y", "x1"]))),
    ("Генерация n = 19", lambda: SyntheticGenerator(n_samples=19).generate()),
    ("Прогноз: текст", lambda: InputValidator.validate_prediction_input(
        {"x1": "abc"}, ["x1"])),
]
for name, action in cases:
    try:
        action()
        print(f"  [!] {name}: ошибка НЕ обнаружена")
    except ValidationError as e:
        print(f"  [ok] {name}: {e}")

title("Проверка завершена")
