"""Класс ExperimentService — обучение, сравнение, прогноз (рис. 15–17)."""

import numpy as np

from .dataset import Dataset
from .errors import ValidationError
from .metrics import MetricsCalculator
from .missing import MissingValueHandler
from .models import RegressionModel, create_model
from .params import ModelParams
from .preparer import DataPreparer
from .result import ModelResult
from .validator import InputValidator


class ExperimentService:
    """Связывает подготовку данных, модель и метрики в один процесс."""

    def __init__(self):
        self.preparer = DataPreparer()
        self.calculator = MetricsCalculator()
        self.model: RegressionModel | None = None
        self.last_params: ModelParams | None = None

    # ---------- обучение одной модели (рис. 15) ----------
    def train(self, ds: Dataset, p: ModelParams) -> ModelResult:
        InputValidator.validate_columns(ds, p.target, p.features)
        InputValidator.validate_model_params(p)

        # 1. Обработка пропусков в выбранных столбцах
        ds = MissingValueHandler.apply(ds, [p.target] + p.features,
                                       p.missing_method)
        X = ds.frame[p.features].to_numpy(dtype=float)
        y = ds.frame[p.target].to_numpy(dtype=float)

        # 2. Разбиение на обучающую и тестовую выборки
        X_tr, X_te, y_tr, y_te = self.preparer.split(X, y, p.test_size, p.seed)

        # 3. Стандартизация (параметры — по обучающей выборке)
        if p.standardize:
            self.preparer.fit_scaler(X_tr)
            X_tr, X_te = self.preparer.scale(X_tr), self.preparer.scale(X_te)

        # 4. Обучение и прогноз для тестовой выборки
        model = create_model(p.algorithm, p.alpha)
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        # 5. Метрики качества
        metrics = self.calculator.compute(y_te, y_pred)

        # 6. Коэффициенты в исходной шкале признаков.
        # Модель обучена на z = (x - μ)/σ: ŷ = w0 + Σ wᵢ·zᵢ.
        # Отсюда bᵢ = wᵢ/σᵢ и b0 = w0 - Σ wᵢ·μᵢ/σᵢ — их и показываем,
        # чтобы коэффициенты относились к настоящим значениям признаков.
        w0, w = model.intercept(), np.array(model.coefficients())
        if p.standardize:
            b = w / self.preparer.scale_
            b0 = w0 - float(np.sum(w * self.preparer.mean_ / self.preparer.scale_))
        else:
            b, b0 = w, w0

        self.model = model
        self.last_params = p
        return ModelResult(p, float(b0), [float(v) for v in b], metrics,
                           y_te, y_pred)

    # ---------- сравнение трёх алгоритмов (рис. 16) ----------
    def compare(self, ds: Dataset, p: ModelParams) -> list[ModelResult]:
        results = []
        for algorithm in ("linear", "ridge", "lasso"):
            q = ModelParams(**{**p.__dict__, "algorithm": algorithm})
            results.append(ExperimentService().train(ds, q))  # те же данные и seed
        return results

    # ---------- прогноз для одного наблюдения (рис. 17) ----------
    def predict_one(self, values: dict[str, float]) -> float:
        if self.model is None or not self.model.fitted:
            raise ValidationError("Модель ещё не обучена.")
        p = self.last_params
        x = np.array([[values[f] for f in p.features]], dtype=float)
        if p.standardize:
            x = self.preparer.scale(x)  # тот же масштаб, что при обучении
        return float(self.model.predict(x)[0])
