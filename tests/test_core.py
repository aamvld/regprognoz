"""Модульные тесты ядра (метод «белого ящика»).

Запуск из папки проекта:  python -m unittest -v
Каждый тест проверяет один класс или одну функцию изолированно.
"""

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from core.csv_reader import CsvReader
from core.dataset import Dataset, SyntheticDataset
from core.errors import ValidationError
from core.experiment import ExperimentService
from core.generator import SyntheticGenerator
from core.metrics import MetricsCalculator
from core.missing import MissingValueHandler
from core.models import LassoModel, LinearModel, RidgeModel, create_model
from core.params import ModelParams
from core.preparer import DataPreparer
from core.solvers import LeastSquaresEstimator
from core.validator import InputValidator


def make_frame(n=50, p=3, seed=0, noise=0.5):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = 2.0 + X @ np.arange(1, p + 1) + rng.normal(0, noise, n)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(1, p + 1)])
    df["y"] = y
    return df


class TempDirMixin:
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def path(self, name):
        return os.path.join(self.tmp, name)


# ---------------------------------------------------------------- Dataset
class TestDataset(unittest.TestCase):
    def test_sizes_and_numeric_columns(self):
        df = make_frame(30, 2)
        df["name"] = "a"
        ds = Dataset(df)
        self.assertEqual(ds.row_count(), 30)
        self.assertEqual(ds.column_count(), 4)
        self.assertEqual(ds.numeric_columns(), ["x1", "x2", "y"])

    def test_preview_limited_to_100(self):
        self.assertEqual(len(Dataset(make_frame(250)).preview(100)), 100)

    def test_missing_detection(self):
        df = make_frame(30)
        df.loc[3, "x2"] = np.nan
        ds = Dataset(df)
        self.assertTrue(ds.has_missing())
        self.assertFalse(ds.has_missing(["x1", "y"]))
        self.assertEqual(ds.missing_count(), 1)

    def test_coefficients_text(self):
        ds = SyntheticDataset(make_frame(), 1.5, [2.0, -3.0])
        self.assertEqual(ds.coefficients_text(),
                         "y = 1.5000 + 2.0000·x1 - 3.0000·x2 + ε")


# ---------------------------------------------------------------- CsvReader
class TestCsvReader(TempDirMixin, unittest.TestCase):
    def test_reads_valid_file(self):
        make_frame(40).to_csv(self.path("ok.csv"), sep=";", index=False)
        ds = CsvReader().read(self.path("ok.csv"))
        self.assertEqual((ds.row_count(), ds.column_count()), (40, 4))

    def test_reads_utf8_with_bom_and_cyrillic_header(self):
        df = make_frame(25).rename(columns={"y": "цена"})
        df.to_csv(self.path("bom.csv"), sep=";", index=False, encoding="utf-8-sig")
        ds = CsvReader().read(self.path("bom.csv"))
        self.assertIn("цена", ds.numeric_columns())
        self.assertEqual(ds.frame.columns[0], "x1")   # BOM не попал в имя

    def test_source_file_not_modified(self):
        make_frame(40).to_csv(self.path("src.csv"), sep=";", index=False)
        with open(self.path("src.csv"), "rb") as f:
            before = f.read()
        CsvReader().read(self.path("src.csv"))
        with open(self.path("src.csv"), "rb") as f:
            self.assertEqual(before, f.read())

    def test_errors(self):
        cases = {
            "comma.csv": lambda p: make_frame(40).to_csv(p, sep=",", index=False),
            "small.csv": lambda p: make_frame(19).to_csv(p, sep=";", index=False),
            "cp1251.csv": lambda p: make_frame(40).rename(columns={"y": "цена"}).to_csv(
                p, sep=";", index=False, encoding="cp1251"),
            "empty.csv": lambda p: open(p, "w").close(),
            "onecol.csv": lambda p: make_frame(40)[["y"]].to_csv(p, sep=";", index=False),
        }
        for name, create in cases.items():
            with self.subTest(file=name):
                create(self.path(name))
                with self.assertRaises(ValidationError):
                    CsvReader().read(self.path(name))

    def test_wrong_extension_and_missing_file(self):
        with self.assertRaises(ValidationError):
            CsvReader().read(self.path("data.txt"))
        with self.assertRaises(ValidationError):
            CsvReader().read(self.path("нет_такого.csv"))


# ---------------------------------------------------------------- InputValidator
class TestInputValidator(unittest.TestCase):
    def setUp(self):
        self.ds = Dataset(make_frame(30))

    def test_columns_ok(self):
        InputValidator.validate_columns(self.ds, "y", ["x1", "x2"])

    def test_columns_errors(self):
        for target, feats in (("", ["x1"]), ("y", []), ("y", ["y", "x1"]),
                              ("y", ["нет"]), ("нет", ["x1"])):
            with self.subTest(target=target, feats=feats):
                with self.assertRaises(ValidationError):
                    InputValidator.validate_columns(self.ds, target, feats)

    def test_gen_params_boundaries(self):
        ok = [(20, 1, 0.0, 0), (10_000, 20, 5.0, 42)]
        bad = [(19, 3, 1.0, 42), (10_001, 3, 1.0, 42), (500, 0, 1.0, 42),
               (500, 21, 1.0, 42), (500, 3, -0.1, 42)]
        for args in ok:
            InputValidator.validate_gen_params(*args)
        for args in bad:
            with self.subTest(args=args), self.assertRaises(ValidationError):
                InputValidator.validate_gen_params(*args)

    def test_model_params_boundaries(self):
        for ts in (0.10, 0.40):
            InputValidator.validate_model_params(ModelParams(test_size=ts))
        for p in (ModelParams(test_size=0.09), ModelParams(test_size=0.41),
                  ModelParams(algorithm="ridge", alpha=0),
                  ModelParams(algorithm="lasso", alpha=-1),
                  ModelParams(algorithm="xxx")):
            with self.subTest(p=p), self.assertRaises(ValidationError):
                InputValidator.validate_model_params(p)
        # для линейной регрессии α не используется и не проверяется
        InputValidator.validate_model_params(ModelParams(algorithm="linear", alpha=0))

    def test_prediction_input(self):
        v = InputValidator.validate_prediction_input(
            {"a": "1.5", "b": " -2,25 "}, ["a", "b"])
        self.assertEqual(v, {"a": 1.5, "b": -2.25})
        for raw in ({"a": "", "b": "1"}, {"a": "abc", "b": "1"}, {"b": "1"}):
            with self.subTest(raw=raw), self.assertRaises(ValidationError):
                InputValidator.validate_prediction_input(raw, ["a", "b"])


# ---------------------------------------------------------------- MissingValueHandler
class TestMissingValueHandler(unittest.TestCase):
    def setUp(self):
        df = make_frame(30)
        df.loc[[0, 1], "x1"] = np.nan
        self.ds = Dataset(df)

    def test_drop(self):
        out = MissingValueHandler.apply(self.ds, ["x1", "y"], "drop")
        self.assertEqual(out.row_count(), 28)
        self.assertFalse(out.has_missing())
        self.assertEqual(self.ds.row_count(), 30)       # исходный не изменён

    def test_mean(self):
        mean = self.ds.frame["x1"].mean()
        out = MissingValueHandler.apply(self.ds, ["x1", "y"], "mean")
        self.assertEqual(out.row_count(), 30)
        self.assertAlmostEqual(out.frame.loc[0, "x1"], mean)
        self.assertTrue(np.isnan(self.ds.frame.loc[0, "x1"]))  # исходный не изменён

    def test_drop_too_many(self):
        df = make_frame(30)
        df.loc[:15, "x2"] = np.nan                     # останется 14 строк
        with self.assertRaises(ValidationError):
            MissingValueHandler.apply(Dataset(df), ["x2", "y"], "drop")

    def test_no_missing_returns_same(self):
        ds = Dataset(make_frame(30))
        self.assertIs(MissingValueHandler.apply(ds, ["x1", "y"], "drop"), ds)

    def test_keeps_synthetic_coefficients(self):
        syn = SyntheticGenerator(50, 2, 1.0, 1).generate()
        syn.frame.loc[0, "x1"] = np.nan
        out = MissingValueHandler.apply(syn, ["x1", "y"], "mean")
        self.assertIsInstance(out, SyntheticDataset)
        self.assertEqual(out.true_coefficients, syn.true_coefficients)


# ---------------------------------------------------------------- SyntheticGenerator
class TestSyntheticGenerator(TempDirMixin, unittest.TestCase):
    def test_shape_and_defaults(self):
        g = SyntheticGenerator()
        self.assertEqual((g.n_samples, g.n_features, g.noise, g.seed), (500, 3, 1.0, 42))
        ds = g.generate()
        self.assertEqual(list(ds.frame.columns), ["x1", "x2", "x3", "y"])
        self.assertEqual(ds.row_count(), 500)

    def test_reproducible_by_seed(self):
        a = SyntheticGenerator(100, 2, 1.0, 7).generate()
        b = SyntheticGenerator(100, 2, 1.0, 7).generate()
        c = SyntheticGenerator(100, 2, 1.0, 8).generate()
        pd.testing.assert_frame_equal(a.frame, b.frame)
        self.assertFalse(a.frame.equals(c.frame))

    def test_zero_noise_exact_dependency(self):
        ds = SyntheticGenerator(200, 4, 0.0, 3).generate()
        X = ds.frame[["x1", "x2", "x3", "x4"]].to_numpy()
        y_true = ds.true_intercept + X @ np.array(ds.true_coefficients)
        np.testing.assert_allclose(ds.frame["y"], y_true, atol=1e-3)

    def test_save_csv_roundtrip(self):
        ds = SyntheticGenerator(60, 2, 1.0, 1).generate()
        SyntheticGenerator.save_csv(ds, self.path("g.csv"))
        back = CsvReader().read(self.path("g.csv"))
        pd.testing.assert_frame_equal(back.frame, ds.frame)


# ---------------------------------------------------------------- DataPreparer
class TestDataPreparer(unittest.TestCase):
    def test_split_sizes_and_reproducibility(self):
        X, y = np.arange(200).reshape(100, 2), np.arange(100)
        a = DataPreparer.split(X, y, 0.2, 42)
        b = DataPreparer.split(X, y, 0.2, 42)
        self.assertEqual((len(a[0]), len(a[1])), (80, 20))
        for u, v in zip(a, b):
            np.testing.assert_array_equal(u, v)
        # обучающая и тестовая выборки не пересекаются и покрывают всё
        self.assertEqual(sorted(np.concatenate([a[2], a[3]])), list(range(100)))

    def test_split_rounds_test_size_up(self):
        X, y = np.zeros((21, 1)), np.zeros(21)
        self.assertEqual(len(DataPreparer.split(X, y, 0.1, 0)[1]), 3)  # ⌈2.1⌉

    def test_scaler_uses_train_statistics(self):
        dp = DataPreparer()
        X_train = np.array([[1.0, 10.0], [3.0, 10.0]])
        dp.fit_scaler(X_train)
        np.testing.assert_allclose(dp.mean_, [2.0, 10.0])
        np.testing.assert_allclose(dp.scale_, [1.0, 1.0])   # σ=0 → 1
        np.testing.assert_allclose(dp.scale(np.array([[5.0, 12.0]])), [[3.0, 2.0]])


# ---------------------------------------------------------------- Solvers и модели
class TestSolvers(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = rng.normal(size=(300, 3))
        self.b = np.array([1.5, -2.0, 0.5])
        self.y = 4.0 + self.X @ self.b

    def test_ols_exact_on_noiseless_data(self):
        e = LeastSquaresEstimator(None).fit(self.X, self.y)
        np.testing.assert_allclose(e.coef_, self.b, atol=1e-10)
        self.assertAlmostEqual(e.intercept_, 4.0, places=10)

    def test_ridge_matches_closed_form(self):
        a = 5.0
        e = LeastSquaresEstimator("l2", a).fit(self.X, self.y)
        Xc, yc = self.X - self.X.mean(0), self.y - self.y.mean()
        w = np.linalg.solve(Xc.T @ Xc + a * np.eye(3), Xc.T @ yc)
        np.testing.assert_allclose(e.coef_, w, atol=1e-12)

    def test_ridge_shrinks_coefficients(self):
        small = LeastSquaresEstimator("l2", 0.01).fit(self.X, self.y).coef_
        big = LeastSquaresEstimator("l2", 1000).fit(self.X, self.y).coef_
        self.assertLess(np.abs(big).sum(), np.abs(small).sum())

    def test_lasso_zeroes_useless_feature(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(500, 3))
        y = 3 * X[:, 0] + rng.normal(0, 0.1, 500)          # x2, x3 не нужны
        e = LeastSquaresEstimator("l1", 0.1).fit(X, y)
        self.assertNotEqual(e.coef_[0], 0.0)
        self.assertEqual(e.coef_[1], 0.0)
        self.assertEqual(e.coef_[2], 0.0)

    def test_lasso_large_alpha_gives_mean(self):
        e = LeastSquaresEstimator("l1", 1e6).fit(self.X, self.y)
        np.testing.assert_array_equal(e.coef_, 0.0)
        self.assertAlmostEqual(e.intercept_, self.y.mean())

    def test_lasso_optimality_conditions(self):
        """Условия оптимальности Lasso (KKT): для bⱼ≠0 |xⱼᵀr|/n = α,
        для bⱼ=0 |xⱼᵀr|/n ≤ α."""
        rng = np.random.default_rng(2)
        X = rng.normal(size=(400, 5))
        y = X @ np.array([2, 0, -1, 0, 0.3]) + rng.normal(0, 1, 400)
        a = 0.2
        e = LeastSquaresEstimator("l1", a).fit(X, y)
        r = y - e.predict(X)
        g = np.abs((X - X.mean(0)).T @ r) / len(y)
        for j, c in enumerate(e.coef_):
            if c != 0:
                self.assertAlmostEqual(g[j], a, places=4)
            else:
                self.assertLessEqual(g[j], a + 1e-6)

    def test_models_and_factory(self):
        self.assertIsInstance(create_model("linear", 1), LinearModel)
        self.assertIsInstance(create_model("ridge", 2), RidgeModel)
        self.assertIsInstance(create_model("lasso", 3), LassoModel)
        with self.assertRaises(ValueError):
            create_model("xxx", 1)
        m = LinearModel()
        self.assertFalse(m.fitted)
        m.fit(self.X, self.y)
        self.assertTrue(m.fitted)
        self.assertEqual(len(m.coefficients()), 3)


# ---------------------------------------------------------------- MetricsCalculator
class TestMetrics(unittest.TestCase):
    def test_known_values(self):
        y, yh = np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.0, 4.0, 2.0])
        mc = MetricsCalculator()
        self.assertAlmostEqual(mc.mse(y, yh), 1.25)          # (0+0+1+4)/4
        self.assertAlmostEqual(mc.mae(y, yh), 0.75)          # (0+0+1+2)/4
        self.assertAlmostEqual(mc.r2(y, yh), 1 - 5 / 5)      # SSres=5, SStot=5

    def test_perfect_and_constant(self):
        mc = MetricsCalculator()
        self.assertEqual(mc.r2([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertTrue(np.isnan(mc.r2([2, 2, 2], [1, 2, 3])))
        m = mc.compute([1, 2, 3], [1, 2, 3])
        self.assertEqual((m.mse, m.mae, m.r2), (0.0, 0.0, 1.0))


# ---------------------------------------------------------------- ExperimentService
class TestExperimentService(unittest.TestCase):
    def setUp(self):
        self.ds = SyntheticGenerator(400, 3, 0.0, 5).generate()
        self.feats = ["x1", "x2", "x3"]

    def test_train_recovers_true_coefficients(self):
        for std in (True, False):
            with self.subTest(standardize=std):
                r = ExperimentService().train(self.ds, ModelParams(
                    target="y", features=self.feats, standardize=std))
                np.testing.assert_allclose(r.coefficients,
                                           self.ds.true_coefficients, atol=1e-3)
                self.assertAlmostEqual(r.intercept, self.ds.true_intercept, places=3)
                self.assertGreater(r.metrics.r2, 0.999999)
                self.assertEqual(len(r.y_test), 80)

    def test_predict_one_matches_equation(self):
        ds = SyntheticGenerator(400, 3, 2.0, 5).generate()
        for algo in ("linear", "ridge", "lasso"):
            with self.subTest(algo=algo):
                svc = ExperimentService()
                r = svc.train(ds, ModelParams(algorithm=algo, target="y",
                                              features=self.feats))
                vals = {"x1": 0.3, "x2": -1.2, "x3": 2.0}
                manual = r.intercept + sum(b * vals[f] for b, f in
                                           zip(r.coefficients, self.feats))
                self.assertAlmostEqual(svc.predict_one(vals), manual, places=8)

    def test_predict_before_training(self):
        with self.assertRaises(ValidationError):
            ExperimentService().predict_one({"x1": 1.0})

    def test_compare_three_algorithms_same_split(self):
        rs = ExperimentService().compare(self.ds, ModelParams(
            target="y", features=self.feats))
        self.assertEqual([r.params.algorithm for r in rs], ["linear", "ridge", "lasso"])
        for r in rs[1:]:
            np.testing.assert_array_equal(r.y_test, rs[0].y_test)

    def test_missing_values_handled_in_train(self):
        ds = SyntheticGenerator(100, 2, 1.0, 1).generate()
        ds.frame.loc[[3, 7], "x1"] = np.nan
        r1 = ExperimentService().train(ds, ModelParams(
            target="y", features=["x1", "x2"], missing_method="drop"))
        r2 = ExperimentService().train(ds, ModelParams(
            target="y", features=["x1", "x2"], missing_method="mean"))
        self.assertEqual(len(r1.y_test), 20)   # ⌈0.2·98⌉
        self.assertEqual(len(r2.y_test), 20)   # ⌈0.2·100⌉


if __name__ == "__main__":
    unittest.main()
