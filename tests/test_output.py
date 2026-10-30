"""Модульные тесты подсистем визуализации и экспорта."""

import os
import tempfile
import unittest

import pandas as pd

from core.experiment import ExperimentService
from core.generator import SyntheticGenerator
from core.params import ModelParams
from output.plot_builder import PlotBuilder
from output.report_exporter import ReportExporter


class TestOutput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ds = SyntheticGenerator(200, 2, 1.0, 3).generate()
        cls.r = ExperimentService().train(ds, ModelParams(
            algorithm="ridge", target="y", features=["x1", "x2"], alpha=2.0))
        cls.tmp = tempfile.mkdtemp()

    def test_plot(self):
        fig = PlotBuilder.actual_vs_predicted(self.r)
        ax = fig.axes[0]
        self.assertEqual(len(ax.collections[0].get_offsets()), len(self.r.y_test))
        self.assertIn("R²", ax.get_title())
        self.assertEqual(len(ax.lines), 1)                 # линия ŷ = y

    def test_plot_reuses_figure(self):
        fig = PlotBuilder.actual_vs_predicted(self.r)
        PlotBuilder.actual_vs_predicted(self.r, fig)
        self.assertEqual(len(fig.axes), 1)                 # старый график стёрт

    def test_save_txt(self):
        path = os.path.join(self.tmp, "отчёт.txt")
        ReportExporter.save_txt(self.r, path, "data.csv", 200)
        with open(path, encoding="utf-8") as f:
            text = f.read()
        for s in ("Гребневая", "α: 2.0", "Свободный член", "MSE", "MAE", "R²",
                  "data.csv", "x1", "x2"):
            self.assertIn(s, text)

    def test_save_predictions_csv(self):
        path = os.path.join(self.tmp, "p.csv")
        ReportExporter.save_predictions_csv(self.r, path)
        df = pd.read_csv(path, sep=";")
        self.assertEqual(list(df.columns), ["№", "Фактическое", "Прогнозное", "Остаток"])
        self.assertEqual(len(df), len(self.r.y_test))
        pd.testing.assert_series_equal(df["Остаток"], df["Фактическое"] - df["Прогнозное"],
                                       check_names=False, atol=1e-9)

    def test_save_png(self):
        path = os.path.join(self.tmp, "g.png")
        ReportExporter.save_plot_png(PlotBuilder.actual_vs_predicted(self.r), path)
        with open(path, "rb") as f:
            self.assertEqual(f.read(8), b"\x89PNG\r\n\x1a\n")

    def test_save_to_bad_path_gives_message(self):
        from core.errors import ValidationError
        bad = os.path.join(self.tmp, "нет_папки", "x.txt")
        with self.assertRaises(ValidationError):
            ReportExporter.save_txt(self.r, bad)


if __name__ == "__main__":
    unittest.main()
