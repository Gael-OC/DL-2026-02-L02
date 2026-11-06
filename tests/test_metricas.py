"""Casos del alfabeto completo y de las métricas probabilísticas."""

import unittest

import numpy as np
from sklearn.metrics import cohen_kappa_score

from src.evaluation.metrics import MetricsCalculator as M
from src.reporting.reports import MarkdownTables, ResultRows
from src.training.nested_cv import ExperimentResult, OuterFoldResult
from src.training.search import CandidateScore
from src.config import ModelConfig, TrainingConfig


class TestMetricas(unittest.TestCase):
    def test_clases_ausentes_y_distancias_qwk(self):
        y = np.array([0, 0, 2, 2])
        pred = np.array([0, 2, 0, 2])
        metrics = M.compute(y, pred, num_classes=4)
        self.assertAlmostEqual(metrics["qwk"], cohen_kappa_score(y, pred, weights="quadratic", labels=[0, 1, 2, 3]))
        self.assertAlmostEqual(metrics["f1_macro"], 0.25)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.5)
        self.assertEqual(M.confusion_matrix(y, pred, 4).shape, (4, 4))
        self.assertIn("40", M.classification_report(y, pred, [10, 20, 30, 40]))
        # El alfabeto comprimido [0, 2, 3] cambia las distancias y la QWK.
        y, pred = [0, 0, 2, 2, 3, 3], [0, 2, 3, 0, 3, 2]
        qwk = M.quadratic_weighted_kappa(y, pred, 4)
        self.assertAlmostEqual(qwk, cohen_kappa_score(y, pred, weights="quadratic", labels=[0, 1, 2, 3]))
        self.assertNotAlmostEqual(qwk, cohen_kappa_score(y, pred, weights="quadratic", labels=[0, 2, 3]))
        with self.assertRaises(ValueError):
            M.compute([0, 3], [0, 0], 3)

    def test_qwk_no_estimable_y_resumen(self):
        self.assertTrue(np.isnan(M.quadratic_weighted_kappa([1, 1], [1, 1], 3)))
        self.assertEqual(M.quadratic_weighted_kappa([1, 1], [0, 0], 3), 0)
        score = CandidateScore({}, ModelConfig(), TrainingConfig())
        folds = [OuterFoldResult(i, score, False, M.compute(y, p, 3), np.array(y), np.array(p), 0)
                 for i, (y, p) in enumerate([([1, 1], [1, 1]), ([0, 1], [0, 1])], 1)]
        result = ExperimentResult("a", "b", [1, 2, 3], 4, 15, 2, 2, False, "cpu", 1, folds)
        for key in folds[0].metrics:
            valid = np.array([f.metrics[key] for f in folds])
            valid = valid[np.isfinite(valid)]
            result.summary[f"valid_{key}"] = len(valid)
            result.summary[f"mean_{key}"] = float(valid.mean()) if len(valid) else float("nan")
            result.summary[f"std_{key}"] = float(valid.std()) if len(valid) else float("nan")
        row = ResultRows.summary_row(result)
        self.assertEqual(row["qwk_valid"], 1)
        self.assertEqual(row["qwk"], 1)
        self.assertIn("1/2", MarkdownTables.results([row]))
        self.assertIn("desacuerdo esperado", ResultRows.selection_rows(result)[0]["qwk_no_estimable"])

    def test_metricas_probabilisticas_y_r3(self):
        y = np.array([0, 1, 1, 0])
        probabilities = np.array([[0.8, 0.2], [0.25, 0.75], [0.6, 0.4], [0.1, 0.9]])
        pred = probabilities.argmax(axis=1)
        metrics = M.compute(y, pred, 2)
        self.assertAlmostEqual(metrics["mae_ordinal"], 1 - metrics["accuracy"])
        self.assertEqual(metrics["accuracy_pm1"], 1)
        self.assertEqual(metrics["errores_graves"], 0)
        self.assertAlmostEqual(M.rps(y, probabilities), np.mean((probabilities[:, 0] - (y == 0)) ** 2))
        self.assertAlmostEqual(M.nll(y, probabilities), -np.log([0.8, 0.75, 0.4, 0.1]).mean())
        perfect = np.eye(3, dtype=np.float64)[[0, 2]]
        self.assertEqual(M.rps([0, 2], perfect), 0)
        self.assertEqual(M.nll([0, 2], perfect), 0)
        self.assertAlmostEqual(M.rps([2], np.array([[1., 0., 0.]])), 1)
        zeros = np.array([[1, 0]], dtype=np.float32)
        self.assertAlmostEqual(M.nll([1], zeros), -float(np.log(np.nextafter(np.float32(0), np.float32(1)))), places=5)
        for invalid in (np.array([[0.5, -0.1, 0.6]]), np.array([[0.2, 0.2]]), np.array([[np.nan, 1.0]])):
            with self.assertRaises(ValueError):
                M.rps([0], invalid)


if __name__ == "__main__":
    unittest.main()
