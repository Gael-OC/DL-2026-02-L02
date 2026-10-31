import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.config import ModelConfig, TrainingConfig
from src.data.preprocessing import ExperimentData, StratifiedSplitter
from src.training.nested_cv import NestedCrossValidator
from src.training.search import GridSearch
from src.training.trainer import FitResult


class DummyTrainer:
    def fit_and_evaluate(self, X_train, y_train, X_eval, y_eval, seed):
        from src.evaluation.metrics import MetricsCalculator
        y_pred = np.zeros(len(y_eval), dtype=int)
        return FitResult(MetricsCalculator.compute(y_eval, y_pred, 3), y_eval, y_pred)


class TestParticiones(unittest.TestCase):
    def test_indices_globales_y_compatibilidad(self):
        y = np.repeat(np.arange(3), 10)
        data = ExperimentData("GDS_R2", np.zeros((30, 15)), y, [1, 2, 3])
        with tempfile.TemporaryDirectory() as folder, patch("src.training.search.Trainer.build", return_value=DummyTrainer()):
            path = Path(folder) / "splits.npz"
            validator = NestedCrossValidator("conv1d_softmax", ModelConfig(), TrainingConfig(epochs=1), 5, 3, None, 42, torch.device("cpu"), verbose=False, split_path=path)
            validator.run(data)
            validator.run(data)
            with np.load(path) as saved:
                tests = []
                for i, (train, test) in enumerate(StratifiedSplitter(5, 42).split(y), 1):
                    np.testing.assert_array_equal(saved[f"outer_{i}_train"], train)
                    np.testing.assert_array_equal(saved[f"outer_{i}_test"], test)
                    tests.extend(test)
                    for j, (a, b) in enumerate(StratifiedSplitter(3, 42 + i * 100).split(y[train]), 1):
                        np.testing.assert_array_equal(saved[f"outer_{i}_inner_{j}_train"], train[a])
                        np.testing.assert_array_equal(saved[f"outer_{i}_inner_{j}_val"], train[b])
                        self.assertFalse(set(train[a]) & set(train[b]))
                        self.assertFalse(set(test) & set(train[b]))
                self.assertEqual(sorted(tests), list(range(len(y))))
            validator.split_seed = 314
            with self.assertRaises(ValueError):
                validator.run(data)

    def test_semillas_independientes_del_orden(self):
        X, y = np.zeros((30, 15)), np.repeat(np.arange(3), 10)
        seen = []
        def build(*args, **kwargs):
            seen.append((args[1].hidden_dim, kwargs["seed"]))
            return DummyTrainer()
        candidates = [{"hidden_dim": 16}, {"hidden_dim": 32}]
        with patch("src.training.search.Trainer.build", side_effect=build):
            for grid in (candidates, candidates[::-1]):
                search = GridSearch("conv1d_softmax", grid, ModelConfig(), TrainingConfig(), 3, torch.device("cpu"))
                search.run(X, y, 3, seed=142, init_seed=900)
        self.assertEqual(sorted(seen[:6]), sorted(seen[6:]))
        self.assertEqual({seed for _, seed in seen}, {900, 901, 902})

    def test_gds_sin_busqueda_interna(self):
        data = ExperimentData("GDS", np.zeros((30, 15)), np.repeat(np.arange(3), 10), [1, 2, 3])
        with tempfile.TemporaryDirectory() as folder, patch("src.training.search.Trainer.build", return_value=DummyTrainer()):
            path = Path(folder) / "splits.npz"
            result = NestedCrossValidator("conv1d_softmax", ModelConfig(), TrainingConfig(), 2, 2, [{"hidden_dim": 16}], 42, torch.device("cpu"), verbose=False, split_path=path).run(data)
            self.assertTrue(all(f.inner_skipped for f in result.folds))
            self.assertTrue(all(f.selected.model_config.hidden_dim == ModelConfig().hidden_dim for f in result.folds))
            with np.load(path) as saved:
                self.assertTrue(saved["outer_1_inner_skipped"])
                self.assertNotIn("outer_1_inner_1_train", saved)


if __name__ == "__main__":
    unittest.main()
