import unittest

import numpy as np
import torch

from src.config import ModelConfig, TrainingConfig
from src.evaluation.metrics import MetricsCalculator
from src.tasks.ordinal_task import CoralTask
from src.tasks.softmax_task import SoftmaxTask
from src.training.trainer import Trainer


class TestProbabilidades(unittest.TestCase):
    def test_conversiones_y_decisiones(self):
        for k in (2, 3, 7):
            logits = torch.linspace(3, -3, k - 1).repeat(2, 1)
            task = CoralTask(k)
            probs = task.probabilities(logits)
            self.assertEqual(tuple(probs.shape), (2, k))
            self.assertTrue(torch.isfinite(probs).all() and (probs >= 0).all())
            torch.testing.assert_close(probs.sum(1), torch.ones(2))
            torch.testing.assert_close(probs[:, 1:].sum(1), torch.sigmoid(logits[:, 0]))
            torch.testing.assert_close(task.predict(logits), (torch.sigmoid(logits) > .5).sum(1))
            soft = SoftmaxTask(k)
            soft_logits = torch.randn(2, k)
            torch.testing.assert_close(soft.probabilities(soft_logits), torch.softmax(soft_logits, 1))
            torch.testing.assert_close(soft.predict(soft_logits), soft_logits.argmax(1))
        # La decision CORAL por umbrales no necesariamente es la moda.
        task = CoralTask(3)
        logits = torch.logit(torch.tensor([[0.6, 0.3]]))
        self.assertEqual(task.predict(logits).item(), 1)
        self.assertEqual(task.probabilities(logits).argmax(1).item(), 0)

    def test_rechaza_invalidas(self):
        for logits in (torch.tensor([[0., 1.]]), torch.tensor([[float('nan'), 0.]]), torch.ones(2, 3)):
            with self.assertRaises(ValueError):
                CoralTask(3).probabilities(logits)
        with self.assertRaises(ValueError):
            SoftmaxTask(3).probabilities(torch.tensor([[0., float('inf'), 1.]]))

    def test_entrenador_y_metricas(self):
        X = np.random.default_rng(1).normal(size=(9, 15)).astype(np.float32)
        y = np.arange(9) % 3
        for algorithm in ("conv1d_softmax", "conv1d_coral"):
            trainer = Trainer.build(algorithm, ModelConfig(), TrainingConfig(epochs=1, batch_size=4), 15, 3, torch.device('cpu'), 42)
            trainer.fit(X, y, 42)
            logits, probs, pred = trainer.predict_logits(X), trainer.predict_proba(X), trainer.predict(X)
            self.assertEqual(logits.shape, (9, 3 if algorithm.endswith('softmax') else 2))
            self.assertEqual(probs.shape, (9, 3))
            np.testing.assert_allclose(probs.sum(axis=1), 1, atol=1e-6)
            np.testing.assert_array_equal(pred, trainer.task.predict(torch.from_numpy(logits)).numpy())
            self.assertTrue(np.isfinite(MetricsCalculator.nll(y, probs)))
            self.assertTrue(np.isfinite(MetricsCalculator.rps(y, probs)))
            self.assertEqual(trainer.predict_proba(X[:0]).shape, (0, 3))


if __name__ == '__main__':
    unittest.main()
