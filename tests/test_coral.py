import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.config import TrainingConfig
from src.models.coral import CoralLayer, Conv1DCoralNet
from src.tasks.ordinal_task import CoralTask, coral_loss, effective_number_weights, labels_to_levels
from src.training.trainer import Trainer


class TestCabezaCoral(unittest.TestCase):
    def test_cabeza_orden_y_gradientes_tras_optimizacion(self):
        for k in (2, 3, 7):
            with self.subTest(k=k):
                layer = CoralLayer(5, k)
                inputs = torch.randn(4, 5)
                optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)
                for step in range(2):
                    logits = layer(inputs)
                    self.assertEqual(logits.shape, (4, k - 1))
                    self.assertTrue(torch.all(logits[:, :-1] >= logits[:, 1:]))
                    if step == 0:
                        logits.square().mean().backward()
                        for parameter in layer.parameters():
                            self.assertIsNotNone(parameter.grad)
                            self.assertTrue(torch.isfinite(parameter.grad).all())
                        optimizer.step()


class TestPerdidaCoral(unittest.TestCase):

    def test_logits_extremos_y_pesos_por_clase(self):
        logits = torch.tensor([[1000.0, -1000.0], [-1000.0, 1000.0]], requires_grad=True)
        labels = torch.tensor([0, 2])
        weights = torch.tensor([2.0, 1.0, 3.0])
        individual = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels_to_levels(labels, 3), reduction="none"
        ).sum(dim=1)
        torch.testing.assert_close(coral_loss(logits, labels, 3), individual.mean())
        loss = coral_loss(logits, labels, 3, weights)
        torch.testing.assert_close(loss, (individual * weights[labels]).mean())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())


class TestPesosCoral(unittest.TestCase):

    def test_conteos_solo_de_entrenamiento(self):
        weights = effective_number_weights(np.array([0, 0, 0, 1]), 3)
        raw = np.array([(1 - 0.99) / (1 - 0.99 ** 3), 1.0, 0.0])
        torch.testing.assert_close(weights, torch.tensor(raw / raw.mean(), dtype=torch.float32))
        train_labels = np.array([0, 0, 1, 1])
        eval_labels = np.array([2, 2])
        trainer = Trainer(
            Conv1DCoralNet(15, 3), CoralTask(3),
            TrainingConfig(epochs=1, use_class_weights=True), torch.device("cpu")
        )
        with patch("src.tasks.ordinal_task.effective_number_weights", wraps=effective_number_weights) as calculate, \
             patch.object(trainer, "train_one_epoch", return_value=0.0), \
             patch.object(trainer, "evaluate", return_value=({}, np.zeros(2))):
            trainer.fit_and_evaluate(np.zeros((4, 15)), train_labels, np.zeros((2, 15)), eval_labels, 42)
        calculate.assert_called_once()
        np.testing.assert_array_equal(calculate.call_args.args[0], train_labels)


class TestNivelesEInferencia(unittest.TestCase):
    def test_niveles_para_todos_los_extremos_y_clases(self):
        for k in (2, 3, 7):
            with self.subTest(k=k):
                labels = torch.arange(k, dtype=torch.int64)
                levels = labels_to_levels(labels, k)
                expected = torch.tensor(
                    [[float(y > threshold) for threshold in range(k - 1)]
                     for y in range(k)]
                )
                self.assertEqual(levels.shape, (k, k - 1))
                self.assertEqual(levels.dtype, torch.float32)
                self.assertEqual(levels.device, labels.device)
                torch.testing.assert_close(levels, expected)


if __name__ == "__main__":
    unittest.main()
