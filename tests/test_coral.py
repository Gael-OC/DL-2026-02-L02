import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.config import ModelConfig, TrainingConfig
from src.models.coral import CoralLayer, Conv1DCoralNet
from src.tasks.ordinal_task import CoralTask, coral_loss, effective_number_weights, labels_to_levels, logits_to_ordinal_predictions
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

    def test_red_acepta_ambas_formas(self):
        for k in (2, 3, 7):
            with self.subTest(k=k):
                model = Conv1DCoralNet(num_features=15, num_classes=k).eval()
                inputs = torch.randn(4, 15)
                logits = model(inputs)
                self.assertEqual(logits.shape, (4, k - 1))
                torch.testing.assert_close(logits, model(inputs.unsqueeze(1)))
                self.assertTrue(torch.all(logits[:, :-1] >= logits[:, 1:]))
                logits.square().mean().backward()
                for parameter in model.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_requiere_dos_clases(self):
        with self.assertRaises(ValueError):
            CoralLayer(5, 1)


class TestPerdidaCoral(unittest.TestCase):
    def test_reduccion_y_equivalencia_binaria(self):
        for k in (2, 3, 7):
            with self.subTest(k=k):
                logits = torch.randn(k, k - 1, requires_grad=True)
                labels = torch.arange(k)
                levels = labels_to_levels(labels, k)
                reference = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, levels, reduction="none"
                ).sum(dim=1).mean()
                loss = coral_loss(logits, labels, k)
                self.assertEqual(loss.ndim, 0)
                torch.testing.assert_close(loss, reference)
                if k == 2:
                    torch.testing.assert_close(
                        loss, torch.nn.functional.binary_cross_entropy_with_logits(logits, levels)
                    )
                torch.testing.assert_close(loss, coral_loss(logits, labels, k, torch.ones(k)))

    def test_logits_extremos_y_pesos_por_clase(self):
        logits = torch.tensor([[1000.0, -1000.0], [-1000.0, 1000.0]], requires_grad=True)
        labels = torch.tensor([0, 2])
        weights = torch.tensor([2.0, 1.0, 3.0])
        individual = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels_to_levels(labels, 3), reduction="none"
        ).sum(dim=1)
        loss = coral_loss(logits, labels, 3, weights)
        torch.testing.assert_close(loss, (individual * weights[labels]).mean())
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())


class TestPesosCoral(unittest.TestCase):
    def test_conteos_y_clases_ausentes(self):
        torch.testing.assert_close(
            effective_number_weights(np.array([0, 0, 1, 1, 2, 2]), 3), torch.ones(3)
        )
        weights = effective_number_weights(np.array([0, 0, 0, 1]), 3)
        raw = np.array([(1 - 0.99) / (1 - 0.99 ** 3), 1.0, 0.0])
        torch.testing.assert_close(weights, torch.tensor(raw / raw.mean(), dtype=torch.float32))
        self.assertTrue(torch.isfinite(weights).all())
        self.assertEqual(weights.shape, (3,))
        self.assertAlmostEqual(weights.mean().item(), 1.0)

    def test_conteos_solo_de_entrenamiento(self):
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

    def test_entrada_invalida(self):
        for labels, beta in ((np.array([], dtype=int), 0.99), (np.array([3]), 0.99),
                             (np.array([0]), 1.0)):
            with self.subTest(labels=labels, beta=beta), self.assertRaises(ValueError):
                effective_number_weights(labels, 3, beta)


class TestIntegracionCoral(unittest.TestCase):
    def test_entrenamiento_y_prediccion_con_y_sin_pesos(self):
        rng = np.random.default_rng(42)
        X = rng.integers(0, 2, size=(12, 15)).astype(np.float32)
        y = np.tile(np.arange(3), 4)
        for weighted in (False, True):
            with self.subTest(weighted=weighted):
                trainer = Trainer.build(
                    "conv1d_coral", ModelConfig(),
                    TrainingConfig(epochs=1, batch_size=6, use_class_weights=weighted),
                    num_features=15, num_classes=3, device=torch.device("cpu"), seed=42,
                )
                result = trainer.fit_and_evaluate(X[:9], y[:9], X[9:], y[9:], seed=42)
                self.assertTrue(np.isfinite(result.final_train_loss))
                self.assertTrue(all(np.isfinite(value) for value in result.metrics.values()))
                self.assertEqual(result.y_pred.shape, (3,))
                self.assertTrue(np.all((0 <= result.y_pred) & (result.y_pred < 3)))


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

    def test_predicciones_umbral_y_forma(self):
        for k in (2, 3, 7):
            with self.subTest(k=k):
                logits = torch.stack([
                    torch.full((k - 1,), -2.0),
                    torch.zeros(k - 1),
                    torch.full((k - 1,), 2.0),
                ])
                predictions = logits_to_ordinal_predictions(logits)
                self.assertEqual(predictions.shape, (3,))
                self.assertEqual(predictions.dtype, torch.int64)
                torch.testing.assert_close(predictions, torch.tensor([0, 0, k - 1]))

    def test_umbral_configurable(self):
        logits = torch.tensor([[0.0, 1.0, -1.0]])
        torch.testing.assert_close(
            logits_to_ordinal_predictions(logits, threshold=0.7), torch.tensor([1])
        )


if __name__ == "__main__":
    unittest.main()
