import unittest

import torch

from src.models.coral import CoralLayer, Conv1DCoralNet
from src.tasks.ordinal_task import coral_loss, labels_to_levels, logits_to_ordinal_predictions


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
