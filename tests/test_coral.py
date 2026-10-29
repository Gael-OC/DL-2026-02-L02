import unittest

import torch

from src.tasks.ordinal_task import labels_to_levels, logits_to_ordinal_predictions


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
