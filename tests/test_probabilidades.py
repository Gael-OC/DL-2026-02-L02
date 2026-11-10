import unittest

import torch

from src.tasks.ordinal_task import CoralTask
from src.tasks.softmax_task import SoftmaxTask


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


if __name__ == '__main__':
    unittest.main()
