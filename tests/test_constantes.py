"""Las constantes usan solo etiquetas de entrenamiento e índices de clase."""
import unittest

import numpy as np

from experimentos.constantes import predict_fold
from src.data.preprocessing import ExperimentData


class TestConstantes(unittest.TestCase):
    def test_conteos_empates_y_sin_fuga(self):
        data = ExperimentData('GDS_R2', np.zeros((8, 15)), np.array([0, 1, 2, 2, 2, 0, 1, 1]), [1, 2, 3])
        train, test = np.arange(6), np.array([6, 7])
        before = predict_fold(data, train, test, 1)
        np.testing.assert_allclose(np.stack(before.probs), [[2/6, 1/6, 3/6]] * 2)
        self.assertEqual(before.moda.tolist(), [2, 2])
        self.assertEqual(before.mediana.tolist(), [1, 1])
        data.y[test] = 0
        after = predict_fold(data, train, test, 1)
        np.testing.assert_array_equal(np.stack(before.probs), np.stack(after.probs))
        np.testing.assert_array_equal(before.moda, after.moda)
        np.testing.assert_array_equal(before.mediana, after.mediana)
