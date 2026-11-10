"""Contrato del predictor por perfil y evaluación con splits persistidos."""

import unittest

import numpy as np

from experimentos.perfiles import predict_fold
from src.data.preprocessing import ExperimentData


class TestProfilePredictor(unittest.TestCase):


    def test_test_labels_do_not_affect_predictions(self):
        X = np.array([np.zeros(15), np.zeros(15), np.ones(15), np.ones(15)])
        y = np.array([0, 1, 2, 1])
        train, test = np.array([0, 2]), np.array([1, 3])
        data = ExperimentData('GDS_R2', X, y, [1, 2, 3])
        before = predict_fold(data, train, test, 1)
        np.testing.assert_allclose(np.stack(before.probs), [[0.5, 0.25, 0.25], [0.25, 0.25, 0.5]])
        data.y = np.array([0, 2, 2, 0])  # Solo cambian las etiquetas externas.
        after = predict_fold(data, train, test, 1)
        for column in ('moda', 'mediana', 'visto'):
            np.testing.assert_array_equal(before[column], after[column])
        np.testing.assert_array_equal(np.stack(before.probs), np.stack(after.probs))
        self.assertFalse(np.array_equal(before.verdad, after.verdad))


if __name__ == '__main__':
    unittest.main()
