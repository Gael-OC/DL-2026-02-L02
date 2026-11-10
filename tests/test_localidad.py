"""
Pruebas unitarias para los modelos de localidad y sus órdenes.
"""

from __future__ import annotations

import unittest
import numpy as np

from experimentos.localidad import get_feature_permutation, apply_order_to_features


class TestProtocoloLocalidad(unittest.TestCase):

    def test_permutaciones_y_misma_transformacion(self):
        expected = {
            'original': list(range(15)),
            'blocks': [3, 1, 0, 2, 4, 5, 6, 7, 9, 13, 14, 12, 8, 10, 11],
            'global': [4, 2, 13, 5, 9, 10, 7, 14, 3, 0, 11, 1, 8, 6, 12],
        }
        X = np.arange(120).reshape(8, 15)
        train, test = [0, 2, 4, 6], [1, 3, 5, 7]
        for order, indices in expected.items():
            self.assertEqual(get_feature_permutation(order), indices)
            ordered = apply_order_to_features(X, order)
            np.testing.assert_array_equal(ordered[train], X[train][:, indices])
            np.testing.assert_array_equal(ordered[test], X[test][:, indices])


if __name__ == "__main__":
    unittest.main()
