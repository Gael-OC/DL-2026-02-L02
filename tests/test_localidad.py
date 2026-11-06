"""
Pruebas para los modelos de localidad y sus órdenes.
"""

from __future__ import annotations

import unittest
import torch
import numpy as np

from experimentos.localidad import (
    build_locality_model,
    get_feature_permutation,
    apply_order_to_features,
)


class TestLocalidadPanel(unittest.TestCase):

    def test_model_shapes_and_parameters(self):
        model_k3 = build_locality_model("cnn_k3", num_features=15, num_classes=3)
        model_k1 = build_locality_model("cnn_k1", num_features=15, num_classes=3)
        model_mlp = build_locality_model("mlp_capacity", num_features=15, num_classes=3)

        params_k3 = sum(p.numel() for p in model_k3.parameters())
        params_mlp = sum(p.numel() for p in model_mlp.parameters())

        self.assertEqual(params_k3, 17219, f"CNN-k3 tiene {params_k3} parámetros (esperados 17219)")
        self.assertEqual(params_mlp, 17231, f"MLP capacidad tiene {params_mlp} parámetros (esperados 17231)")

        dummy_flat = torch.randn(8, 15)
        dummy_seq = torch.randn(8, 1, 15)

        for model in [model_k3, model_k1, model_mlp]:
            model.eval()
            with torch.no_grad():
                out_flat = model(dummy_flat)
                self.assertEqual(out_flat.shape, (8, 3))
                
                if isinstance(model, build_locality_model("mlp_capacity").__class__):
                    out_seq = model(dummy_seq)
                    self.assertEqual(out_seq.shape, (8, 3))

    def test_feature_orders(self):
        orig_indices = get_feature_permutation("original")
        blocks_indices = get_feature_permutation("blocks")
        global_indices = get_feature_permutation("global")

        self.assertEqual(len(orig_indices), 15)
        self.assertEqual(len(blocks_indices), 15)
        self.assertEqual(len(global_indices), 15)

        self.assertEqual(len(set(blocks_indices)), 15)
        self.assertEqual(len(set(global_indices)), 15)

        X_dummy = np.arange(30).reshape(2, 15)
        X_reordered = apply_order_to_features(X_dummy, "blocks")
        self.assertEqual(X_reordered.shape, (2, 15))
        np.testing.assert_array_equal(X_reordered[0], np.array(blocks_indices))


if __name__ == "__main__":
    unittest.main()
