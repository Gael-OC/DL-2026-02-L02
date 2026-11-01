"""
//comando de comprobación:
`` python -m unittest discover -s tests -p "test_controles.py" -v . ``
"""

import unittest
import numpy as np

from experimentos.controles import (
    ConstantPredictor,
    LogisticControlPredictor,
    MLPControlPredictor,
    HistGradientBoostingControlPredictor,
)

from src.config import DEFAULT_RANDOM_SEED

class TestControlesLab(unittest.TestCase):

    def setUp(self):
        self.y_train_const = np.array([0, 0, 1, 1, 2])
        self.classes_const = np.array([0, 1, 2])

        np.random.seed(DEFAULT_RANDOM_SEED)
        self.X_train = np.random.randint(0, 2, size=(60, 15)).astype(np.float32)
        self.y_train = np.random.randint(0, 3, size=(60,))
        self.X_val = np.random.randint(0, 2, size=(20, 15)).astype(np.float32)
        self.y_val = np.random.randint(0, 3, size=(20,))
        self.classes = np.array([0, 1, 2])

    def test_constant_predictor(self):
        model = ConstantPredictor(strategy='mode', classes=self.classes_const)
        model.fit(self.y_train_const)
        self.assertEqual(model.pred_class, 0)
        probs = model.predict_proba(np.zeros((5, 15)))
        self.assertEqual(probs.shape, (5, 3))

    def test_logistic_controls(self):
        clf = LogisticControlPredictor(mode='features')
        clf.fit_grid_search(self.X_train, self.y_train, self.X_val, self.y_val, self.classes)
        self.assertIn(clf.best_C, [0.01, 0.1, 1.0, 10.0])
        self.assertEqual(clf.predict(self.X_val).shape, (20,))

    def test_mlp_control(self):
            mlp = MLPControlPredictor(epochs=2, batch_size=16) # Épocas reducidas para test rápido
            mlp.fit_grid_search(self.X_train, self.y_train, self.X_val, self.y_val, self.classes)
            self.assertIsNotNone(mlp.best_config)
            self.assertEqual(mlp.predict(self.X_val).shape, (20,))
            self.assertEqual(mlp.predict_proba(self.X_val).shape, (20, 3))

    def test_hist_gb_control(self):
        hgb = HistGradientBoostingControlPredictor()
        hgb.fit_grid_search(self.X_train, self.y_train, self.X_val, self.y_val, self.classes)
        self.assertIn(hgb.best_leaves, [3, 7, 15, 31])
        self.assertEqual(hgb.predict(self.X_val).shape, (20,))
        self.assertEqual(hgb.predict_proba(self.X_val).shape, (20, 3))


if __name__ == '__main__':
    unittest.main()
