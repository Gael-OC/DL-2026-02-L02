"""
//comando de comprobación:
bash:
`` python -m unittest discover -s tests -p 'test_controles.py' -v . ``
anaconda prompt:
`` python -m unittest discover -s tests -p "test_controles.py" -v . ``
"""

import unittest
import numpy as np
from experimentos.controles import ConstantPredictor, LogisticControlPredictor
from src.config import DEFAULT_RANDOM_SEED

class TestControlesLab(unittest.TestCase):
    
    def setUp(self):
        """Inicializa datos y variables comunes para las pruebas de controles."""
        # Datos para predictores constantes
        self.y_train_const = np.array([0, 0, 1, 1, 2])
        self.classes_const = np.array([0, 1, 2])
        
        # Datos sintéticos simulando el entorno (N, 15) binarias para regresión logística
        np.random.seed(DEFAULT_RANDOM_SEED)
        self.X_train = np.random.randint(0, 2, size=(60, 15)).astype(np.float32)
        self.y_train = np.random.randint(0, 3, size=(60,))
        self.X_val = np.random.randint(0, 2, size=(20, 15)).astype(np.float32)
        self.y_val = np.random.randint(0, 3, size=(20,))
        self.classes_log = np.array([0, 1, 2])

    def test_constant_predictor_mode_and_median(self):
        # Moda con desempate hacia el índice menor utilizando variables de setUp
        model_mode = ConstantPredictor(strategy='mode', classes=self.classes_const)
        model_mode.fit(self.y_train_const)
        self.assertEqual(model_mode.pred_class, 0)
        
        # Probabilidades válidas y alineadas
        probs = model_mode.predict_proba(np.zeros((5, 15)))
        self.assertEqual(probs.shape, (5, 3))
        np.testing.assert_allclose(np.sum(probs, axis=1), 1.0)

    def test_logistic_control_variants(self):
        # 1. Control de Regresión Logística sobre la suma de bits (modo 'sum')
        clf_sum = LogisticControlPredictor(mode='sum')
        clf_sum.fit_grid_search(self.X_train, self.y_train, self.X_val, self.y_val, self.classes_log)
        
        self.assertIn(clf_sum.best_C, [0.01, 0.1, 1.0, 10.0])
        preds_sum = clf_sum.predict(self.X_val)
        probs_sum = clf_sum.predict_proba(self.X_val)
        self.assertEqual(preds_sum.shape, (20,))
        self.assertEqual(probs_sum.shape, (20, 3))

        # 2. Control de Regresión Logística sobre las 15 entradas (modo 'features')
        clf_feat = LogisticControlPredictor(mode='features')
        clf_feat.fit_grid_search(self.X_train, self.y_train, self.X_val, self.y_val, self.classes_log)
        
        self.assertIn(clf_feat.best_C, [0.01, 0.1, 1.0, 10.0])
        preds_feat = clf_feat.predict(self.X_val)
        probs_feat = clf_feat.predict_proba(self.X_val)
        self.assertEqual(preds_feat.shape, (20,))
        self.assertEqual(probs_feat.shape, (20, 3))

if __name__ == '__main__':
    unittest.main()