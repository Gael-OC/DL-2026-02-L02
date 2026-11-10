import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.model_selection import KFold

from experimentos.ejecutar_controles import evaluate


class TestEvaluacionControles(unittest.TestCase):
    def test_seleccion_tres_folds_refit_y_sin_fuga(self):
        n = 30
        X = np.arange(n, dtype=np.float32).reshape(-1, 1)
        y = np.tile([0, 1, 2], 10)
        splits = {}
        for outer, (train, test) in enumerate(KFold(5).split(X), 1):
            splits[f'outer_{outer}_train'] = train
            splits[f'outer_{outer}_test'] = test
            for i, (tr, val) in enumerate(KFold(3).split(train), 1):
                splits[f'outer_{outer}_inner_{i}_train'] = train[tr]
                splits[f'outer_{outer}_inner_{i}_val'] = train[val]
        calls = []
        def fake_fit(name, config, seed, X_train, y_train, X_eval, k):
            calls.append((config['C'], seed, set(X_train[:, 0]), set(X_eval[:, 0])))
            true = y[X_eval[:, 0].astype(int)]
            # La primera candidata gana solo en el primer fold interno;
            # la segunda gana los otros dos y debe ser seleccionada.
            pred = true.copy() if (config['C'] == 1) == (seed % 100 == 0) else (true + 1) % k
            probs = np.eye(k)[pred] * 0.98 + 0.02 / k
            return pred, probs
        with tempfile.TemporaryDirectory() as folder, patch('experimentos.ejecutar_controles.CANDIDATES', {'logistica_15': [{'C': 1}, {'C': 2}]}), patch('experimentos.ejecutar_controles.fit_predict', side_effect=fake_fit):
            evaluate(X, y, splits, np.array([1, 2, 3]), Path(folder))
            import json
            fits = json.loads((Path(folder) / 'fits.json').read_text())
            choices = json.loads((Path(folder) / 'candidatas.json').read_text())
            self.assertEqual(len(fits), 5 * (2 * 3 + 1))
            self.assertTrue(all(r['candidata'] == 2 for r in choices if r['seleccionada']))
            for row in fits:
                outer = row['fold']
                expected = splits[f"outer_{outer}_train"] if row['fase'] == 'refit' else splits[f"outer_{outer}_inner_{row['inner_fold']}_train"]
                self.assertEqual(row['train_indices'], expected.tolist())
                self.assertFalse(set(row['train_indices']) & set(row['eval_indices']))
                if row['fase'] == 'inner':
                    self.assertFalse(set(row['train_indices'] + row['eval_indices']) & set(splits[f'outer_{outer}_test']))
            self.assertEqual(len(calls), 35)


if __name__ == '__main__':
    unittest.main()
