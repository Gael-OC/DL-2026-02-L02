"""Pares completos, errores firmados, recalls y regeneración del análisis."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experimentos.analisis import load_rows, main, tables
from experimentos.decisiones import decisions
from experimentos.perfiles import main as profile_main
from src.config import TARGET_COLUMNS
from tests import test_perfiles


class TestAnalysis(unittest.TestCase):
    def make_inputs(self, root):
        data, central = test_perfiles.TestProfileEvaluation().make_inputs(root)
        profile_dir = root / 'profiles'
        profile_main(data, central, profile_dir, root / 'profile_summary')
        weighted = root / 'weighted'
        shutil.copytree(central, weighted)
        records = []
        for target in TARGET_COLUMNS:
            with np.load(central / target / 'splits.npz') as split:
                k = len(split['classes'])
                for model in ('softmax', 'coral', 'coral_weighted'):
                    source = weighted if model == 'coral_weighted' else central
                    (source / target / model).mkdir()
                    for fold in range(1, (2 if target == 'GDS' else 5) + 1):
                        indices = split[f'outer_{fold}_test']
                        probs = np.tile(np.arange(1, k+1) / np.arange(1, k+1).sum(), (len(indices), 1)).astype('float32')
                        mode, median = decisions(probs)
                        original = mode if model == 'softmax' else median
                        saved = {'row_ids': indices.tolist(), 'y_true': split['labels'][indices].tolist(),
                                 'y_pred': original.tolist(), 'probabilities': probs.tolist(), 'probability_dtype': 'float32'}
                        (source / target / model / f'fold_{fold}.json').write_text(json.dumps(saved))
                        for i, idx in enumerate(indices):
                            records.append({'objetivo': target, 'modelo': model, 'row_id': int(idx), 'fold': fold,
                                            'verdad': saved['y_true'][i], 'k': k, 'original': int(original[i]),
                                            'moda': int(mode[i]), 'mediana': int(median[i]), 'probs': json.dumps(probs[i].tolist())})
        path = root / 'decisions.csv'
        pd.DataFrame(records).sample(frac=1, random_state=7).to_csv(path, index=False)
        return path, profile_dir, central, weighted

    def test_signed_errors_pairs_and_absent_recall(self):
        probs = np.array([[.4, .35, .25], [.4, .35, .25], [.3, .3, .4], [.1, .1, .8]])
        mode, median = decisions(probs)
        rows = pd.DataFrame({'objetivo': 'GDS_R2', 'modelo': 'softmax', 'fold': 1,
                             'verdad': [0, 1, 2, 2], 'k': 3, 'moda': mode, 'mediana': median,
                             'original': mode, 'visto': [True, True, False, False], 'probs': list(probs)})
        metrics, recalls, pairs = tables(rows, {'GDS_R2': [1, 2, 3]}, True)
        p = pairs.loc[(pairs.grupo == 'todos') & (pairs.comparacion == 'mediana-moda')].iloc[0]
        self.assertEqual((p.n, p.n_cambios, p.mejoras_mae, p.empeora_mae, p.igual_mae), (4, 3, 1, 2, 1))
        self.assertEqual((p.delta_mae, p.delta_accuracy, p.delta_errores_graves), (.25, -.25, 0))
        med = metrics.loc[(metrics.grupo == 'todos') & (metrics.regla == 'mediana')].iloc[0]
        self.assertEqual((med.mae_ordinal, med.error_medio, med.subestimacion, med.sobreestimacion), (.5, 0, .25, .25))
        absent = recalls.loc[(recalls.grupo == 'visto') & (recalls.regla == 'mediana') & (recalls.clase_indice == 2)].iloc[0]
        self.assertEqual(absent.soporte, 0)
        self.assertTrue(np.isnan(absent.recall))

    def test_complete_pairs_dtype_and_invalid_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, profile_dir, central, weighted = self.make_inputs(root)
            rows, classes, _, _ = load_rows(path, profile_dir, central, weighted)
            self.assertEqual(len(rows), 70 * 6 * 4)
            self.assertEqual(classes['GDS_R3'], [1, 3])
            self.assertEqual(rows.loc[rows.modelo == 'softmax', 'probs'].iloc[0].dtype, np.float32)
            self.assertFalse(rows.duplicated(['objetivo', 'modelo', 'row_id']).any())
            original = pd.read_csv(path)
            changed_fold = original.copy()
            changed_fold.loc[0, 'fold'] = 99
            changed_truth = original.copy()
            changed_truth.loc[0, 'verdad'] = 99
            changed_probs = original.copy()
            changed_probs.loc[0, 'probs'] = '[1,0]'
            for bad in (pd.concat([original, original.iloc[:1]]), original.iloc[1:], changed_fold, changed_truth, changed_probs):
                bad.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    main(path, profile_dir, central, weighted, root / 'invalid_out')
                self.assertFalse((root / 'invalid_out').exists())

    def test_different_split_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            args = self.make_inputs(root)
            split_path = args[3] / 'GDS_R2' / 'splits.npz'
            with np.load(split_path) as s:
                data = dict(s)
            data['outer_1_test'] = data['outer_1_test'][::-1]
            np.savez(split_path, **data)
            with self.assertRaisesRegex(ValueError, 'Particiones distintas'):
                load_rows(*args)

    def test_empty_cohort_omitted_and_binary_rules_equal(self):
        probs = np.array([[.5, .5], [.1, .9]])
        mode, median = decisions(probs)
        rows = pd.DataFrame({'objetivo': 'GDS_R3', 'modelo': 'perfil', 'fold': 1, 'verdad': [0, 1],
                             'k': 2, 'moda': mode, 'mediana': median, 'original': np.nan,
                             'visto': True, 'probs': list(probs)})
        metrics, _, pairs = tables(rows, {'GDS_R3': [1, 3]}, False)
        self.assertNotIn('nuevo', metrics.grupo.tolist())
        self.assertTrue((pairs.n_cambios == 0).all())
        self.assertTrue((pairs.delta_mae == 0).all())


if __name__ == '__main__':
    unittest.main()
