"""Contrato del predictor por perfil y evaluación con splits persistidos."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experimentos.decisiones import decisions
from experimentos.perfiles import ProfilePredictor, compare_reference, main, predict_fold, validated_splits
from src.config import FEATURE_COLUMNS, TARGET_COLUMNS
from src.data.loader import DataFrameLoader
from src.data.preprocessing import ExperimentData, StratifiedSplitter, prepare_experiment_data


class TestProfilePredictor(unittest.TestCase):
    def test_known_conflicting_duplicates_and_unseen_fallback(self):
        a, b, new = np.zeros(15), np.ones(15), np.eye(1, 15)[0]
        X = np.array([a, a, a, b])
        model = ProfilePredictor(3).fit(X, np.array([0, 2, 2, 1]))
        p = model.predict_proba(np.array([a, b, new]))
        np.testing.assert_allclose(p, [[2/6, 1/6, 3/6], [1/4, 2/4, 1/4], [2/7, 2/7, 3/7]])
        np.testing.assert_array_equal(model.seen([a, b, new]), [True, True, False])
        np.testing.assert_allclose(p.sum(1), 1)
        self.assertTrue(np.all(p > 0))

    def test_missing_classes_and_lower_ties(self):
        X = np.zeros((2, 15))
        p = ProfilePredictor(3).fit(X, np.array([0, 0])).predict_proba(X[:1])
        np.testing.assert_allclose(p, [[3/5, 1/5, 1/5]])
        tied = ProfilePredictor(2).fit(X, np.array([0, 1])).predict_proba(X[:1])
        mode, median = decisions(tied)
        self.assertEqual((mode.tolist(), median.tolist()), ([0], [0]))

    def test_test_labels_do_not_affect_predictions(self):
        X = np.array([np.zeros(15), np.zeros(15), np.ones(15), np.ones(15)])
        y = np.array([0, 1, 2, 1])
        train, test = np.array([0, 2]), np.array([1, 3])
        data = ExperimentData('GDS_R2', X, y, [1, 2, 3])
        before = predict_fold(data, train, test, 1)
        data.y = np.array([0, 2, 2, 0])  # Solo cambian las etiquetas externas.
        after = predict_fold(data, train, test, 1)
        for column in ('moda', 'mediana', 'visto'):
            np.testing.assert_array_equal(before[column], after[column])
        np.testing.assert_array_equal(np.stack(before.probs), np.stack(after.probs))
        self.assertFalse(np.array_equal(before.verdad, after.verdad))

    def test_invalid_inputs_and_refit_reset(self):
        X = np.zeros((2, 15))
        for k in (1, 2.5):
            with self.assertRaises(ValueError):
                ProfilePredictor(k)
        model = ProfilePredictor(3)
        with self.assertRaises(RuntimeError):
            model.predict_proba(X)
        with self.assertRaises(RuntimeError):
            model.seen(X)
        for bad in (np.zeros((2, 14)), np.full((2, 15), np.nan), np.full((2, 15), 2)):
            with self.assertRaises(ValueError):
                model.fit(bad, np.array([0, 1]))
        for labels in (np.array([]), np.array([0]), np.array([0., 1.]), np.array([0, 3]), np.array([-1, 0])):
            with self.assertRaises(ValueError):
                model.fit(X, labels)
        model.fit(X, np.array([0, 0])).fit(np.ones((1, 15)), np.array([2]))
        np.testing.assert_array_equal(model.seen(X), [False, False])
        np.testing.assert_allclose(model.predict_proba(X), [[1/4, 1/4, 2/4]] * 2)


class TestProfileEvaluation(unittest.TestCase):
    def make_inputs(self, root):
        # Archivo sintético con siete clases; R3 conserva el alfabeto original 1/3.
        n = 70
        frame = pd.DataFrame({name: (np.arange(n) >> (i % 4)) % 2 for i, name in enumerate(FEATURE_COLUMNS)})
        frame['GDS'] = np.repeat(np.arange(1, 8), 10)
        maps = {'GDS_R1': [1, 1, 1, 2, 2, 3, 3], 'GDS_R2': [1, 1, 2, 3, 3, 3, 3],
                'GDS_R3': [1, 1, 1, 3, 3, 3, 3], 'GDS_R4': [1, 2, 2, 2, 3, 3, 3],
                'GDS_R5': [1, 2, 2, 3, 3, 3, 3]}
        for target, mapping in maps.items():
            frame[target] = np.array(mapping)[frame.GDS - 1]
        data_path = root / 'datos.csv'
        frame.to_csv(data_path, index=False)
        splits_dir = root / 'central'
        splits_dir.mkdir()
        manifest = {'status': 'terminado', 'identity': {'data_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest()}}
        (splits_dir / 'manifest.json').write_text(json.dumps(manifest))
        for target in TARGET_COLUMNS:
            data = prepare_experiment_data(frame, target)
            splits = StratifiedSplitter(2 if target == 'GDS' else 5, 42).split(data.y)
            payload = {'labels': data.y, 'classes': data.classes, 'target': target, 'split_seed': 42}
            for fold, (train, test) in enumerate(splits, 1):
                payload.update({f'outer_{fold}_train': train, f'outer_{fold}_test': test})
            (splits_dir / target).mkdir()
            np.savez(splits_dir / target / 'splits.npz', **payload)
        return data_path, splits_dir

    def test_six_targets_exports_reference_and_preserves_previous(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, splits = self.make_inputs(root)
            local, out = root / 'run', root / 'summary'
            rows = main(path, splits, local, out)
            self.assertEqual(len(rows), 70 * 6)
            self.assertEqual(rows.groupby('objetivo').row_id.nunique().tolist(), [70] * 6)
            self.assertFalse(rows.duplicated(['objetivo', 'row_id']).any())
            self.assertEqual(len(pd.read_csv(out / 'folds.csv')), 27 * 2)
            self.assertEqual(len(pd.read_csv(out / 'resumen_folds.csv')), 6 * 2)
            self.assertEqual(len(pd.read_csv(out / 'oof.csv')), 6 * 2)
            manifest = json.loads((local / 'manifest.json').read_text())
            self.assertEqual(manifest['targets']['GDS_R3']['classes'], [1, 3])
            self.assertEqual(manifest['targets']['GDS']['folds'], 2)
            self.assertEqual(set(manifest['splits_sha256']), set(TARGET_COLUMNS))
            reference = root / 'reference.csv'
            old = pd.read_csv(local / 'filas.csv').sample(frac=1, random_state=7)
            pd.concat([old, old.assign(modelo='constante')]).to_csv(reference, index=False)
            second = main(path, splits, root / 'second', root / 'second_summary', reference)
            self.assertEqual(compare_reference(second, reference)['filas'], 420)
            saved = (local / 'filas.csv').read_bytes()
            with self.assertRaises(FileExistsError):
                main(path, splits, local, out)
            self.assertEqual((local / 'filas.csv').read_bytes(), saved)
            with self.assertRaises(ValueError):
                main(path, splits, root / 'overlap', root / 'overlap' / 'summary')

    def test_invalid_splits_data_and_reference_do_not_export(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, splits = self.make_inputs(root)
            data = prepare_experiment_data(DataFrameLoader().load(path), 'GDS_R5')
            split_path = splits / 'GDS_R5' / 'splits.npz'
            original = split_path.read_bytes()
            with np.load(split_path) as s:
                saved = dict(s)
            # Un fold repetido conserva separación, pero rompe la cobertura OOF.
            bad = {**saved, 'outer_2_train': saved['outer_1_train'], 'outer_2_test': saved['outer_1_test']}
            cases = [bad, {**saved, 'classes': [2, 3, 4]}, {**saved, 'labels': data.y[::-1]},
                     {**saved, 'outer_1_test': saved['outer_1_test'].astype(float)},
                     {**saved, 'outer_1_test': np.array([-1])},
                     {**saved, 'outer_1_train': saved['outer_1_test']},
                     {k: v for k, v in saved.items() if k != 'outer_5_test'}]
            for payload in cases:
                np.savez(split_path, **payload)
                with self.assertRaises(ValueError):
                    validated_splits(split_path, data)
            with self.assertRaises(ValueError):
                main(path, splits, root / 'run', root / 'summary')
            self.assertFalse((root / 'run').exists())
            self.assertFalse((root / 'summary').exists())
            split_path.write_bytes(original)
            path.write_text(path.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'datos no corresponden'):
                main(path, splits, root / 'run', root / 'summary')

    def test_reference_difference_does_not_export(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path, splits = self.make_inputs(root)
            main(path, splits, root / 'first', root / 'first_summary')
            reference = root / 'reference.csv'
            old = pd.read_csv(root / 'first' / 'filas.csv')
            for column, value in [('moda', 99), ('probs', '[1,0]')]:
                bad = old.copy()
                bad.loc[0, column] = value
                bad.to_csv(reference, index=False)
                with self.assertRaisesRegex(ValueError, 'Diferencias con la referencia'):
                    main(path, splits, root / 'new', root / 'new_summary', reference)
                self.assertFalse((root / 'new').exists())
                self.assertFalse((root / 'new_summary').exists())


if __name__ == '__main__':
    unittest.main()
