import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from src.config import ModelConfig, TrainingConfig
from src.data.preprocessing import ExperimentData
from src.training.nested_cv import NestedCrossValidator
from src.training.records import RunRecords
from src.training.trainer import Trainer


class TestRegistros(unittest.TestCase):
    def test_interrupcion_reanudacion_y_manifiesto(self):
        rng = np.random.default_rng(7)
        data = ExperimentData('GDS_R2', rng.normal(size=(24, 15)).astype('float32'), np.tile(np.arange(3), 8), [1, 2, 3])
        with tempfile.TemporaryDirectory() as folder:
            records = RunRecords(folder, {'seed': 42})
            validator = NestedCrossValidator('conv1d_softmax', ModelConfig(), TrainingConfig(epochs=1, batch_size=8), 2, 2,
                                              [{'hidden_dim': 16}, {'hidden_dim': 32}], 42, torch.device('cpu'),
                                              verbose=False, records=records, split_path=Path(folder) / 'splits.npz')
            original = records.save_fold
            def interrupt(target, variant, fold, result, candidates):
                if fold == 2:
                    raise RuntimeError('interrupcion simulada')
                original(target, variant, fold, result, candidates)
            with patch.object(records, 'save_fold', side_effect=interrupt):
                with self.assertRaises(RuntimeError):
                    validator.run(data)
            checkpoint = records.fold_path('GDS_R2', 'softmax', 1).read_bytes()
            with patch.object(Trainer, 'build', wraps=Trainer.build) as builds:
                result = validator.run(data)
                self.assertEqual(builds.call_count, 5)
            self.assertEqual(len(result.folds), 2)
            self.assertEqual(checkpoint, records.fold_path('GDS_R2', 'softmax', 1).read_bytes())
            with (Path(folder) / 'candidatas.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 4)
            for fold in ('1', '2'):
                items = [r for r in rows if r['fold'] == fold]
                self.assertEqual(sum(r['seleccionada'] == 'True' for r in items), 1)
                winner = next(r for r in items if r['seleccionada'] == 'True')
                self.assertEqual(float(winner['mae_mean']), min(float(r['mae_mean']) for r in items))
            events = [json.loads(x) for x in (Path(folder) / 'ajustes.jsonl').read_text().splitlines()]
            self.assertTrue(all(e['support_train'] and e['train_losses'] and e['status'] == 'terminado' for e in events))
            with self.assertRaises(ValueError):
                RunRecords(folder, {'seed': 314})

class TestReportes(unittest.TestCase):
    def test_regeneracion_y_variantes(self):
        from experimentos.reportes import regenerate_reports
        from src.evaluation.metrics import MetricsCalculator
        rng = np.random.default_rng(10)
        data = ExperimentData('GDS_R2', rng.normal(size=(24, 15)).astype('float32'), np.tile(np.arange(3), 8), [1, 3, 5])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            records = RunRecords(path, {'test': True})
            for weighted in (False, True):
                validator = NestedCrossValidator('conv1d_coral', ModelConfig(), TrainingConfig(epochs=1, use_class_weights=weighted),
                                                2, 2, None, 42, torch.device('cpu'), verbose=False, records=records,
                                                split_path=path / 'GDS_R2' / 'splits.npz')
                validator.run(data)
            with patch.object(Trainer, 'build', side_effect=AssertionError('no debe entrenar')):
                outputs = regenerate_reports(path)
            self.assertEqual({x.name for x in outputs}, {'coral', 'coral_weighted'})
            for variant in outputs:
                with (variant / 'predicciones_oof.csv').open() as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual([int(r['row_id']) for r in rows], list(range(24)))
                self.assertTrue(all(int(r['verdad_original']) == data.classes[int(r['verdad_indice'])] for r in rows))
                true = np.array([int(r['verdad_indice']) for r in rows])
                pred = np.array([int(r['decision_indice']) for r in rows])
                probs = np.array([json.loads(r['probabilidades']) for r in rows], dtype=np.float32)
                report = json.loads((variant / 'reporte.json').read_text())
                self.assertAlmostEqual(report['agregado_oof']['nll'], MetricsCalculator.nll(true, probs), places=6)
                self.assertAlmostEqual(report['agregado_oof']['mae_ordinal'], MetricsCalculator.mae_ordinal(true, pred))
                with (variant / 'confusion_oof.csv').open() as f:
                    matrix = list(csv.DictReader(f))
                self.assertEqual(sum(int(row[str(c)]) for row in matrix for c in data.classes), 24)
                self.assertEqual(report['resumen_folds_externos']['nll']['folds_validos'], 2)


if __name__ == '__main__':
    unittest.main()
