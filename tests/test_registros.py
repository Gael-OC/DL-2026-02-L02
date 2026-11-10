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
