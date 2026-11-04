"""Corrupciones que deben invalidar los resultados agregados."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
import numpy as np
from experimentos.auditar_corrida import auditar
from experimentos.comparacion import generar

SOURCE = Path('results/g4_2_central_unweighted')


class TestAuditoria(unittest.TestCase):
    def setUp(self):
        if not SOURCE.exists(): self.skipTest('Corrida central no disponible')
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.dest = Path(self.tmp.name) / 'corrida'; shutil.copytree(SOURCE, self.dest)

    def test_protocolo(self):
        p = self.dest / 'manifest.json'; data = json.loads(p.read_text())
        data['identity']['config']['training']['epochs'] = 19; p.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'Protocolo'): auditar(self.dest, 654)

    def test_indice_duplicado(self):
        p = self.dest / 'GDS/splits.npz'
        with np.load(p) as data: arrays = {k: data[k].copy() for k in data.files}
        a = arrays['outer_1_train']; a[1] = a[0]; np.savez(p, **arrays)
        with self.assertRaises(ValueError): auditar(self.dest, 654)

    def test_decision_y_probabilidad(self):
        p = self.dest / 'GDS/softmax/fold_1.json'; original = json.loads(p.read_text())
        for field in ('y_pred', 'probabilities'):
            data = json.loads(json.dumps(original))
            if field == 'y_pred': data[field][0] = (data[field][0] + 1) % 7
            else: data[field][0][0] += 0.1
            p.write_text(json.dumps(data))
            with self.subTest(field=field), self.assertRaises(ValueError): auditar(self.dest, 654)

    def test_comparacion_duplicada(self):
        with self.assertRaisesRegex(ValueError, 'duplicada'):
            generar([self.dest, self.dest], Path(self.tmp.name) / 'out')

if __name__ == '__main__': unittest.main()
