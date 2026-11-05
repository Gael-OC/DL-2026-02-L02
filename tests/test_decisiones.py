import unittest
import numpy as np
from experimentos.decisiones import decisions


class DecisionTests(unittest.TestCase):
    def test_rules_ties_and_preserve_probabilities(self):
        p = np.array([[.5, .5, 0], [.3, .3, .4]])
        before = p.copy()
        moda, mediana = decisions(p)
        self.assertEqual(moda.tolist(), [0, 2])
        self.assertEqual(mediana.tolist(), [0, 1])
        np.testing.assert_array_equal(p, before)

    def test_invalid_probabilities(self):
        for p in ([[.2, .2]], [[float('nan'), 1]], [[-.1, 1.1]]):
            with self.assertRaises(ValueError): decisions(p)

    def test_faltan_entradas_no_escribe_salidas(self):
        import tempfile
        from pathlib import Path
        from experimentos.decisiones import main
        from src.config import TARGET_COLUMNS
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); local = root / 'local'; out = root / 'out'
            local.mkdir(); out.mkdir()
            for p in (local / 'filas.csv', out / 'folds.csv'): p.write_text('conservar')
            with self.assertRaisesRegex(ValueError, 'Faltan objetivos'):
                main(root / 'ausente', root / 'ausente', local, out)
            for target in TARGET_COLUMNS:
                target_path = root / 'source' / target; target_path.mkdir(parents=True)
                np.savez(target_path / 'splits.npz', labels=np.array([0, 1]), classes=np.array([1, 2]))
            with self.assertRaisesRegex(ValueError, 'Faltan folds de la variante'):
                main(root / 'source', root / 'ausente', local, out)
            self.assertEqual((local / 'filas.csv').read_text(), 'conservar')
            self.assertEqual((out / 'folds.csv').read_text(), 'conservar')
