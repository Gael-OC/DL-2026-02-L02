"""Comprueba el contrato de formas y la localidad de la red base."""

import unittest

import torch
import torch.nn.functional as F

from experimentos.inspeccionar_arquitectura import inspeccionar
from src.models.conv_softmax import Conv1DSoftmaxNet


class ArquitecturaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resumen = inspeccionar()

    def test_formas_y_equivalencia_de_entradas(self):
        self.assertEqual(self.resumen["entrada"], [2, 15])
        self.assertEqual(self.resumen["entrada_con_canal"], [2, 1, 15])
        self.assertEqual(self.resumen["formas"], {
            "conv_1": [2, 16, 15], "conv_2": [2, 32, 15],
            "flatten": [2, 480], "mlp": [2, 32], "salida": [2, 3],
        })
        self.assertTrue(self.resumen["salidas_finitas"])
        self.assertTrue(self.resumen["entradas_equivalentes_en_evaluacion"])

    def test_parametros(self):
        self.assertEqual(self.resumen["parametros_totales"], 17219)
        self.assertEqual(self.resumen["parametros_por_capa"], {
            "conv_1": 96, "conv_2": 1632, "flatten": 0, "mlp": 15392, "salida": 99,
        })

    def test_ventana_y_campo_receptivo(self):
        ejemplo = self.resumen["ejemplo_convolucion"]
        self.assertEqual(ejemplo["ventana"], [6, 7, 8])
        self.assertEqual(ejemplo["ventana_borde"], [0, 0, 1])
        self.assertEqual(ejemplo["salida"], 44)
        self.assertEqual(self.resumen["dos_convoluciones_posiciones_afectadas_por_impulso_central"], [5, 6, 7, 8, 9])

    def test_flatten_conserva_posiciones(self):
        # El mismo filtro activado en posiciones distintas produce vectores distintos.
        x = torch.zeros(1, 32, 15)
        y = x.clone()
        x[0, 0, 3] = 1
        y[0, 0, 4] = 1
        flatten = torch.nn.Flatten()
        self.assertFalse(torch.equal(flatten(x), flatten(y)))
        self.assertTrue(torch.equal(F.adaptive_avg_pool1d(x, 1), F.adaptive_avg_pool1d(y, 1)))

    def test_lote_individual_en_evaluacion(self):
        modelo = Conv1DSoftmaxNet(num_features=15, num_classes=3).eval()
        with torch.no_grad():
            salida = modelo(torch.zeros(1, 15))
        self.assertEqual(tuple(salida.shape), (1, 3))
        self.assertTrue(torch.isfinite(salida).all())


if __name__ == "__main__":
    unittest.main()
