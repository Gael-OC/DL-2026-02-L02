"""Pruebas del contrato de datos, sin publicar registros individuales."""

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyreadstat

from experimentos.auditoria_datos import auditar
from src.config import FEATURE_COLUMNS, TARGET_COLUMNS
from src.data.loader import DataFrameLoader


class AuditoriaDatosTest(unittest.TestCase):
    def setUp(self):
        # Dos filas comparten respuestas pero tienen GDS distinto.
        self.datos = pd.DataFrame({col: [0, 0, 1] for col in FEATURE_COLUMNS})
        self.datos["ID"] = [10, 11, 12]
        self.datos["GDS"] = [1, 2, 7]
        self.datos["GDS_R1"] = [1, 1, 3]
        self.datos["GDS_R2"] = [1, 1, 3]
        self.datos["GDS_R3"] = [1, 1, 3]
        self.datos["GDS_R4"] = [1, 2, 3]
        self.datos["GDS_R5"] = [1, 2, 3]

    def test_resumen_y_etiquetas_no_consecutivas(self):
        resumen = auditar(self.datos, {"GDS": [{"lo": 0, "hi": 0}]})
        self.assertEqual((resumen["perfiles"], resumen["perfiles_ambiguos_gds"], resumen["filas_en_perfiles_ambiguos_gds"]), (2, 1, 2))
        self.assertEqual(resumen["conteos_objetivos"]["GDS_R3"], {"1": 2, "3": 1})
        self.assertTrue(all(n == 0 for n in resumen["discrepancias_recodificacion"].values()))

    def test_rechaza_problemas_que_invalidan_el_dataset(self):
        casos = [
            ("Día", np.inf, {}, "no finitos"),
            ("Día", 2, {}, "dominio"),
            ("GDS", 0, {"GDS": [{"lo": 0, "hi": 0}]}, "declarados como ausentes"),
            ("GDS_R2", 2, {}, "Discrepancias"),
            ("ID", 11, {}, "ID duplicados"),
        ]
        for columna, valor, rangos, mensaje in casos:
            with self.subTest(columna=columna, valor=valor):
                datos = self.datos.copy()
                datos[columna] = datos[columna].astype(float)
                datos.loc[0, columna] = valor
                with self.assertRaisesRegex(ValueError, mensaje):
                    auditar(datos, rangos)

    def test_no_acepta_identificador_como_entrada(self):
        loader = DataFrameLoader([*FEATURE_COLUMNS[:-1], "ID"])
        with patch("experimentos.auditoria_datos.DataFrameLoader", return_value=loader):
            with self.assertRaisesRegex(ValueError, "entradas"):
                auditar(self.datos, {})

    def test_rechaza_columna_ausente(self):
        with self.assertRaisesRegex(ValueError, "Faltan columnas"):
            auditar(self.datos.drop(columns=TARGET_COLUMNS[-1]), {})

    def test_archivo_real_si_esta_disponible(self):
        archivo = Path(__file__).resolve().parents[1] / "dataset/15 atributos R0-R5.sav"
        if not archivo.exists():
            self.skipTest("SAV local no disponible")
        datos, metadata = pyreadstat.read_sav(str(archivo), user_missing=True)
        resumen = auditar(datos, metadata.missing_ranges)
        self.assertEqual((resumen["filas"], resumen["entradas"], resumen["id_unicos"]), (1119, 15, 1119))
        self.assertEqual((resumen["perfiles"], resumen["perfiles_ambiguos_gds"], resumen["filas_en_perfiles_ambiguos_gds"]), (321, 97, 851))
        self.assertEqual(resumen["ausencias_declaradas"]["GDS"]["valores_observados"], 0)
        self.assertTrue(all(n == 0 for n in resumen["discrepancias_recodificacion"].values()))


if __name__ == "__main__":
    unittest.main()
