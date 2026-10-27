"""Auditoria reproducible del SAV; el informe contiene solo datos agregados."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

# Permite ejecutar el archivo directamente desde cualquier directorio.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FEATURE_COLUMNS, ID_COLUMN, TARGET_COLUMNS
from src.data.loader import DataFrameLoader
from src.data.preprocessing import prepare_experiment_data

# El indice es el valor original de GDS menos uno; R3 conserva sus etiquetas 1 y 3.
AGRUPACIONES = {
    "GDS_R1": [1, 1, 1, 2, 2, 3, 3],
    "GDS_R2": [1, 1, 2, 3, 3, 3, 3],
    "GDS_R3": [1, 1, 1, 3, 3, 3, 3],
    "GDS_R4": [1, 2, 2, 2, 3, 3, 3],
    "GDS_R5": [1, 2, 2, 3, 3, 3, 3],
}
DOMINIOS = {"GDS": set(range(1, 8)), **{k: set(v) for k, v in AGRUPACIONES.items()}}


def auditar(dataframe: pd.DataFrame, rangos_ausentes: dict) -> dict:
    """Comprueba el contrato del dataset y devuelve exclusivamente agregados.

    Rechaza datos invalidos antes de calcular perfiles o preparar el modelo.
    Los rangos de ausencia declarados en el SAV se comprueban sobre los valores
    originales, leidos con user_missing=True.
    """
    columnas = [ID_COLUMN, *FEATURE_COLUMNS, *TARGET_COLUMNS]
    if len(set(columnas)) != len(columnas):
        raise ValueError("ID, entradas y objetivos deben ser columnas distintas.")
    if dataframe.columns.duplicated().any():
        raise ValueError("Hay nombres de columnas duplicados.")
    faltan = [col for col in columnas if col not in dataframe.columns]
    if faltan:
        raise ValueError(f"Faltan columnas requeridas: {faltan}")
    if dataframe.empty:
        raise ValueError("El archivo no tiene filas.")

    datos = dataframe[columnas].apply(pd.to_numeric, errors="coerce")
    no_finitos = {col: int((~np.isfinite(datos[col])).sum()) for col in columnas}
    no_finitos = {col: n for col, n in no_finitos.items() if n}
    if no_finitos:
        raise ValueError(f"Valores ausentes, no numericos o no finitos: {no_finitos}")

    ausencias = {}
    for col, rangos in rangos_ausentes.items():
        if col not in datos.columns:
            continue
        mascara = pd.Series(False, index=datos.index)
        for rango in rangos:
            mascara |= datos[col].between(rango["lo"], rango["hi"])
        ausencias[col] = {"rangos": rangos, "valores_observados": int(mascara.sum())}
    observadas = {col: info["valores_observados"] for col, info in ausencias.items() if info["valores_observados"]}
    if observadas:
        raise ValueError(f"Valores declarados como ausentes en el SAV: {observadas}")

    incorrectos = {col: int((~datos[col].isin({0, 1})).sum()) for col in FEATURE_COLUMNS}
    incorrectos.update({col: int((~datos[col].isin(dominio)).sum()) for col, dominio in DOMINIOS.items()})
    incorrectos = {col: n for col, n in incorrectos.items() if n}
    if incorrectos:
        raise ValueError(f"Valores fuera del dominio esperado: {incorrectos}")

    id_duplicados = int(datos[ID_COLUMN].duplicated().sum())
    if id_duplicados:
        raise ValueError(f"ID duplicados: {id_duplicados}")

    discrepancias = {}
    for col, valores in AGRUPACIONES.items():
        esperado = datos["GDS"].map(dict(zip(range(1, 8), valores)))
        discrepancias[col] = int((datos[col] != esperado).sum())
    if any(discrepancias.values()):
        raise ValueError(f"Discrepancias en recodificaciones: {discrepancias}")

    loader = DataFrameLoader()
    if loader.feature_columns != FEATURE_COLUMNS or set(loader.feature_columns) & {ID_COLUMN, *TARGET_COLUMNS}:
        raise ValueError("Las entradas no respetan el orden o incluyen ID/objetivos.")
    for objetivo in TARGET_COLUMNS:
        preparado = prepare_experiment_data(datos, objetivo, loader)
        if preparado.X.shape != (len(datos), 15) or not np.array_equal(preparado.X, datos[FEATURE_COLUMNS].to_numpy()):
            raise ValueError("La matriz X no coincide con las 15 entradas esperadas.")
        clases = sorted(datos[objetivo].unique().astype(int).tolist())
        esperado = datos[objetivo].map({clase: i for i, clase in enumerate(clases)}).to_numpy()
        if preparado.classes != clases or not np.array_equal(preparado.y, esperado):
            raise ValueError(f"Codificacion ordinal incorrecta: {objetivo}")

    perfiles = datos.groupby(FEATURE_COLUMNS, sort=False).agg(
        filas=("GDS", "size"), clases_gds=("GDS", "nunique")
    )
    ambiguos = perfiles[perfiles["clases_gds"] > 1]
    return {
        "filas": len(datos),
        "entradas": len(FEATURE_COLUMNS),
        "faltantes_o_no_finitos": 0,
        "id_unicos": int(datos[ID_COLUMN].nunique()),
        "ausencias_declaradas": ausencias,
        "discrepancias_recodificacion": discrepancias,
        "conteos_objetivos": {
            col: {str(int(clase)): int(n) for clase, n in datos[col].value_counts().sort_index().items()}
            for col in TARGET_COLUMNS
        },
        "perfiles": len(perfiles),
        "perfiles_ambiguos_gds": len(ambiguos),
        "filas_en_perfiles_ambiguos_gds": int(ambiguos["filas"].sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archivo", type=Path, help="Archivo SAV del laboratorio")
    parser.add_argument("--salida", type=Path, help="Guardar resumen agregado en JSON")
    args = parser.parse_args()
    try:
        if args.salida and args.salida.resolve() == args.archivo.resolve():
            raise ValueError("La salida no puede sobrescribir el archivo de datos.")
        dataframe, metadata = pyreadstat.read_sav(str(args.archivo), user_missing=True)
        resumen = auditar(dataframe, metadata.missing_ranges)
        resumen = {"sha256_sav": hashlib.sha256(args.archivo.read_bytes()).hexdigest(), **resumen}
        texto = json.dumps(resumen, ensure_ascii=False, indent=2) + "\n"
        if args.salida:
            args.salida.parent.mkdir(parents=True, exist_ok=True)
            args.salida.write_text(texto, encoding="utf-8")
        print(texto, end="")
    except (ValueError, OSError, pyreadstat.ReadstatError) as error:
        print(f"Auditoria fallida: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
