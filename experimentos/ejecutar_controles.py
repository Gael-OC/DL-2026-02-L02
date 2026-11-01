"""
Ejecución de controles (Ficha J2.3) para el Laboratorio 02.
Evalúa los modelos de control en GDS_R2 con 5 folds externos y selección interna.
//ejecutar con:
python experimentos/ejecutar_controles.py --data-path "dataset/15 atributos R0-R5.sav" --target-name GDS_R2
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import DataFrameLoader
from src.data.preprocessing import prepare_experiment_data, StratifiedSplitter
from src.evaluation.metrics import MetricsCalculator
from experimentos.controles import (
    ConstantPredictor,
    LogisticControlPredictor,
    MLPControlPredictor,
    HistGradientBoostingControlPredictor,
)
from src.config import DEFAULT_RANDOM_SEED

def ejecutar_evaluacion_controles(sav_path: Path, target_name: str = "GDS_R2", output_dir: Path = Path("results/controles")):
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Cargar datos utilizando el loader oficial del proyecto
    loader = DataFrameLoader()
    df = loader.load(sav_path)
    preparado = prepare_experiment_data(df, target_name, loader)
    X, y, classes = preparado.X, preparado.y, preparado.classes
    
    print(f"Ejecutando controles para objetivo: {target_name} | Muestras: {len(X)} | Clases: {classes}")

    # 2. Configurar particionador externo oficial (5 folds y semilla)
    splitter = StratifiedSplitter(n_splits=5, random_state=DEFAULT_RANDOM_SEED)
    if not splitter.can_split(y):
        print(f"Advertencia: El objetivo {target_name} no cumple con la estratificación estricta de 5 folds.")

    resultados_folds = []
    
    for outer_fold, (train_idx, test_idx) in enumerate(splitter.split(y), start=1):
        print(f"\n--- Outer Fold {outer_fold} / 5 ---")
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        
        # Partición interna (3 folds) para la búsqueda de hiperparámetros (grid interno)
        inner_splitter = StratifiedSplitter(n_splits=3, random_state=DEFAULT_RANDOM_SEED + 100 * outer_fold)
        if inner_splitter.can_split(y_train):
            inner_train_idx, inner_val_idx = next(iter(inner_splitter.split(y_train)))
        else:
            # Fallback seguro si el fold interno es muy pequeño
            n_val = max(1, len(y_train) // 4)
            inner_val_idx = np.arange(n_val)
            inner_train_idx = np.arange(n_val, len(y_train))

        X_tr, y_tr = X_train[inner_train_idx], y_train[inner_train_idx]
        X_va, y_va = X_train[inner_val_idx], y_train[inner_val_idx]

        # Definir modelos de control a evaluar
        modelos = {
            "constante_moda": ConstantPredictor(strategy='mode', classes=classes),
            "constante_mediana": ConstantPredictor(strategy='median', classes=classes),
            "logistica_suma": LogisticControlPredictor(mode='sum'),
            "logistica_15": LogisticControlPredictor(mode='features'),
            "mlp_sencilla": MLPControlPredictor(epochs=10, batch_size=32),
            "hist_gradient_boosting": HistGradientBoostingControlPredictor(),
        }

        for nombre, modelo in modelos.items():
            if hasattr(modelo, "fit_grid_search"):
                modelo.fit_grid_search(X_tr, y_tr, X_va, y_va, classes)
            else:
                modelo.fit(y_tr)
                
            y_pred = modelo.predict(X_test)
            metricas = MetricsCalculator.compute(y_test, y_pred)
            
            registro = {
                "objetivo": target_name,
                "fold": outer_fold,
                "modelo": nombre,
                **metricas
            }
            resultados_folds.append(registro)
            print(f"[{nombre}] - Acc: {metricas['accuracy']:.4f} | MAE: {metricas['mae_ordinal']:.4f} | QWK: {metricas['qwk']:.4f}")

    # Guardar resultados consolidados en CSV
    df_resultados = pd.DataFrame(resultados_folds)
    csv_path = output_dir / f"metricas_controles_{target_name}.csv"
    df_resultados.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\nResultados guardados exitosamente en: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Ejecutar controles para J2.3")
    parser.add_argument("--data-path", type=Path, required=True, help="Ruta al archivo SAV")
    parser.add_argument("--target-name", type=str, default="GDS_R2", help="Objetivo a evaluar")
    parser.add_argument("--output-dir", type=Path, default=Path("results/controles"), help="Directorio de salida")
    args = parser.parse_args()
    
    ejecutar_evaluacion_controles(args.data_path, args.target_name, args.output_dir)


if __name__ == "__main__":
    main()