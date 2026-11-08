"""
Ejecución masiva del panel de localidad y orden de características (J4.1).
Realiza los 405 ajustes especificados en el protocolo del Laboratorio 02,
incorporando registro de logs y manejo robusto de errores.

ejecutar:
nohup python experimentos/ejecutar_localidad.py --data-path "dataset/15 atributos R0-R5.sav" --target-name GDS_R2 > results/localidad/console_output.log 2>&1 &

logs con errores:
cat results/localidad/console_output.log

ver logs de progreso en tiempo real:
tail -f results/localidad/resultados_localidad.log

verificar que se está ejecutando (sin contar el pid de grep):
ps aux | grep ejecutar_localidad.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import logging
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import DataFrameLoader
from src.data.preprocessing import prepare_experiment_data, StratifiedSplitter
from src.data.dataset import CognitiveSequenceDataset
from src.evaluation.metrics import MetricsCalculator
from experimentos.localidad import build_locality_model, apply_order_to_features
from src.utils import resolve_device


def configurar_logger(output_dir: Path) -> logging.Logger:
    """Configura el logger para escribir simultáneamente en consola y archivo."""
    logger = logging.getLogger("LocalidadRunner")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Handler para consola
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Handler para archivo de log
    log_path = output_dir / "resultados_localidad.log"
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def ejecutar_panel_localidad(
    sav_path: Path,
    target_name: str = "GDS_R2",
    output_dir: Path = Path("results/localidad")
):
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = configurar_logger(output_dir)
    
    logger.info(f"Iniciando carga de datos desde: {sav_path} para objetivo: {target_name}")
    
    # 1. Cargar datos con el loader oficial
    loader = DataFrameLoader()
    df = loader.load(sav_path)
    preparado = prepare_experiment_data(df, target_name, loader)
    X_base, y, classes = preparado.X, preparado.y, preparado.classes
    num_classes = len(classes)
    
    logger.info(f"Datos cargados con éxito. Muestras: {len(X_base)} | Clases: {num_classes}")

    # Parámetros del panel según protocolo (Sección 3.4)
    modelos_nombres = ["cnn_k3", "cnn_k1", "mlp_capacity"]
    ordenes = ["original", "blocks", "global"]
    particion_seeds = [42, 314, 2718]
    init_seeds = [7, 17, 29]
    n_splits = 5
    epochs = 20
    batch_size = 32
    lr = 1e-3
    weight_decay = 1e-4

    device = resolve_device("auto")
    logger.info(f"Dispositivo seleccionado para ejecución: {device}")

    resultados_totales = []
    total_ajustes = len(modelos_nombres) * len(ordenes) * len(particion_seeds) * len(init_seeds) * n_splits
    contador = 0
    errores_contador = 0

    # 2. Iteración completa sobre las dimensiones del panel
    for p_seed in particion_seeds:
        splitter = StratifiedSplitter(n_splits=n_splits, random_state=p_seed)
        
        for order_type in ordenes:
            X_ord = apply_order_to_features(X_base, order_type)
            
            for outer_fold, (train_idx, test_idx) in enumerate(splitter.split(y), start=1):
                X_tr, y_tr = X_ord[train_idx], y[train_idx]
                X_te, y_te = X_ord[test_idx], y[test_idx]
                
                for init_seed in init_seeds:
                    for mod_name in modelos_nombres:
                        contador += 1
                        msg_base = f"[{contador}/{total_ajustes}] Split: {p_seed} | Orden: {order_type} | Fold: {outer_fold} | Init: {init_seed} | Modelo: {mod_name}"
                        
                        try:
                            # Fijar semillas
                            torch.manual_seed(init_seed)
                            np.random.seed(init_seed)
                            
                            # Construir modelo
                            model = build_locality_model(mod_name, num_features=15, num_classes=num_classes).to(device)
                            
                            # Preparar dataloaders
                            train_dataset = CognitiveSequenceDataset(X_tr, y_tr.astype(np.int64))
                            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                            
                            test_dataset = CognitiveSequenceDataset(X_te, np.zeros(len(X_te), dtype=np.int64))
                            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                            
                            # Entrenamiento
                            criterion = nn.CrossEntropyLoss()
                            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                            
                            model.train()
                            for _ in range(epochs):
                                for inputs, targets in train_loader:
                                    inputs = inputs.to(device)
                                    targets = targets.to(device).long()
                                    
                                    optimizer.zero_grad()
                                    logits = model(inputs)
                                    if logits.ndim > 2:
                                        logits = logits.squeeze(1)
                                    loss = criterion(logits, targets)
                                    loss.backward()
                                    optimizer.step()
                            
                            # Evaluación OOF
                            model.eval()
                            preds_list = []
                            with torch.no_grad():
                                for inputs, _ in test_loader:
                                    logits = model(inputs.to(device))
                                    if logits.ndim > 2:
                                        logits = logits.squeeze(1)
                                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                                    preds_list.append(preds)
                            y_pred = np.concatenate(preds_list)
                            
                            metricas = MetricsCalculator.compute(y_te, y_pred)
                            
                            registro = {
                                "objetivo": target_name,
                                "split_seed": p_seed,
                                "orden": order_type,
                                "fold": outer_fold,
                                "init_seed": init_seed,
                                "modelo": mod_name,
                                **metricas
                            }
                            resultados_totales.append(registro)
                            logger.info(f"{msg_base} -> OK | Acc: {metricas['accuracy']:.4f} | MAE: {metricas['mae_ordinal']:.4f}")

                        except Exception as e:
                            errores_contador += 1
                            logger.error(f"{msg_base} -> ERROR: {str(e)}", exc_info=True)

    # Guardar resultados consolidados en CSV
    df_resultados = pd.DataFrame(resultados_totales)
    csv_path = output_dir / f"metricas_localidad_{target_name}.csv"
    df_resultados.to_csv(csv_path, index=False, encoding="utf-8")
    
    logger.info(f"Ejecución finalizada. Ajustes exitosos: {len(resultados_totales)}/{total_ajustes}. Errores: {errores_contador}")
    logger.info(f"Resultados guardados en: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Ejecutar panel de localidad J4.1 con logs")
    parser.add_argument("--data-path", type=Path, required=True, help="Ruta al archivo SAV")
    parser.add_argument("--target-name", type=str, default="GDS_R2", help="Objetivo a evaluar")
    parser.add_argument("--output-dir", type=Path, default=Path("results/localidad"), help="Directorio de salida")
    args = parser.parse_args()
    
    ejecutar_panel_localidad(args.data_path, args.target_name, args.output_dir)


if __name__ == "__main__":
    main()