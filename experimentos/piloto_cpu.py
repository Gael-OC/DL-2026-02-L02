"""Mide ajustes representativos R2 en CPU, solos o en procesos independientes."""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time


def ajuste(data_path):
    import numpy as np
    import torch
    from src.config import ModelConfig, TrainingConfig
    from src.data.loader import DataFrameLoader
    from src.data.preprocessing import StratifiedSplitter, prepare_experiment_data
    from src.training.trainer import Trainer

    inicio = time.perf_counter()
    datos = prepare_experiment_data(DataFrameLoader().load(data_path), "GDS_R2")
    train, test = StratifiedSplitter(5, 42).split(datos.y)[0]
    trainer = Trainer.build("conv1d_softmax", ModelConfig(), TrainingConfig(),
                            datos.num_features, datos.num_classes, torch.device("cpu"), 42)
    resultado = trainer.fit_and_evaluate(datos.X[train], datos.y[train],
                                        datos.X[test], datos.y[test], 42)
    if not all(np.isfinite(v) for v in resultado.metrics.values()):
        raise ValueError("Métricas no finitas")
    return {"segundos": time.perf_counter() - inicio, "metricas": resultado.metrics,
            "filas_train": len(train), "filas_test": len(test)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(ajuste(args.data_path)))
        return
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="")

    def run_one():
        output = subprocess.check_output([sys.executable, "-m", "experimentos.piloto_cpu",
                                          args.data_path, "--worker"], env=env, text=True)
        return json.loads(output)

    for workers in (1, 2):
        for repetition in range(1, 4):
            inicio = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(lambda _: run_one(), range(workers)))
            total = time.perf_counter() - inicio
            print(json.dumps({"procesos": workers, "repeticion": repetition,
                              "segundos_totales": total, "ajustes_por_minuto": 60 * workers / total,
                              "resultados": results}), flush=True)


if __name__ == "__main__":
    main()
