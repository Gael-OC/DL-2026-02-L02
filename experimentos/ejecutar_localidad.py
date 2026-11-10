"""Panel fijo de localidad; salidas individuales solo en results/localidad."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from experimentos.localidad import apply_order_to_features, build_locality_model, get_feature_permutation
from src.config import FEATURE_COLUMNS, TrainingConfig
from src.data.loader import DataFrameLoader
from src.data.preprocessing import prepare_experiment_data, StratifiedSplitter
from src.evaluation.metrics import METRIC_KEYS, MetricsCalculator
from src.tasks.softmax_task import SoftmaxTask
from src.training.trainer import Trainer
from src.utils import set_seed

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('cnn_k3', 'cnn_k1', 'mlp_capacity')
ORDERS = ('original', 'blocks', 'global')
SPLIT_SEEDS = (42, 314, 2718)
INIT_SEEDS = (7, 17, 29)
METRICS = METRIC_KEYS + ['nll', 'rps']
CODE_FILES = ('experimentos/ejecutar_localidad.py', 'experimentos/localidad.py',
              'src/models/conv_softmax.py', 'src/models/blocks.py', 'src/config.py',
              'src/training/trainer.py', 'src/tasks/softmax_task.py', 'src/tasks/base.py',
              'src/data/loader.py', 'src/data/preprocessing.py', 'src/data/dataset.py',
              'src/evaluation/metrics.py', 'src/utils.py')


def jobs():
    return [dict(model=m, order=o, split_seed=s, init_seed=i, fold=f)
            for m, o, s, i, f in itertools.product(MODELS, ORDERS, SPLIT_SEEDS, INIT_SEEDS, range(1, 6))]


def job_name(job):
    return '{model}_{order}_s{split_seed}_i{init_seed}_f{fold}'.format(**job)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metricas(y, probabilities):
    pred = probabilities.argmax(axis=1)
    return {**MetricsCalculator.compute(y, pred, probabilities.shape[1]),
            'nll': MetricsCalculator.nll(y, probabilities), 'rps': MetricsCalculator.rps(y, probabilities)}


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def run(data_path, output, summary, smoke=False, report_only=False):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output.mkdir(parents=True, exist_ok=True)
    summary.mkdir(parents=True, exist_ok=True)
    folds_dir = output / 'folds'
    folds_dir.mkdir(exist_ok=True)
    data = prepare_experiment_data(DataFrameLoader().load(data_path), 'GDS_R2')
    X, y = data.X, data.y
    assert X.shape == (1119, 15) and len(data.classes) == 3
    splits = {s: StratifiedSplitter(5, s).split(y) for s in SPLIT_SEEDS}
    selected = [j for j in jobs() if not smoke or (j['split_seed'] == 42 and j['init_seed'] == 7 and j['fold'] == 1)]
    manifest = {'data_sha256': digest(data_path), 'code_sha256': {p: digest(ROOT / p) for p in CODE_FILES},
                'epochs': 1 if smoke else 20, 'target': 'GDS_R2', 'classes': data.classes,
                'optimizer': 'Adam', 'learning_rate': .001, 'weight_decay': .0001, 'batch_size': 32,
                'device': 'cpu', 'threads': 1, 'inner_folds': 0, 'grid': False,
                'seed_rule': 'init_seed + 1000 * fold (one-based); same for all models and orders',
                'orders': {o: {'indices': get_feature_permutation(o), 'names': [FEATURE_COLUMNS[i] for i in get_feature_permutation(o)]} for o in ORDERS},
                'parameters': {m: sum(p.numel() for p in build_locality_model(m).parameters()) for m in MODELS},
                'planned_jobs': len(selected), 'smoke': smoke}
    manifest_path = output / 'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError('Identidad distinta: use otro directorio; no se mezclan corridas.')
    else:
        atomic_json(manifest_path, manifest)
    atomic_json(output / 'planned_jobs.json', selected)
    atomic_json(output / 'splits.json', {str(s): [{'fold': f, 'train': tr.tolist(), 'test': te.tolist()} for f, (tr, te) in enumerate(parts, 1)] for s, parts in splits.items()})
    for n, job in enumerate(selected, 1):
        path = folds_dir / (job_name(job) + '.npz')
        train_idx, test_idx = splits[job['split_seed']][job['fold'] - 1]
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if (not np.array_equal(saved['index'], test_idx) or not np.array_equal(saved['y'], y[test_idx])
                        or saved['probabilities'].shape != (len(test_idx), len(data.classes))
                        or int(saved['epochs']) != manifest['epochs']):
                    raise ValueError(f'Fold inconsistente: {path}')
                metricas(saved['y'], saved['probabilities'])
            continue
        if report_only:
            raise ValueError(f'Falta fold: {path}')
        seed = job['init_seed'] + 1000 * job['fold']
        set_seed(seed)
        model = build_locality_model(job['model'])
        trainer = Trainer(model, SoftmaxTask(len(data.classes)), TrainingConfig(epochs=manifest['epochs']), torch.device('cpu'))
        ordered = apply_order_to_features(X, job['order'])
        losses = trainer.fit(ordered[train_idx], y[train_idx], seed)
        probabilities = trainer.predict_proba(ordered[test_idx])
        metricas(y[test_idx], probabilities)
        tmp = path.with_suffix('.tmp')
        with tmp.open('wb') as file:
            np.savez_compressed(file, index=test_idx, y=y[test_idx], probabilities=probabilities, losses=losses, epochs=manifest['epochs'])
        os.replace(tmp, path)
        print(f'{n}/{len(selected)} {job_name(job)}', flush=True)
    if smoke:
        print('Preflight: nueve ajustes de una época completos.', flush=True)
        return
    records, oof_records = [], []
    for model, order, split_seed, init_seed in itertools.product(MODELS, ORDERS, SPLIT_SEEDS, INIT_SEEDS):
        condition = dict(model=model, order=order, split_seed=split_seed, init_seed=init_seed)
        all_index, all_y, all_p = [], [], []
        for fold in range(1, 6):
            job = {**condition, 'fold': fold}
            with np.load(folds_dir / (job_name(job) + '.npz'), allow_pickle=False) as saved:
                idx, truth, probs = saved['index'], saved['y'], saved['probabilities']
                all_index.append(idx); all_y.append(truth); all_p.append(probs)
                records.append({**job, 'n': len(idx), **metricas(truth, probs)})
        index, truth, probs = map(np.concatenate, (all_index, all_y, all_p))
        if not np.array_equal(np.sort(index), np.arange(len(y))) or not np.array_equal(truth, y[index]):
            raise ValueError(f'Cobertura OOF incorrecta: {condition}')
        oof_records.append({**condition, 'n': len(index), **metricas(truth, probs)})
    frame = pd.DataFrame(records)
    oof = pd.DataFrame(oof_records)
    if len(frame) != 405 or len(oof) != 81:
        raise ValueError('Cobertura incompleta')
    frame.to_csv(summary / 'folds.csv', index=False)
    oof.to_csv(summary / 'oof_condiciones.csv', index=False)
    groups = frame.groupby(['model', 'order', 'split_seed', 'init_seed'])[METRICS]
    agg = groups.agg(['mean', lambda x: x.std(ddof=0)])
    agg.columns = [f'{a}_{"std0" if b == "<lambda_0>" else b}' for a, b in agg.columns]
    agg.reset_index().to_csv(summary / 'resumen_folds.csv', index=False)
    paired = oof.merge(oof[oof.model == 'cnn_k3'][['order', 'split_seed', 'init_seed', 'mae_ordinal']], on=['order', 'split_seed', 'init_seed'], suffixes=('', '_k3'))
    paired['delta_mae_vs_k3'] = paired.mae_ordinal - paired.mae_ordinal_k3
    paired[['model', 'order', 'split_seed', 'init_seed', 'delta_mae_vs_k3']].to_csv(summary / 'comparacion_pareada.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, model in zip(axes, MODELS):
        for k, split in enumerate(SPLIT_SEEDS):
            for j, init in enumerate(INIT_SEEDS):
                rows = oof[(oof.model == model) & (oof.split_seed == split) & (oof.init_seed == init)].set_index('order').loc[list(ORDERS)]
                ax.plot(np.arange(3), rows.mae_ordinal, marker=('o', 's', '^')[j], color=('tab:blue', 'tab:orange', 'tab:green')[k], alpha=.75, label=f'partición {split}, init {init}')
        ax.set_xticks(range(3), ORDERS); ax.set_title(model); ax.grid(alpha=.2)
    axes[0].set_ylabel('MAE ordinal OOF'); axes[-1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout(); fig.savefig(summary / 'mae_localidad.png', dpi=160); plt.close(fig)
    atomic_json(summary / 'metadatos.json', manifest)
    print('Validado: 405 folds, 81 condiciones y 1119 filas OOF por condición.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=Path, default=ROOT / 'dataset/15 atributos R0-R5.sav')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    suffix = 'preflight' if args.smoke else 'completo'
    run(args.data_path, ROOT / 'results/localidad' / suffix, ROOT / 'results/resumen/localidad' / suffix, args.smoke, args.report_only)


if __name__ == '__main__':
    main()
