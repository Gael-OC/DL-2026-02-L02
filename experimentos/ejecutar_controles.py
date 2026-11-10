"""Evalúa controles con las particiones anidadas guardadas de GDS_R2."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import platform

import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from experimentos.controles import ConstantPredictor, MLPControlPredictor, ShallowMultiClassNet
from src.config import ModelConfig, TrainingConfig
from src.data.loader import DataFrameLoader
from src.data.preprocessing import prepare_experiment_data
from src.evaluation.metrics import METRIC_KEYS, MetricsCalculator as Metrics
from src.training.search import CandidateScore
from src.utils import set_seed

CENTRAL = Path('results/g4_2_central_unweighted')
CANDIDATES = {
    'logistica_suma': [{'C': c} for c in (0.01, 0.1, 1.0, 10.0)],
    'logistica_15': [{'C': c} for c in (0.01, 0.1, 1.0, 10.0)],
    'mlp_sencilla': [{'hidden_dim': h, 'dropout': d} for h in (32, 64) for d in (0.15, 0.30)],
    'hist_gradient_boosting': [{'max_leaf_nodes': n} for n in (3, 7, 15, 31)],
}


def fit_predict(name, config, seed, X_train, y_train, X_eval, k):
    """Ajusta una candidata y entrega decisiones y probabilidades en orden 0..K-1."""
    if name == 'mlp_sencilla':
        set_seed(seed)
        model = ShallowMultiClassNet(input_dim=X_train.shape[1], output_dim=k, **config)
        trainer = MLPControlPredictor(seed=seed, epochs=20, batch_size=32, lr=0.001, weight_decay=0.0001)
        trainer._train_model(model, X_train, y_train, torch.device('cpu'))
        model.eval()
        with torch.no_grad():
            logits = model(torch.as_tensor(X_eval, dtype=torch.float32))
            assert logits.shape == (len(X_eval), k)
            probs = torch.softmax(logits, dim=1).numpy()
        return probs.argmax(axis=1), probs
    if name == 'logistica_suma':
        X_train, X_eval = X_train.sum(axis=1, keepdims=True), X_eval.sum(axis=1, keepdims=True)
    if name.startswith('logistica'):
        model = LogisticRegression(**config, penalty='l2', solver='lbfgs', max_iter=2000, random_state=seed)
    else:
        model = HistGradientBoostingClassifier(**config, max_iter=100, learning_rate=0.1,
                                                l2_regularization=1.0, early_stopping=False, random_state=seed)
    model.fit(X_train, y_train)
    probs = np.zeros((len(X_eval), k))
    probs[:, model.classes_.astype(int)] = model.predict_proba(X_eval)
    return model.predict(X_eval), probs


def metrics(y, pred, probs, k):
    result = Metrics.compute(y, pred, k)
    result.update(nll=Metrics.nll(y, probs), rps=Metrics.rps(y, probs))
    return result


def evaluate(X, y, splits, classes, output_dir):
    """Ejecuta selección y refit; índices de todas las fases son globales."""
    k = len(classes)
    fit_records, candidates, folds, oof = [], [], [], []
    for outer in range(1, 6):
        prefix = f'outer_{outer}'
        train, test = splits[f'{prefix}_train'], splits[f'{prefix}_test']
        if set(train) & set(test) or set(train) | set(test) != set(range(len(y))):
            raise ValueError('Partición externa inválida')
        inner = [(splits[f'{prefix}_inner_{i}_train'], splits[f'{prefix}_inner_{i}_val']) for i in range(1, 4)]
        if sorted(np.concatenate([v for _, v in inner]).tolist()) != sorted(train.tolist()):
            raise ValueError('Validaciones internas no cubren entrenamiento externo')
        for tr, val in inner:
            if set(tr) & set(val) or set(tr) | set(val) != set(train):
                raise ValueError('Partición interna inválida')
        for name, configs in CANDIDATES.items():
            scores = []
            for candidate, config in enumerate(configs, 1):
                ms = []
                for i, (tr, val) in enumerate(inner, 1):
                    seed = 42 + outer * 100 + i - 1
                    started = time.perf_counter()
                    pred, probs = fit_predict(name, config, seed, X[tr], y[tr], X[val], k)
                    duration = time.perf_counter() - started
                    m = metrics(y[val], pred, probs, k)
                    ms.append(m)
                    fit_records.append(dict(modelo=name, fold=outer, fase='inner', candidata=candidate,
                        inner_fold=i, config=config, seed=seed, status='completado', duration_seconds=duration, train_indices=tr.tolist(), eval_indices=val.tolist(),
                        support_train=np.bincount(y[tr], minlength=k).tolist(), support_eval=np.bincount(y[val], minlength=k).tolist(), metricas=m))
                score = CandidateScore(config, ModelConfig(), TrainingConfig(),
                    mae_mean=float(np.mean([m['mae_ordinal'] for m in ms])),
                    qwk_mean=float(np.mean([m['qwk'] for m in ms])))
                scores.append(score)
                candidates.append(dict(modelo=name, fold=outer, candidata=candidate, config=config,
                    mae_mean=score.mae_mean, qwk_mean=score.qwk_mean))
            winner = min(range(len(scores)), key=lambda i: scores[i].selection_key())
            for record in candidates[-len(configs):]:
                record['seleccionada'] = record['candidata'] == winner + 1
            seed = 42 + outer * 1000
            started = time.perf_counter()
            pred, probs = fit_predict(name, configs[winner], seed, X[train], y[train], X[test], k)
            duration = time.perf_counter() - started
            m = metrics(y[test], pred, probs, k)
            fit_records.append(dict(modelo=name, fold=outer, fase='refit', candidata=winner + 1,
                inner_fold=None, config=configs[winner], seed=seed, status='completado', duration_seconds=duration, train_indices=train.tolist(), eval_indices=test.tolist(),
                support_train=np.bincount(y[train], minlength=k).tolist(), support_eval=np.bincount(y[test], minlength=k).tolist(), metricas=m))
            folds.append(dict(modelo=name, fold=outer, **m))
            oof.extend(dict(modelo=name, fold=outer, row_id=int(idx), y_true=int(yt), y_pred=int(yp), probabilities=p.tolist())
                for idx, yt, yp, p in zip(test, y[test], pred, probs))
        for strategy in ('mode', 'median'):
            name = 'constante_moda' if strategy == 'mode' else 'constante_mediana'
            model = ConstantPredictor(strategy=strategy, classes=np.arange(k)).fit(y[train])
            pred, probs = model.predict(X[test]), model.predict_proba(X[test])
            folds.append(dict(modelo=name, fold=outer, **metrics(y[test], pred, probs, k)))
            oof.extend(dict(modelo=name, fold=outer, row_id=int(idx), y_true=int(yt), y_pred=int(yp), probabilities=p.tolist())
                for idx, yt, yp, p in zip(test, y[test], pred, probs))
        print(f'Fold {outer}/5 terminado', flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'fits.json').write_text(json.dumps(fit_records, ensure_ascii=False, indent=2))
    (output_dir / 'candidatas.json').write_text(json.dumps(candidates, ensure_ascii=False, indent=2))
    pd.DataFrame(folds).to_csv(output_dir / 'metricas_folds.csv', index=False)
    (output_dir / 'predicciones_oof.json').write_text(json.dumps(oof, ensure_ascii=False))
    summary = {}
    for name in [*CANDIDATES, 'constante_moda', 'constante_mediana']:
        rows = sorted((r for r in oof if r['modelo'] == name), key=lambda r: r['row_id'])
        if [r['row_id'] for r in rows] != list(range(len(y))):
            raise ValueError(f'Cobertura OOF inválida: {name}')
        fs = [r for r in folds if r['modelo'] == name]
        summary[name] = {'resumen_folds_externos': {key: {'media': float(np.mean([f[key] for f in fs])),
            'std_ddof_0': float(np.std([f[key] for f in fs]))} for key in [*METRIC_KEYS, 'nll', 'rps']},
            'agregado_oof': metrics(np.array([r['y_true'] for r in rows]), np.array([r['y_pred'] for r in rows]),
                np.array([r['probabilities'] for r in rows]), k)}
    (output_dir / 'resumen.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_and_regenerate(output_dir, central_dir, summary_dir, target_name='GDS_R2'):
    """Valida los registros y reconstruye métricas, sin ajustar modelos."""
    output_dir, central_dir, summary_dir = map(Path, (output_dir, central_dir, summary_dir))
    manifest = json.loads((output_dir / 'manifest.json').read_text())
    split_path = central_dir / target_name / 'splits.npz'
    if manifest['status'] != 'completado' or file_hash(split_path) != manifest['split_sha256']:
        raise ValueError('Manifiesto incompleto o particiones modificadas')
    with np.load(split_path, allow_pickle=False) as saved:
        splits = {key: saved[key].copy() for key in saved.files}
    y, k = splits['labels'], len(splits['classes'])
    fits = json.loads((output_dir / 'fits.json').read_text())
    candidates = json.loads((output_dir / 'candidatas.json').read_text())
    rows = json.loads((output_dir / 'predicciones_oof.json').read_text())
    if len(fits) != 260 or len(candidates) != 80:
        raise ValueError('Cantidad de ajustes o candidatas inválida')
    def require(condition, message):
        if not condition:
            raise ValueError(message)
    for name, configs in CANDIDATES.items():
        for fold in range(1, 6):
            group = [c for c in candidates if c['modelo'] == name and c['fold'] == fold]
            require(len(group) == 4 and {c['candidata'] for c in group} == set(range(1, 5)), 'Candidatas incompletas')
            for candidate in group:
                number = candidate['candidata']
                require(candidate['config'] == configs[number - 1], 'Configuración inválida')
                inner = [r for r in fits if r['modelo'] == name and r['fold'] == fold and r['fase'] == 'inner' and r['candidata'] == number]
                require(len(inner) == 3 and {r['inner_fold'] for r in inner} == {1, 2, 3}, 'Folds internos incompletos')
                for record in inner:
                    i = record['inner_fold']
                    prefix = f'outer_{fold}_inner_{i}'
                    train, val = splits[prefix + '_train'], splits[prefix + '_val']
                    require(record['train_indices'] == train.tolist() and record['eval_indices'] == val.tolist()
                        and record['seed'] == 42 + 100 * fold + i - 1 and record['config'] == candidate['config']
                        and record['status'] == 'completado' and record['duration_seconds'] >= 0
                        and record['support_train'] == np.bincount(y[train], minlength=k).tolist()
                        and record['support_eval'] == np.bincount(y[val], minlength=k).tolist(), 'Registro interno inválido')
                require(np.isclose(candidate['mae_mean'], np.mean([r['metricas']['mae_ordinal'] for r in inner]), equal_nan=True)
                    and np.isclose(candidate['qwk_mean'], np.mean([r['metricas']['qwk'] for r in inner]), equal_nan=True), 'Promedio interno inválido')
            def key(c):
                return CandidateScore({}, ModelConfig(), TrainingConfig(), mae_mean=c['mae_mean'], qwk_mean=c['qwk_mean']).selection_key()
            best = min(group, key=key)
            require(best['seleccionada'] and sum(c['seleccionada'] for c in group) == 1, 'Selección inválida')
            refits = [r for r in fits if r['modelo'] == name and r['fold'] == fold and r['fase'] == 'refit']
            require(len(refits) == 1, 'Refit ausente o duplicado')
            r = refits[0]
            train, test = splits[f'outer_{fold}_train'], splits[f'outer_{fold}_test']
            require(r['train_indices'] == train.tolist() and r['eval_indices'] == test.tolist()
                and r['seed'] == 42 + 1000 * fold and r['candidata'] == best['candidata']
                and r['config'] == best['config'] and r['status'] == 'completado' and r['duration_seconds'] >= 0
                and r['support_train'] == np.bincount(y[train], minlength=k).tolist()
                and r['support_eval'] == np.bincount(y[test], minlength=k).tolist(), 'Registro refit inválido')
    names = [*CANDIDATES, 'constante_moda', 'constante_mediana']
    require(len(rows) == len(names) * len(y), 'Cantidad OOF inválida')
    summary, folds = {}, []
    for name in names:
        group = sorted((r for r in rows if r['modelo'] == name), key=lambda r: r['row_id'])
        require([r['row_id'] for r in group] == list(range(len(y))), 'Cobertura OOF inválida')
        require(all(r['y_true'] == int(y[r['row_id']]) for r in group), 'Etiquetas OOF inválidas')
        fold_metrics = []
        for fold in range(1, 6):
            subset = [r for r in group if r['fold'] == fold]
            require(sorted(r['row_id'] for r in subset) == sorted(splits[f'outer_{fold}_test'].tolist()), 'Fold OOF inválido')
            m = metrics(np.array([r['y_true'] for r in subset]), np.array([r['y_pred'] for r in subset]),
                        np.array([r['probabilities'] for r in subset]), k)
            if name in CANDIDATES:
                stored = next(r for r in fits if r['modelo'] == name and r['fold'] == fold and r['fase'] == 'refit')['metricas']
                require(all(np.isclose(m[key], stored[key], rtol=1e-6, atol=1e-7, equal_nan=True) for key in m), 'Métricas refit inválidas')
            folds.append(dict(modelo=name, fold=fold, **m))
            fold_metrics.append(m)
        summary[name] = {'resumen_folds_externos': {key: {'media': float(np.mean([m[key] for m in fold_metrics])),
            'std_ddof_0': float(np.std([m[key] for m in fold_metrics]))} for key in [*METRIC_KEYS, 'nll', 'rps']},
            'agregado_oof': metrics(np.array([r['y_true'] for r in group]), np.array([r['y_pred'] for r in group]),
                np.array([r['probabilities'] for r in group]), k)}
    (output_dir / 'resumen.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    pd.DataFrame(folds).to_csv(output_dir / 'metricas_folds.csv', index=False)
    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / 'resumen.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    lines = ['# Controles en GDS_R2', '', 'Mismos datos y particiones 5×3 de la corrida central sin ponderación; selección por MAE interno y desempate QWK. Los controles se reentrenan con todo el entrenamiento externo. Media ± desviación de cinco folds (ddof=0); OOF se calcula por separado.', '', '| Modelo | MAE media ± DE | QWK media ± DE | MAE OOF | QWK OOF |', '|---|---:|---:|---:|---:|']
    for name in [*summary, 'softmax', 'coral']:
        report = summary[name] if name in summary else json.loads((central_dir / target_name / name / 'reporte.json').read_text())
        f, o = report['resumen_folds_externos'], report['agregado_oof']
        lines.append(f"| {name} | {f['mae_ordinal']['media']:.4f} ± {f['mae_ordinal']['std_ddof_0']:.4f} | {f['qwk']['media']:.4f} ± {f['qwk']['std_ddof_0']:.4f} | {o['mae_ordinal']:.4f} | {o['qwk']:.4f} |")
    lines += ['', 'La comparación es descriptiva para esta partición; los modelos y sus configuraciones difieren. NLL, RPS y las demás métricas están en el resumen JSON.']
    (summary_dir / 'comparacion.md').write_text('\n'.join(lines) + '\n')
    return summary


def ejecutar_evaluacion_controles(sav_path, target_name='GDS_R2', output_dir=Path('results/controles_validacion'), central_dir=CENTRAL, summary_dir=Path('results/resumen/controles')):
    if target_name != 'GDS_R2':
        raise ValueError('Esta ejecución requiere las particiones centrales de GDS_R2')
    central_dir, output_dir = Path(central_dir), Path(output_dir)
    central_manifest = json.loads((central_dir / 'manifest.json').read_text())
    digest = file_hash(sav_path)
    if digest != central_manifest['identity']['data_sha256']:
        raise ValueError('El SHA-256 del SAV no coincide con la corrida central')
    split_path = central_dir / target_name / 'splits.npz'
    loader = DataFrameLoader()
    data = prepare_experiment_data(loader.load(sav_path), target_name, loader)
    with np.load(split_path, allow_pickle=False) as saved:
        splits = {key: saved[key].copy() for key in saved.files}
    if not np.array_equal(data.y, splits['labels']) or not np.array_equal(data.classes, splits['classes']):
        raise ValueError('Etiquetas o clases distintas a la corrida central')
    root = Path(__file__).resolve().parents[1]
    sources = ['experimentos/ejecutar_controles.py', 'experimentos/controles.py', 'src/config.py', 'src/utils.py',
               'src/data/loader.py', 'src/data/preprocessing.py', 'src/evaluation/metrics.py', 'src/training/search.py']
    manifest = {'status': 'en_curso', 'data_sha256': digest, 'split_sha256': file_hash(split_path),
        'sources_sha256': {name: file_hash(root / name) for name in sources},
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__, 'torch': torch.__version__},
        'config': {'target': target_name, 'seed': 42, 'outer_folds': 5, 'inner_folds': 3, 'candidates': CANDIDATES,
            'device': 'cpu', 'threads': 1, 'mlp': {'epochs': 20, 'batch_size': 32, 'optimizer': 'Adam', 'learning_rate': 0.001,
                'weight_decay': 0.0001, 'loss': 'CrossEntropyLoss'},
            'hist_gradient_boosting': {'max_iter': 100, 'learning_rate': 0.1, 'l2_regularization': 1.0, 'early_stopping': False},
            'logistic': {'penalty': 'l2', 'solver': 'lbfgs', 'max_iter': 2000}}}
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / 'manifest.json'
    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    save_manifest()
    try:
        torch.set_num_threads(1)
        evaluate(data.X, data.y, splits, data.classes, output_dir)
        manifest['status'] = 'completado'
        save_manifest()
        audit_and_regenerate(output_dir, central_dir, summary_dir, target_name)
    except Exception as exc:
        manifest['status'] = 'fallido'
        manifest['error'] = str(exc)
        save_manifest()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', type=Path)
    parser.add_argument('--target-name', default='GDS_R2')
    parser.add_argument('--output-dir', type=Path, default=Path('results/controles_validacion'))
    parser.add_argument('--central-dir', type=Path, default=CENTRAL)
    parser.add_argument('--summary-dir', type=Path, default=Path('results/resumen/controles'))
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    if args.report_only:
        audit_and_regenerate(args.output_dir, args.central_dir, args.summary_dir, args.target_name)
    else:
        if args.data_path is None:
            parser.error('--data-path es obligatorio salvo con --report-only')
        ejecutar_evaluacion_controles(args.data_path, args.target_name, args.output_dir, args.central_dir, args.summary_dir)


if __name__ == '__main__':
    main()
