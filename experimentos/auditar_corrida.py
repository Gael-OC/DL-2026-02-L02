"""Audita una corrida sin modificarla: python -m experimentos.auditar_corrida CARPETA --eventos N."""
import argparse
import csv
import math
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from src.tasks.softmax_task import SoftmaxTask
from src.tasks.ordinal_task import CoralTask
from src.training.search import CandidateScore
from src.evaluation.metrics import METRIC_KEYS, MetricsCalculator
from src.tasks.ordinal_task import effective_number_weights
from src.config import HYPERPARAMETER_GRID, ModelConfig, TrainingConfig, apply_overrides

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition: raise ValueError(message)


def auditar(folder, eventos, data_path=None):
    folder = Path(folder)
    manifest = json.loads((folder / 'manifest.json').read_text())
    identity = manifest['identity']; config = identity['config']
    require(manifest['status'] == 'terminado', 'Manifiesto no terminado')
    require(config['model'] == {'conv_channels':[16,32], 'kernel_size':3, 'hidden_dim':32, 'dropout':0.15}, 'Protocolo modelo')
    require(config['training'] == {'learning_rate':0.001, 'weight_decay':0.0001, 'batch_size':32, 'epochs':20, 'use_class_weights':config['training']['use_class_weights'], 'beta':0.99}, 'Protocolo entrenamiento')
    require(config['validation'] == {'outer_folds':5, 'inner_folds':3, 'gds_outer_folds':2, 'gds_inner_folds':2, 'use_grid':True}, 'Protocolo validación')
    split_seed = config['seed'] if config['split_seed'] is None else config['split_seed']
    init_seed = config['seed'] if config['init_seed'] is None else config['init_seed']
    require(split_seed in (42,314,2718) and init_seed == 42 and identity['effective_split_seed'] == split_seed and identity['effective_init_seed'] == init_seed, 'Semillas manifiesto')
    variants = {('softmax' if a == 'conv1d_softmax' else 'coral_weighted' if config['training']['use_class_weights'] else 'coral') for a in config['algorithms']}
    require(set(config['algorithms']) <= {'conv1d_softmax','conv1d_coral'} and len(variants) == len(config['algorithms']), 'Algoritmos manifiesto')
    require({p.parent.name for p in folder.glob('*/splits.npz')} == set(config['targets']), 'Objetivos manifiesto')
    expected_ids = set()
    for target in config['targets']:
        for variant in variants:
            require({p.name for p in (folder / target).iterdir() if p.is_dir()} == variants, 'Variantes manifiesto')
            for outer in range(1, (2 if target == 'GDS' and config['all_targets'] else 5) + 1):
                expected_ids.add((target,variant,outer,'refit',None,None))
                if not (target == 'GDS' and config['all_targets']):
                    expected_ids.update((target,variant,outer,'inner',c,i) for c in range(1,5) for i in range(1,4))
    require(hashlib.sha256(Path(data_path if data_path is not None else config['data_path']).read_bytes()).hexdigest() == identity['data_sha256'], 'Hash datos')
    digest = hashlib.sha256()
    for path in [ROOT / 'main.py', ROOT / 'experimentos/reportes.py', *sorted((ROOT / 'src').rglob('*.py'))]:
        digest.update(str(path.relative_to(ROOT)).encode()); digest.update(path.read_bytes())
    require(digest.hexdigest() == identity['code_sha256'], 'Hash código')
    records = [json.loads(x) for x in (folder / 'ajustes.jsonl').read_text().splitlines()]
    require(len(records) == eventos and len({r['fit_id'] for r in records}) == eventos, 'Número o unicidad de eventos')
    actual_ids = [(r['target'],r['variant'],r['outer_fold'],r['phase'],r['candidate'],r.get('inner_fold')) for r in records]
    require(set(actual_ids) == expected_ids and len(actual_ids) == len(expected_ids), 'Claves de eventos')
    for r in records:
        expected_seed = init_seed + r['outer_fold'] * (1000 if r['phase'] == 'refit' else 100) + (0 if r['phase'] == 'refit' else r['inner_fold'] - 1)
        require(r['seed'] == expected_seed, 'Semilla ajuste')
        require(r['status'] == 'terminado' and r['epochs'] == config['training']['epochs'] and len(r['train_losses']) == r['epochs'], f"Evento incompleto: {r['fit_id']}")
        s = np.load(folder / r['target'] / 'splits.npz'); labels = s['labels']; k = len(s['classes']); prefix = f"outer_{r['outer_fold']}"
        if r['phase'] == 'refit': train, val = s[prefix+'_train'], s[prefix+'_test']
        else:
            prefix += f"_inner_{r['inner_fold']}"; train, val = s[prefix+'_train'], s[prefix+'_val']
        require(np.array_equal(np.bincount(labels[train], minlength=k), r['support_train']) and np.array_equal(np.bincount(labels[val], minlength=k), r['support_eval']), f"Soportes: {r['fit_id']}")
        expected = effective_number_weights(labels[train], k, config['training']['beta']) if r['variant'] == 'coral_weighted' else None
        require((r['class_weights'] is None if expected is None else np.allclose(expected, r['class_weights'])), f"Pesos: {r['fit_id']}")
        s.close()
    for split in folder.glob('*/splits.npz'):
        with np.load(split) as s:
            labels = s['labels']; universe = set(range(len(labels))); tests = []
            require(int(s['split_seed']) == split_seed and str(s['target']) == split.parent.name, 'Identidad partición')
            for name in s.files:
                if name.endswith(('_train','_test','_val')):
                    indices = s[name]
                    require(indices.ndim == 1 and np.issubdtype(indices.dtype, np.integer) and len(np.unique(indices)) == len(indices) and np.all((indices >= 0) & (indices < len(labels))), f'Índices inválidos {name}')
            outer = sorted(k for k in s.files if k.startswith('outer_') and k.endswith('_test'))
            skipped = split.parent.name == 'GDS' and config['all_targets']
            require(len(outer) == (2 if skipped else 5), 'Cantidad folds')
            for key in outer:
                prefix = key[:-5]; require(bool(s[prefix+'_inner_skipped']) == skipped, 'Búsqueda omitida')
                require(len([k for k in s.files if k.startswith(prefix+'_inner_') and k.endswith('_train')]) == (0 if skipped else 3), 'Cantidad folds internos')
                train, test = set(s[prefix+'_train']), set(s[key]); tests.extend(s[key].tolist())
                require(not train & test and train | test == universe, f'Partición {key}')
                for inner in [k for k in s.files if k.startswith(prefix+'_inner_') and k.endswith('_train')]:
                    a, b = set(s[inner]), set(s[inner[:-6]+'_val'])
                    require(not a & b and a | b == train, f'Partición {inner}')
            require(sorted(tests) == list(range(len(labels))), 'Cobertura externa')
            for variant in (p for p in split.parent.iterdir() if p.is_dir()):
                all_rows = {}; fold_metrics = []
                for key in outer:
                    fold = json.loads((variant / ('fold_'+key.split('_')[1]+'.json')).read_text())
                    idx = s[key]; true = np.asarray(fold['y_true']); pred = np.asarray(fold['y_pred']); probs = np.asarray(fold['probabilities'], dtype=fold['probability_dtype'])
                    require(np.array_equal(fold['row_ids'], idx) and np.array_equal(true, labels[idx]), f'Alineación {variant}/{key}')
                    require(probs.shape == (len(idx), len(s['classes'])) and np.all(np.isfinite(probs)) and np.all(probs >= 0) and np.allclose(probs.sum(axis=1), 1), f'Probabilidades {variant}/{key}')
                    logits = torch.tensor(fold['logits'], dtype=torch.float32)
                    task = SoftmaxTask(len(s['classes'])) if variant.name == 'softmax' else CoralTask(len(s['classes']))
                    require(np.allclose(task.probabilities(logits).numpy(), probs, rtol=1e-5, atol=1e-7) and np.array_equal(task.predict(logits).numpy(), pred), 'Logits, probabilidades o decisión')
                    require(fold['inner_skipped'] == skipped, 'Búsqueda fold')
                    candidates = fold['candidates']
                    if skipped:
                        require(len(candidates) == 1 and candidates[0]['candidata'] == 0 and candidates[0]['seleccionada'] and json.loads(candidates[0]['configuracion']) == {'model': config['model'], 'training': config['training']} and fold['selected']['model_config'] == config['model'] and fold['selected']['training_config'] == config['training'], 'Configuración base GDS')
                    if not fold['inner_skipped']:
                        algorithm = 'conv1d_softmax' if variant.name == 'softmax' else 'conv1d_coral'
                        grid = HYPERPARAMETER_GRID[algorithm]
                        require(len(candidates) == len(grid), f'Grid {variant}/{key}')
                        for c, override in zip(candidates, grid):
                            model, training = apply_overrides(ModelConfig(**config['model']), TrainingConfig(**config['training']), override)
                            require(json.loads(c['configuracion']) == {'model': asdict(model), 'training': asdict(training)} or json.loads(c['configuracion']) == json.loads(json.dumps({'model': asdict(model), 'training': asdict(training)})), 'Configuración grid')
                            internal = json.loads(c['resultados_internos'])
                            events = sorted((r for r in records if r['target'] == split.parent.name and r['variant'] == variant.name and r['outer_fold'] == fold['outer_fold'] and r['phase'] == 'inner' and r['candidate'] == c['candidata']), key=lambda r:r['inner_fold'])
                            require(len(internal) == len(events) and all(a == b['metrics'] for a,b in zip(internal, events)), 'Resultados internos')
                            for name, metric in [('mae', 'mae_ordinal'), ('qwk', 'qwk')]:
                                values = [x[metric] for x in internal]
                                require(np.isclose(c[name+'_mean'], np.mean(values), equal_nan=True) and np.isclose(c[name+'_std'], np.std(values), equal_nan=True), 'Promedio candidata')
                        def rank(c):
                            return CandidateScore({}, ModelConfig(), TrainingConfig(), mae_mean=c['mae_mean'], qwk_mean=c['qwk_mean']).selection_key()
                        selected = [c for c in candidates if c['seleccionada']]
                        require(len(selected) == 1 and selected[0] is min(candidates, key=rank), 'Selección candidata')
                        require(json.loads(selected[0]['configuracion']) == {'model': fold['selected']['model_config'], 'training': fold['selected']['training_config']}, 'Configuración seleccionada')
                    metrics = MetricsCalculator.compute(true, pred, len(s['classes']))
                    require(all(np.isclose(metrics[m], fold['metrics'][m], equal_nan=True) for m in METRIC_KEYS), f'Métricas {variant}/{key}')
                    metrics.update(nll=MetricsCalculator.nll(true, probs), rps=MetricsCalculator.rps(true, probs))
                    with (folder / 'metricas_folds.csv').open() as f:
                        matching = [r for r in csv.DictReader(f) if r['objetivo'] == split.parent.name and r['variante'] == variant.name and int(r['fold']) == fold['outer_fold']]
                    require(len(matching) == 1 and all(np.isclose(float(matching[0][m]), value, equal_nan=True) for m, value in metrics.items()), 'Métricas CSV fold')
                    fold_metrics.append(metrics)
                    for i, row_id in enumerate(idx):
                        all_rows[int(row_id)] = (int(true[i]), int(pred[i]), probs[i], int(fold['outer_fold']), logits[i].numpy())
                with (variant / 'predicciones_oof.csv').open() as f:
                    oof = list(csv.DictReader(f))
                require(len(oof) == len(labels) and [int(r['row_id']) for r in oof] == list(range(len(labels))), 'Cobertura OOF')
                for r in oof:
                    true, pred, probs, fold_number, logits = all_rows[int(r['row_id'])]
                    require(int(r['verdad_indice']) == true and int(r['decision_indice']) == pred and float(r['verdad_original']) == s['classes'][true] and float(r['decision_original']) == s['classes'][pred] and int(r['fold']) == fold_number and int(r['repeticion']) == 1 and np.allclose(json.loads(r['probabilidades']), probs) and np.allclose(json.loads(r['logits']), logits), 'Alineación OOF')
                report = json.loads((variant / 'reporte.json').read_text())
                for metric in [*METRIC_KEYS, 'nll', 'rps']:
                    values = np.asarray([m[metric] for m in fold_metrics]); values = values[np.isfinite(values)]
                    summary = report['resumen_folds_externos'][metric]
                    require(summary['folds_validos'] == len(values) and summary['folds_total'] == len(fold_metrics), 'Conteo métricas')
                    if len(values):
                        require(np.isclose(summary['media'], values.mean()) and np.isclose(summary['std_ddof_0'], values.std()), 'Resumen métricas')
                true = np.array([all_rows[i][0] for i in range(len(labels))]); pred = np.array([all_rows[i][1] for i in range(len(labels))]); probs = np.array([all_rows[i][2] for i in range(len(labels))])
                aggregate = MetricsCalculator.compute(true, pred, len(s['classes']))
                aggregate.update(nll=MetricsCalculator.nll(true, probs), rps=MetricsCalculator.rps(true, probs))
                require(all(np.isclose(value, report['agregado_oof'][name], equal_nan=True) for name, value in aggregate.items()), 'Métricas OOF')
                with (variant / 'confusion_oof.csv').open() as f: matrix = list(csv.DictReader(f))
                require(np.array_equal([[int(row[str(c)]) for c in s['classes']] for row in matrix], MetricsCalculator.confusion_matrix(true, pred, len(s['classes']))), 'Confusión OOF')
    return len(records)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('folder'); p.add_argument('--eventos', type=int, required=True)
    p.add_argument('--data-path', help='SAV alternativo con el mismo SHA-256')
    a = p.parse_args(); print(f'Auditoría aprobada: {auditar(a.folder, a.eventos, a.data_path)} eventos')
