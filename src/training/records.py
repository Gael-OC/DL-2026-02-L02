"""Manifiesto y archivos recuperables de una corrida de validacion."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import sklearn
import torch

from src.config import ModelConfig, TrainingConfig
from src.evaluation.metrics import METRIC_KEYS
from src.training.search import CandidateScore


def _hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_manifest(config, device):
    root = Path(__file__).resolve().parents[2]
    files = [root / 'main.py', root / 'experimentos' / 'reportes.py', *sorted((root / 'src').rglob('*.py'))]
    code = hashlib.sha256()
    for path in files:
        code.update(str(path.relative_to(root)).encode())
        code.update(path.read_bytes())
    settings = asdict(config)
    settings['data_path'] = str(config.data_path.resolve())
    settings['output_dir'] = str(config.output_dir.resolve())
    return {
        'config': settings, 'data_sha256': _hash_file(config.data_path),
        'code_sha256': code.hexdigest(), 'python': sys.version,
        'platform': platform.platform(), 'numpy': np.__version__,
        'sklearn': sklearn.__version__, 'torch': torch.__version__, 'device': str(device),
        'effective_split_seed': config.seed if config.split_seed is None else config.split_seed,
        'effective_init_seed': config.seed if config.init_seed is None else config.init_seed,
    }


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


class RunRecords:
    def __init__(self, folder, identity):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.path = self.folder / 'manifest.json'
        # Normaliza tuplas a listas antes de comparar con JSON.
        self.identity = json.loads(json.dumps(identity))
        if self.path.exists():
            current = json.loads(self.path.read_text(encoding='utf-8'))
            if current['identity'] != self.identity:
                raise ValueError('El manifiesto no coincide; use otra carpeta para la corrida.')
        elif any(self.folder.iterdir()):
            raise ValueError('La carpeta no esta vacia y no contiene manifiesto; use otra carpeta.')
        self.status('en_curso')

    def status(self, state, error=None):
        atomic_json(self.path, {'identity': self.identity, 'status': state, 'error': error})

    def event(self, record):
        with (self.folder / 'ajustes.jsonl').open('a', encoding='utf-8') as out:
            out.write(json.dumps(record, ensure_ascii=False) + '\n')

    def fold_path(self, target, variant, fold):
        path = self.folder / target / variant / f'fold_{fold}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save_fold(self, target, variant, fold, result, candidates):
        atomic_json(self.fold_path(target, variant, fold), {
            'outer_fold': fold, 'selected': asdict(result.selected),
            'inner_skipped': result.inner_skipped, 'metrics': result.metrics,
            'y_true': result.y_true.tolist(), 'y_pred': result.y_pred.tolist(),
            'final_train_loss': result.final_train_loss, 'candidates': candidates,
            'row_ids': result.row_ids.tolist(), 'logits': result.logits.tolist(),
            'probabilities': result.probabilities.tolist(), 'probability_dtype': str(result.probabilities.dtype),
        })
        self.write_tables()

    def load_fold(self, target, variant, fold):
        path = self.fold_path(target, variant, fold)
        if not path.exists():
            return None
        from src.training.nested_cv import OuterFoldResult
        saved = json.loads(path.read_text(encoding='utf-8'))
        candidate = saved['selected']
        candidate['model_config']['conv_channels'] = tuple(candidate['model_config']['conv_channels'])
        candidate['model_config'] = ModelConfig(**candidate['model_config'])
        candidate['training_config'] = TrainingConfig(**candidate['training_config'])
        return OuterFoldResult(fold, CandidateScore(**candidate), saved['inner_skipped'], saved['metrics'],
                               np.asarray(saved['y_true']), np.asarray(saved['y_pred']), saved['final_train_loss'],
                               np.asarray(saved['row_ids']), np.asarray(saved['logits']), np.asarray(saved['probabilities']))

    def write_tables(self):
        rows, metrics = [], []
        for path in sorted(self.folder.glob('*/*/fold_*.json')):
            data = json.loads(path.read_text(encoding='utf-8'))
            target, variant = path.parent.parent.name, path.parent.name
            metrics.append({'objetivo': target, 'variante': variant, 'repeticion': 1, 'fold': data['outer_fold'], **data['metrics']})
            rows.extend({'objetivo': target, 'variante': variant, 'fold': data['outer_fold'], **c}
                        for c in data['candidates'])
        for name, values, fields in (
            ('metricas_folds.csv', metrics, ['objetivo', 'variante', 'repeticion', 'fold', *METRIC_KEYS, 'nll', 'rps']),
            ('candidatas.csv', rows, ['objetivo', 'variante', 'fold', 'candidata', 'configuracion', 'resultados_internos', 'mae_mean', 'mae_std', 'qwk_mean', 'qwk_std', 'seleccionada']),
        ):
            path = self.folder / name
            temp = path.with_suffix('.csv.tmp')
            with temp.open('w', encoding='utf-8', newline='') as out:
                writer = csv.DictWriter(out, fieldnames=fields)
                writer.writeheader()
                writer.writerows(values)
            temp.replace(path)


def recorded_fit(records, trainer, X_train, y_train, X_eval, y_eval, seed, info):
    import time
    from src.tasks.ordinal_task import effective_number_weights

    k = trainer.task.num_classes
    config = trainer.config
    weights = None
    if trainer.task.name == 'coral' and config.use_class_weights:
        weights = effective_number_weights(y_train, k, config.beta).tolist()
    record = {**info, 'fit_id': '/'.join(str(info.get(key, '')) for key in ('target', 'variant', 'outer_fold', 'phase', 'candidate', 'inner_fold')), 'seed': seed, 'support_train': np.bincount(y_train, minlength=k).tolist(),
              'support_eval': np.bincount(y_eval, minlength=k).tolist(), 'class_weights': weights,
              'epochs': config.epochs}
    start = time.perf_counter()
    try:
        result = trainer.fit_and_evaluate(X_train, y_train, X_eval, y_eval, seed)
    except BaseException as error:
        records.event({**record, 'status': 'fallido', 'seconds': time.perf_counter() - start,
                       'error': f'{type(error).__name__}: {error}'})
        raise
    records.event({**record, 'status': 'terminado', 'seconds': time.perf_counter() - start,
                   'train_losses': result.train_losses, 'metrics': result.metrics})
    return result
