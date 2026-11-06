"""Predictor por perfil con suavizado aditivo y evaluación externa local."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from experimentos.decisiones import decisions, fold_summary, summarize
from src.config import FEATURE_COLUMNS, TARGET_COLUMNS
from src.data.loader import DataFrameLoader
from src.data.preprocessing import prepare_experiment_data


class ProfilePredictor:
    """Frecuencias de etiquetas por las 15 respuestas; alpha=1 sobre K clases.

    fit recibe exclusivamente entrenamiento. Un perfil nuevo usa las
    frecuencias globales de ese entrenamiento, también suavizadas.
    """

    def __init__(self, num_classes):
        if not isinstance(num_classes, (int, np.integer)) or num_classes < 2:
            raise ValueError('num_classes debe ser un entero >= 2.')
        self.num_classes = int(num_classes)
        self.counts_ = None

    @staticmethod
    def _features(X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != len(FEATURE_COLUMNS) or not np.isin(X, [0, 1]).all():
            raise ValueError('X debe tener forma (N,15) y respuestas binarias.')
        return X

    def fit(self, X, y):
        X = self._features(X)
        y = np.asarray(y)
        if (y.ndim != 1 or not len(y) or len(y) != len(X)
                or not np.issubdtype(y.dtype, np.integer)
                or np.any(y < 0) or np.any(y >= self.num_classes)):
            raise ValueError('y debe contener índices enteros 0..K-1, uno por fila de entrenamiento.')
        counts = {}
        for row, label in zip(X, y):
            counts.setdefault(tuple(row), np.zeros(self.num_classes, dtype=np.int64))[label] += 1
        self.counts_ = counts
        self.global_counts_ = np.bincount(y, minlength=self.num_classes)
        return self

    def seen(self, X):
        if self.counts_ is None:
            raise RuntimeError('Se requiere fit antes de predecir.')
        return np.array([tuple(row) in self.counts_ for row in self._features(X)], dtype=bool)

    def predict_proba(self, X):
        if self.counts_ is None:
            raise RuntimeError('Se requiere fit antes de predecir.')
        counts = np.array([self.counts_.get(tuple(row), self.global_counts_)
                           for row in self._features(X)], dtype=np.float64).reshape(-1, self.num_classes)
        return (counts + 1) / (counts.sum(axis=1, keepdims=True) + self.num_classes)


def predict_fold(data, train, test, fold):
    """Ajusta con train; las etiquetas de test solo se adjuntan a la salida."""
    model = ProfilePredictor(data.num_classes).fit(data.X[train], data.y[train])
    probs = model.predict_proba(data.X[test])
    mode, median = decisions(probs)
    return pd.DataFrame({
        'objetivo': data.target_name, 'modelo': 'perfil', 'fold': fold,
        'row_id': test, 'verdad': data.y[test], 'k': data.num_classes,
        'visto': model.seen(data.X[test]), 'probs': list(probs),
        'moda': mode, 'mediana': median,
    })


def validated_splits(path, data):
    """Verifica identidad, índices, separación train/test y cobertura externa."""
    expected = 2 if data.target_name == 'GDS' else 5
    with np.load(path, allow_pickle=False) as saved:
        required = {'labels', 'classes', 'target', 'split_seed'}
        required.update(f'outer_{fold}_{part}' for fold in range(1, expected + 1) for part in ('train', 'test'))
        test_keys = {key for key in saved.files if key.startswith('outer_') and key.endswith('_test')}
        if (not required <= set(saved.files)
                or test_keys != {f'outer_{fold}_test' for fold in range(1, expected + 1)}):
            raise ValueError(f'Folds externos incompletos o inesperados: {path}')
        if (str(saved['target']) != data.target_name or int(saved['split_seed']) != 42
                or not np.array_equal(saved['labels'], data.y)
                or not np.array_equal(saved['classes'], data.classes)):
            raise ValueError(f'Etiquetas, clases o identidad de particiones incompatibles: {path}')
        pairs, covered = [], []
        universe = np.arange(len(data.y))
        for fold in range(1, expected + 1):
            train, test = (saved[f'outer_{fold}_{part}'] for part in ('train', 'test'))
            for indices in (train, test):
                if (indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer)
                        or np.any(indices < 0) or np.any(indices >= len(data.y))
                        or len(np.unique(indices)) != len(indices)):
                    raise ValueError(f'Índices inválidos: {path}, fold {fold}')
            if not np.array_equal(np.sort(np.concatenate((train, test))), universe):
                raise ValueError(f'Train/test deben ser disjuntos y cubrir todas las filas: {path}, fold {fold}')
            pairs.append((train.copy(), test.copy()))
            covered.extend(test.tolist())
        if sorted(covered) != universe.tolist():
            raise ValueError(f'Cobertura externa incorrecta: {path}')
    return pairs


def compare_reference(rows, reference):
    """Compara por objetivo/row_id, sin depender del orden de las filas del CSV."""
    old = pd.read_csv(reference)
    columns = list(rows.columns)
    if not set(columns) <= set(old.columns):
        raise ValueError('La referencia no contiene las columnas requeridas.')
    old = old.loc[old.modelo == 'perfil', columns]
    keys = ['objetivo', 'row_id']
    if old.duplicated(keys).any() or rows.duplicated(keys).any():
        raise ValueError('Referencia o predicciones con filas duplicadas.')
    old = old.set_index(keys).sort_index()
    new = rows.set_index(keys).sort_index()
    if not old.index.equals(new.index):
        raise ValueError('Cobertura de la referencia incompatible.')
    for column in new.columns:
        if column == 'probs':
            equal = all(np.array_equal(np.asarray(json.loads(a)), b)
                        for a, b in zip(old[column], new[column]))
        else:
            equal = np.array_equal(old[column].to_numpy(), new[column].to_numpy())
        if not equal:
            raise ValueError(f'Diferencias con la referencia en {column}. No se exportaron resultados.')
    return {'estado': 'coincidencia_exacta', 'filas': len(rows)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(data_path='dataset/15 atributos R0-R5.sav', splits_dir='results/g4_2_central_unweighted',
         local='results/g5_2_perfiles', out='results/resumen/perfiles', reference=None):
    local, out, splits_dir = Path(local), Path(out), Path(splits_dir)
    if local.resolve() == out.resolve() or local.resolve() in out.resolve().parents or out.resolve() in local.resolve().parents:
        raise ValueError('Las carpetas individuales y agregadas deben estar separadas.')
    for path in (local, out):
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise FileExistsError(f'La salida ya contiene archivos; use una carpeta nueva: {path}')
    central_manifest = splits_dir / 'manifest.json'
    central = json.loads(central_manifest.read_text())
    data_hash = sha256(data_path)
    if central['status'] != 'terminado' or central['identity']['data_sha256'] != data_hash:
        raise ValueError('Los datos no corresponden a la corrida central terminada.')
    loader = DataFrameLoader()
    dataframe = loader.load(data_path)
    frames, targets, split_hashes = [], {}, {}
    for target in TARGET_COLUMNS:
        data = prepare_experiment_data(dataframe, target, loader)
        split_path = splits_dir / target / 'splits.npz'
        pairs = validated_splits(split_path, data)
        frames.extend(predict_fold(data, train, test, fold)
                      for fold, (train, test) in enumerate(pairs, 1))
        targets[target] = {'classes': data.classes, 'filas': len(data.y), 'folds': len(pairs)}
        split_hashes[target] = sha256(split_path)
    rows = pd.concat(frames, ignore_index=True).sort_values(['objetivo', 'row_id']).reset_index(drop=True)
    comparison = compare_reference(rows, reference) if reference is not None else {'estado': 'no_solicitada'}
    root = Path(__file__).resolve().parents[1]
    sources = ['experimentos/perfiles.py', 'experimentos/decisiones.py', 'src/config.py',
               'src/data/loader.py', 'src/data/preprocessing.py', 'src/evaluation/metrics.py']
    manifest = {
        'status': 'terminado', 'device': 'cpu', 'local_only': True,
        'data_path': str(Path(data_path).resolve()), 'data_sha256': data_hash,
        'splits_dir': str(splits_dir.resolve()), 'central_manifest_sha256': sha256(central_manifest),
        'splits_sha256': split_hashes, 'sources_sha256': {p: sha256(root / p) for p in sources},
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,
                     'scikit-learn': sklearn.__version__},
        'features': FEATURE_COLUMNS, 'alpha': 1, 'fallback': 'global_train_smoothed',
        'split_seed': 42, 'targets': targets, 'filas_oof': len(rows),
        'reference': ({'path': str(Path(reference).resolve()), 'sha256': sha256(reference), **comparison}
                      if reference is not None else comparison),
    }
    # Solo después de validar todos los objetivos y la referencia se crean salidas.
    groups = ['objetivo', 'modelo']
    folds = summarize(rows, out, groups + ['fold'])
    oof = summarize(rows, out, groups)
    local.mkdir(parents=True, exist_ok=True)
    rows.assign(probs=rows.probs.map(lambda p: json.dumps(p.tolist()))).to_csv(local / 'filas.csv', index=False)
    folds.to_csv(out / 'folds.csv', index=False)
    fold_summary(folds, groups + ['regla']).to_csv(out / 'resumen_folds.csv', index=False)
    oof.to_csv(out / 'oof.csv', index=False)
    manifest_path = local / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    audit = {'filas_oof': len(rows), 'folds': sum(t['folds'] for t in targets.values()),
             'vistas': int(rows.visto.sum()), 'nuevas': int((~rows.visto).sum()),
             'referencia': comparison, 'manifest_sha256': sha256(manifest_path),
             'convencion': 'resumen_folds: media/std ddof=0 externos; oof: agregado de filas'}
    (out / 'auditoria.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Perfiles: {len(rows)} filas, {audit["folds"]} folds; referencia: {comparison["estado"]}.')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', default='dataset/15 atributos R0-R5.sav')
    parser.add_argument('--splits-dir', default='results/g4_2_central_unweighted')
    parser.add_argument('--local', default='results/g5_2_perfiles')
    parser.add_argument('--out', default='results/resumen/perfiles')
    parser.add_argument('--reference', help='CSV previo para comprobar coincidencia exacta del modelo perfil.')
    main(**vars(parser.parse_args()))
