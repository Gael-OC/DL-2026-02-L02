"""Decisiones pareadas desde probabilidades OOF centrales persistidas."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.evaluation.metrics import MetricsCalculator
from src.config import TARGET_COLUMNS


def decisions(probs):
    p = np.asarray(probs)
    if p.ndim != 2 or not p.shape[1] or not np.all(np.isfinite(p)) or np.any(p < 0) or not np.allclose(p.sum(axis=1), 1, atol=1e-6):
        raise ValueError('Probabilidades inválidas')
    return p.argmax(axis=1), (p.cumsum(axis=1) >= .5).argmax(axis=1)


def summarize(df, out, groups):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for keys, part in df.groupby(groups, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        k = int(part.k.iloc[0]); y = part.verdad.to_numpy(dtype=int)
        p = np.stack(part.probs.to_numpy())
        for rule in (('original', 'moda', 'mediana') if 'original' in part.columns else ('moda', 'mediana')):
            pred = part[rule].to_numpy(dtype=int)
            metrics = MetricsCalculator.compute(y, pred, k)
            metrics.update(nll=MetricsCalculator.nll(y, p), rps=MetricsCalculator.rps(y, p))
            rows.append({**dict(zip(groups, keys)), 'regla': rule, 'n': len(part), **metrics})
    return pd.DataFrame(rows)



def fold_summary(folds, groups):
    metrics = [c for c in folds.columns if c not in groups + ['fold', 'n']]
    rows = []
    for keys, part in folds.groupby(groups, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(groups, keys))
        for metric in metrics:
            values = part[metric].dropna()
            row[metric + '_media'] = values.mean()
            row[metric + '_std0'] = values.std(ddof=0)
            row[metric + '_folds_validos'] = len(values)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_folds(df, groups):
    rows = []
    for keys, part in df.groupby(groups + ['fold'], sort=True):
        y = part.verdad.to_numpy(dtype=int)
        mode = part.moda.to_numpy(dtype=int)
        median = part.mediana.to_numpy(dtype=int)
        a = np.abs(y - mode); b = np.abs(y - median)
        rows.append({**dict(zip(groups + ['fold'], keys)), 'n': len(part),
                     'n_cambios': int(np.sum(mode != median)),
                     'delta_mae': float(b.mean() - a.mean()),
                     'delta_accuracy': float(np.mean(y == median) - np.mean(y == mode)),
                     'delta_errores_graves': float(np.mean(b >= 2) - np.mean(a >= 2))})
    return pd.DataFrame(rows)

def main(unweighted='results/g4_2_central_unweighted', weighted='results/g4_2_central_weighted', local='results/g5_decisiones', out='results/resumen/decisiones'):
    rows = []
    for weight in ('unweighted', 'weighted'):
        base = Path(unweighted if weight == 'unweighted' else weighted)
        splits = sorted(base.glob('*/splits.npz'))
        if {p.parent.name for p in splits} != set(TARGET_COLUMNS):
            raise ValueError(f'Faltan objetivos o particiones en {base}: se requieren {list(TARGET_COLUMNS)}')
        for split in splits:
            with np.load(split) as s:
                labels, classes = s['labels'], s['classes']
                for model in (('softmax', 'coral') if weight == 'unweighted' else ('coral_weighted',)):
                    covered = []
                    paths = sorted((split.parent / model).glob('fold_*.json'))
                    expected = 2 if split.parent.name == 'GDS' else 5
                    if len(paths) != expected:
                        raise ValueError(f'Faltan folds de la variante {split.parent / model}: se requieren {expected}')
                    for path in paths:
                        f = json.loads(path.read_text()); fold = f['outer_fold']
                        ids = np.asarray(f['row_ids'], dtype=int); y = np.asarray(f['y_true'], dtype=int)
                        p = np.asarray(f['probabilities'], dtype=f['probability_dtype'])
                        if (f'outer_{fold}_test' not in s.files or ids.ndim != 1 or np.any(ids < 0) or np.any(ids >= len(labels))
                                or not np.array_equal(ids, s[f'outer_{fold}_test']) or not np.array_equal(y, labels[ids])
                                or p.shape != (len(ids), len(classes))):
                            raise ValueError(f'Fold incompatible: {path}')
                        mode, median = decisions(p); covered.extend(ids.tolist())
                        for i, idx in enumerate(ids):
                            rows.append(dict(objetivo=split.parent.name, ponderacion=weight, modelo=model, fold=fold, row_id=idx, verdad=y[i], original=f['y_pred'][i], k=len(classes), probs=p[i], moda=mode[i], mediana=median[i]))
                    if sorted(covered) != list(range(len(labels))):
                        raise ValueError(f'Cobertura incorrecta: {split.parent}/{model}')
    df = pd.DataFrame(rows); local = Path(local); local.mkdir(parents=True, exist_ok=True)
    df.assign(probs=df.probs.map(lambda x: json.dumps(x.tolist()))).to_csv(local/'filas.csv', index=False)
    out = Path(out); groups = ['objetivo','ponderacion','modelo']
    folds = summarize(df,out,groups+['fold'])
    folds.to_csv(out/'folds.csv',index=False)
    fold_summary(folds, groups+['regla']).to_csv(out/'resumen_folds.csv',index=False)
    paired_folds(df, groups).to_csv(out/'pareadas_folds.csv',index=False)
    summarize(df,out,groups).to_csv(out/'oof.csv',index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unweighted', default='results/g4_2_central_unweighted')
    parser.add_argument('--weighted', default='results/g4_2_central_weighted')
    parser.add_argument('--local', default='results/g5_decisiones')
    parser.add_argument('--out', default='results/resumen/decisiones')
    main(**vars(parser.parse_args()))
