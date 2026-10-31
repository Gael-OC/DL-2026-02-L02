"""Regenera OOF y reportes desde los folds guardados, sin entrenar."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.evaluation.metrics import METRIC_KEYS, MetricsCalculator


def _write_csv(path, fields, rows):
    temp = path.with_suffix('.tmp')
    with temp.open('w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def regenerate_reports(folder):
    folder = Path(folder)
    outputs = []
    for split_path in sorted(folder.glob('*/splits.npz')):
        target_dir = split_path.parent
        with np.load(split_path, allow_pickle=False) as splits:
            classes = splits['classes'].tolist()
            labels = splits['labels']
            fold_keys = sorted((k for k in splits.files if k.startswith('outer_') and k.endswith('_test')),
                               key=lambda k: int(k.split('_')[1]))
            expected = {int(k.split('_')[1]): splits[k].copy() for k in fold_keys}
        for variant_dir in sorted(p for p in target_dir.iterdir() if p.is_dir()):
            paths = sorted(variant_dir.glob('fold_*.json'))
            if not paths:
                continue
            folds = [json.loads(p.read_text(encoding='utf-8')) for p in paths]
            if {f['outer_fold'] for f in folds} != set(expected):
                raise ValueError(f'Folds incompletos en {variant_dir}; no se genera un agregado OOF.')
            rows, all_true, all_pred, all_probs, fold_metrics = [], [], [], [], []
            for fold in folds:
                index = np.asarray(fold['row_ids'], dtype=int)
                true = np.asarray(fold['y_true'], dtype=int)
                pred = np.asarray(fold['y_pred'], dtype=int)
                probs = np.asarray(fold['probabilities'], dtype=np.dtype(fold['probability_dtype']))
                logits = np.asarray(fold['logits'], dtype=float)
                if (not np.array_equal(index, expected[fold['outer_fold']]) or
                    not np.array_equal(true, labels[index]) or
                    logits.shape[0] != len(index) or probs.shape != (len(index), len(classes)) or
                    pred.shape != true.shape):
                    raise ValueError(f'Datos incompatibles con particiones en {variant_dir}.')
                # Valida probabilidades y obtiene metricas de los archivos, no del checkpoint.
                metrics = MetricsCalculator.compute(true, pred, len(classes))
                metrics.update(nll=MetricsCalculator.nll(true, probs), rps=MetricsCalculator.rps(true, probs))
                fold_metrics.append(metrics)
                for i, row_id in enumerate(index):
                    rows.append({'row_id': int(row_id), 'repeticion': 1, 'fold': fold['outer_fold'],
                                 'verdad_indice': int(true[i]), 'verdad_original': classes[true[i]],
                                 'decision_indice': int(pred[i]), 'decision_original': classes[pred[i]],
                                 'logits': json.dumps(logits[i].tolist()), 'probabilidades': json.dumps(probs[i].tolist())})
                all_true.extend(true); all_pred.extend(pred); all_probs.append(probs)
            rows.sort(key=lambda x: x['row_id'])
            if [r['row_id'] for r in rows] != list(range(len(labels))):
                raise ValueError(f'Cobertura OOF incorrecta en {variant_dir}.')
            true, pred, probs = np.asarray(all_true), np.asarray(all_pred), np.concatenate(all_probs)
            oof = MetricsCalculator.compute(true, pred, len(classes))
            oof.update(nll=MetricsCalculator.nll(true, probs), rps=MetricsCalculator.rps(true, probs))
            summary = {}
            for name in [*METRIC_KEYS, 'nll', 'rps']:
                values = np.array([m[name] for m in fold_metrics])
                valid = values[np.isfinite(values)]
                summary[name] = {'media': float(valid.mean()) if len(valid) else None,
                                 'std_ddof_0': float(valid.std()) if len(valid) else None,
                                 'folds_validos': len(valid), 'folds_total': len(values)}
            fields = ['row_id', 'repeticion', 'fold', 'verdad_indice', 'verdad_original',
                      'decision_indice', 'decision_original', 'logits', 'probabilidades']
            _write_csv(variant_dir / 'predicciones_oof.csv', fields, rows)
            matrix = MetricsCalculator.confusion_matrix(true, pred, len(classes))
            _write_csv(variant_dir / 'confusion_oof.csv', ['real'] + [str(c) for c in classes],
                       [{'real': str(c), **{str(col): int(matrix[i, j]) for j, col in enumerate(classes)}}
                        for i, c in enumerate(classes)])
            report = {'objetivo': target_dir.name, 'variante': variant_dir.name, 'repeticion': 1,
                      'clases_originales': classes, 'soporte_real': np.bincount(true, minlength=len(classes)).tolist(),
                      'resumen_folds_externos': summary, 'agregado_oof': oof,
                      'busqueda_interna_omitida_por_fold': {str(f['outer_fold']): f['inner_skipped'] for f in folds},
                      'nota': 'El agregado OOF y la media de folds son estadisticos distintos.'}
            (variant_dir / 'reporte.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            outputs.append(variant_dir)
    if not outputs:
        raise ValueError('No hay folds completos con particiones para regenerar.')
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('carpeta', type=Path)
    args = parser.parse_args()
    for path in regenerate_reports(args.carpeta):
        print(f'Reportes regenerados: {path}')


if __name__ == '__main__':
    main()
