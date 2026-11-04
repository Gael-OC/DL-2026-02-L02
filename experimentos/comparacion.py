"""Crea comparaciones agregadas a partir de corridas completas, sin entrenar."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from src.evaluation.metrics import METRIC_KEYS


def write(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def generar(folders, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    reports = {}; splits = {}
    for folder in map(Path, folders):
        for p in folder.glob('*/*/reporte.json'):
            r = json.loads(p.read_text()); key = (r['objetivo'], r['variante'])
            if key in reports: raise ValueError(f'Variante duplicada: {key}')
            with np.load(p.parent.parent / 'splits.npz') as data:
                current = {k: data[k].copy() for k in data.files}
            if r['objetivo'] in splits:
                old = splits[r['objetivo']]
                if set(old) != set(current) or any(not np.array_equal(old[k], current[k]) for k in old):
                    raise ValueError(f"Particiones distintas: {r['objetivo']}")
            else: splits[r['objetivo']] = current
            reports[key] = (r, p.parent)
    if not reports: raise ValueError('No hay reportes')
    markdown = ['# Comparación central', '', 'Media ± desviación estándar de folds externos (ddof=0). GDS usa dos folds y omite la búsqueda interna por soporte insuficiente; los demás objetivos usan cinco folds.', '']
    for target in sorted({k[0] for k in reports}):
        rows = []
        for (t, variant), (r, path) in sorted(reports.items()):
            if t != target: continue
            row = {'variante': variant}
            for metric in [*METRIC_KEYS, 'nll', 'rps']:
                s = r['resumen_folds_externos'][metric]
                row[metric + '_media'], row[metric + '_std_ddof0'] = s['media'], s['std_ddof_0']
            rows.append(row)
            # Matriz agregada, sin predicciones individuales.
            (output / f'{target}_{variant}_confusion.csv').write_text((path / 'confusion_oof.csv').read_text())
        write(output / f'{target}.csv', list(rows[0]), rows)
        markdown.extend([f'## {target}', ''])
        for metrics in [METRIC_KEYS, ['nll', 'rps']]:
            markdown.append('| Variante | ' + ' | '.join(metrics) + ' |')
            markdown.append('| --- | ' + ' | '.join(['---:'] * len(metrics)) + ' |')
            for row in rows:
                cells = [f"{row[m+'_media']:.4f} ± {row[m+'_std_ddof0']:.4f}" if row[m+'_media'] is not None else 'No estimable' for m in metrics]
                markdown.append('| ' + row['variante'] + ' | ' + ' | '.join(cells) + ' |')
            markdown.append('')
    (output / 'comparacion.md').write_text('\n'.join(markdown), encoding='utf-8')
    configs = Counter()
    for folder in map(Path, folders):
        with (folder / 'candidatas.csv').open() as f:
            for row in csv.DictReader(f):
                if row['seleccionada'] == 'True':
                    configs[(row['objetivo'], row['variante'], row['configuracion'])] += 1
    write(output / 'configuraciones.csv', ['objetivo', 'variante', 'configuracion', 'frecuencia'],
          [dict(objetivo=t, variante=v, configuracion=c, frecuencia=n) for (t,v,c),n in sorted(configs.items())])

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('folders', nargs='+'); p.add_argument('--output', required=True)
    a = p.parse_args(); generar(a.folders, a.output)
