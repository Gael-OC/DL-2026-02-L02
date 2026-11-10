"""Audita y resume la robustez sin reentrenar: python -m experimentos.robustez."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from experimentos.auditar_corrida import auditar
from experimentos.comparacion import write
from src.evaluation.metrics import METRIC_KEYS


def resumir(output=Path('results/resumen/robustez')):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Salida no vacía: {output}')
    central = [Path('results/g4_2_central_unweighted'), Path('results/g4_2_central_weighted')]
    jobs = [(p, n) for p, n in zip(central, (654, 327))]
    for target in ('GDS_R1', 'GDS_R2'):
        for seed in (314, 2718):
            jobs.extend((Path(f'results/h5_robustez/{target}_split{seed}_{kind}'), n)
                        for kind, n in (('unweighted', 130), ('weighted', 65)))
    identities = []
    for path, n in jobs:
        auditar(path, n)
        identities.append(json.loads((path / 'manifest.json').read_text())['identity'])
    if len({x['data_sha256'] for x in identities}) != 1 or len({x['code_sha256'] for x in identities}) != 1:
        raise ValueError('Datos o código difieren entre corridas')

    rows, selections, pairs = [], [], []
    for target in ('GDS_R1', 'GDS_R2'):
        for seed in (42, 314, 2718):
            folders = central if seed == 42 else [Path(f'results/h5_robustez/{target}_split{seed}_{kind}') for kind in ('unweighted', 'weighted')]
            splits = []
            for path in folders:
                with np.load(path / target / 'splits.npz') as data:
                    splits.append({k: data[k].copy() for k in data.files})
            if set(splits[0]) != set(splits[1]) or any(not np.array_equal(splits[0][k], splits[1][k]) for k in splits[0]):
                raise ValueError(f'Particiones no pareadas: {target}, {seed}')
            reports = {}
            for path in folders:
                for report in (path / target).glob('*/reporte.json'):
                    data = json.loads(report.read_text())
                    variant = data['variante']
                    if variant in reports:
                        raise ValueError(f'Variante duplicada: {target}, {seed}, {variant}')
                    reports[variant] = data
                    row = {'objetivo': target, 'particion': seed, 'variante': variant}
                    for metric in [*METRIC_KEYS, 'nll', 'rps']:
                        summary = data['resumen_folds_externos'][metric]
                        row[metric + '_media'] = summary['media']
                        row[metric + '_std'] = summary['std_ddof_0']
                        row[metric + '_oof'] = data['agregado_oof'][metric]
                    rows.append(row)
                with (path / 'candidatas.csv').open() as f:
                    counts = Counter((r['variante'], r['candidata'], r['configuracion']) for r in csv.DictReader(f)
                                     if r['objetivo'] == target and r['seleccionada'] == 'True')
                for (variant, candidate, config), count in sorted(counts.items()):
                    selections.append(dict(objetivo=target, particion=seed, variante=variant,
                                           candidata=candidate, configuracion=config, frecuencia=count))
            if set(reports) != {'softmax', 'coral', 'coral_weighted'}:
                raise ValueError(f'Variantes incompletas: {target}, {seed}')
            for variant in ('coral', 'coral_weighted'):
                pairs.append(dict(objetivo=target, particion=seed, comparacion=f'{variant} - softmax',
                                  **{m: reports[variant]['agregado_oof'][m] - reports['softmax']['agregado_oof'][m]
                                     for m in ('mae_ordinal', 'qwk', 'nll', 'rps')}))
            for variant in reports:
                if sum(r['frecuencia'] for r in selections if r['objetivo'] == target and r['particion'] == seed and r['variante'] == variant) != 5:
                    raise ValueError('Selecciones incompletas')
    output.mkdir(parents=True, exist_ok=True)
    for name, data in [('metricas.csv', rows), ('selecciones.csv', selections), ('diferencias_oof.csv', pairs)]:
        write(output / name, list(data[0]), data)
    lines = ['# Robustez entre particiones', '',
             'Tres particiones externas (42, 314 y 2718), inicialización 42 y búsqueda anidada completa. Métricas OOF por partición; los folds y particiones se solapan y no son muestras independientes.', '',
             '| Objetivo | Partición | Variante | MAE OOF | QWK OOF | NLL OOF | RPS OOF |', '|---|---:|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['objetivo']} | {r['particion']} | {r['variante']} | {r['mae_ordinal_oof']:.4f} | {r['qwk_oof']:.4f} | {r['nll_oof']:.4f} | {r['rps_oof']:.4f} |")
    lines += ['', 'Las diferencias pareadas por partición están en `diferencias_oof.csv`; valores negativos de MAE, NLL y RPS favorecen la variante frente a Softmax. Las frecuencias por candidata y configuración están en `selecciones.csv`. La dispersión de folds es descriptiva y se informa aparte en `metricas.csv`. No se selecciona una partición ganadora.', '',
              'En R1, Softmax tiene menor MAE y mayor QWK que ambas variantes CORAL en las tres particiones. En R2, la comparación de MAE cambia de signo en 314; NLL y RPS favorecen a Softmax en ambas tareas y las tres particiones. Las candidatas seleccionadas varían entre folds y particiones; no hay una candidata elegida en los cinco folds de ninguna condición. Estos resultados describen sensibilidad a la partición, no incertidumbre poblacional ni una prueba de superioridad.', '']
    (output / 'robustez.md').write_text('\n'.join(lines), encoding='utf-8')
    audit = {'ajustes_adicionales': 780, 'corridas': [{'ruta': str(p), 'eventos': n, 'manifest_sha256': hashlib.sha256((p / 'manifest.json').read_bytes()).hexdigest()} for p, n in jobs], 'data_sha256': identities[0]['data_sha256'], 'code_sha256': identities[0]['code_sha256']}
    (output / 'auditoria.json').write_text(json.dumps(audit, indent=2) + '\n')


if __name__ == '__main__':
    resumir()
