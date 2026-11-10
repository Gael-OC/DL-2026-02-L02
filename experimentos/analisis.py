"""Análisis pareado de decisiones y perfiles desde artefactos locales, sin entrenar."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experimentos.decisiones import decisions, fold_summary
from experimentos.perfiles import sha256
from src.config import TARGET_COLUMNS
from src.evaluation.metrics import MetricsCalculator

MODELS = ['softmax', 'coral', 'coral_weighted', 'perfil']
NAMES = ['Softmax', 'CORAL', 'CORAL + pesos', 'Perfil']
GROUPS = ['objetivo', 'modelo', 'grupo', 'regla']


def load_rows(decision_path, profile_dir, unweighted, weighted):
    """Exige pares completos y coteja las decisiones con sus folds de origen."""
    decision_path, profile_dir = Path(decision_path), Path(profile_dir)
    unweighted, weighted = Path(unweighted), Path(weighted)
    profile_path = profile_dir / 'filas.csv'
    sources = [decision_path, profile_path, profile_dir / 'manifest.json',
               unweighted / 'manifest.json', weighted / 'manifest.json']
    profile_manifest = json.loads((profile_dir / 'manifest.json').read_text())
    central = [json.loads((p / 'manifest.json').read_text()) for p in (unweighted, weighted)]
    if (profile_manifest['status'] != 'terminado' or any(m['status'] != 'terminado' for m in central)
            or any(m['identity']['data_sha256'] != profile_manifest['data_sha256'] for m in central)):
        raise ValueError('Los manifiestos no corresponden a los mismos datos terminados.')
    neural, profiles = pd.read_csv(decision_path), pd.read_csv(profile_path)
    required = {'objetivo', 'modelo', 'row_id', 'fold', 'verdad', 'k', 'probs', 'moda', 'mediana'}
    for frame, models in ((neural, set(MODELS[:-1])), (profiles, {'perfil'})):
        if not required <= set(frame.columns) or set(frame.modelo) != models or set(frame.objetivo) != set(TARGET_COLUMNS):
            raise ValueError('Faltan objetivos, modelos o columnas en las predicciones.')
        if frame.duplicated(['objetivo', 'modelo', 'row_id']).any():
            raise ValueError('Hay filas duplicadas; los pares deben ser uno a uno.')
        for column in ('row_id', 'fold', 'verdad', 'k', 'moda', 'mediana'):
            if not pd.api.types.is_integer_dtype(frame[column]):
                raise ValueError(f'Índices o etiquetas inválidos: {column}')
    if 'original' not in neural or 'visto' not in profiles or profiles.visto.dtype != bool:
        raise ValueError('Faltan decisiones originales o marcas visto/nuevo válidas.')
    if not pd.api.types.is_integer_dtype(neural.original):
        raise ValueError('La decisión original debe contener índices enteros.')
    classes = {}
    for target in TARGET_COLUMNS:
        split_paths = [base / target / 'splits.npz' for base in (unweighted, weighted)]
        sources.extend(split_paths)
        if sha256(split_paths[0]) != profile_manifest['splits_sha256'][target]:
            raise ValueError(f'Las particiones de perfiles cambiaron: {target}')
        with np.load(split_paths[0], allow_pickle=False) as left, np.load(split_paths[1], allow_pickle=False) as right:
            if set(left.files) != set(right.files) or any(not np.array_equal(left[k], right[k]) for k in left.files):
                raise ValueError(f'Particiones distintas entre variantes: {target}')
            labels, original_classes = left['labels'], left['classes']
            classes[target] = original_classes.tolist()
            if classes[target] != profile_manifest['targets'][target]['classes']:
                raise ValueError(f'Alfabeto de perfiles incompatible: {target}')
            for model in MODELS:
                frame = profiles if model == 'perfil' else neural
                part = frame.loc[(frame.objetivo == target) & (frame.modelo == model)]
                if sorted(part.row_id.tolist()) != list(range(len(labels))):
                    raise ValueError(f'Cobertura incompleta: {target}/{model}')
                expected = 2 if target == 'GDS' else 5
                if set(part.fold) != set(range(1, expected + 1)):
                    raise ValueError(f'Folds incompletos: {target}/{model}')
                for fold in range(1, expected + 1):
                    test = left[f'outer_{fold}_test']
                    block = part.loc[part.fold == fold]
                    if sorted(block.row_id.tolist()) != sorted(test.tolist()):
                        raise ValueError(f'Fold incompatible: {target}/{model}/{fold}')
                    positions = pd.Series(block.index.to_numpy(), index=block.row_id).loc[test].to_numpy()
                    block = block.set_index('row_id').loc[test]
                    if not np.array_equal(block.verdad, labels[test]) or not np.all(block.k == len(original_classes)):
                        raise ValueError(f'Etiquetas externas incompatibles: {target}/{model}/{fold}')
                    dtype = np.float64
                    if model != 'perfil':
                        base = weighted if model == 'coral_weighted' else unweighted
                        source = base / target / model / f'fold_{fold}.json'
                        sources.append(source)
                        saved = json.loads(source.read_text())
                        dtype = np.dtype(saved['probability_dtype'])
                        if (not np.array_equal(saved['row_ids'], test) or not np.array_equal(saved['y_true'], labels[test])
                                or not np.array_equal(saved['y_pred'], block.original)):
                            raise ValueError(f'Decisión original incompatible: {source}')
                    probs = np.asarray([json.loads(p) for p in block.probs], dtype=dtype)
                    mode, median = decisions(probs)
                    if (probs.shape != (len(test), len(original_classes)) or not np.array_equal(mode, block.moda)
                            or not np.array_equal(median, block.mediana)):
                        raise ValueError(f'Probabilidades o reglas incompatibles: {target}/{model}/{fold}')
                    if model != 'perfil' and not np.array_equal(probs, np.asarray(saved['probabilities'], dtype=dtype)):
                        raise ValueError(f'Las probabilidades cambiaron: {source}')
                    # Mantiene el dtype de origen al evaluar las reglas; exportar a JSON no lo conserva.
                    frame.loc[positions, 'probs'] = pd.Series(list(probs), index=positions, dtype=object)
    tags = profiles[['objetivo', 'row_id', 'fold', 'verdad', 'k', 'visto']]
    neural = neural.merge(tags, on=['objetivo', 'row_id'], suffixes=('', '_perfil'), validate='many_to_one')
    for column in ('fold', 'verdad', 'k'):
        if not np.array_equal(neural[column], neural[column + '_perfil']):
            raise ValueError(f'Pares incompatibles entre redes y perfiles: {column}')
    columns = ['objetivo', 'modelo', 'row_id', 'fold', 'verdad', 'k', 'probs', 'moda', 'mediana', 'visto']
    rows = pd.concat([neural[columns + ['original']], profiles[columns]], ignore_index=True)
    return rows, classes, {str(path): sha256(path) for path in sources}, profile_manifest['data_sha256']


def cohorts(frame):
    yield 'todos', frame
    yield 'visto', frame.loc[frame.visto]
    yield 'nuevo', frame.loc[~frame.visto]


def tables(rows, classes, by_fold):
    """Métricas, recalls con soporte y diferencias calculadas sobre las mismas filas."""
    metrics, recalls, pairs = [], [], []
    keys = ['objetivo', 'modelo'] + (['fold'] if by_fold else [])
    for group_values, frame in rows.groupby(keys, sort=True):
        identity = dict(zip(keys, group_values))
        for cohort, part in cohorts(frame):
            if part.empty:
                continue
            common = {**identity, 'grupo': cohort}
            y = part.verdad.to_numpy(dtype=int)
            k = int(part.k.iloc[0])
            probabilities = np.stack(part.probs.to_numpy())
            probability_metrics = {'nll': MetricsCalculator.nll(y, probabilities),
                                   'rps': MetricsCalculator.rps(y, probabilities)}
            rules = ['moda', 'mediana'] + (['original'] if part.original.notna().all() else [])
            for rule in rules:
                pred = part[rule].to_numpy(dtype=int)
                error = pred - y
                scores = MetricsCalculator.compute(y, pred, k)
                scores.update(probability_metrics, error_medio=float(error.mean()),
                              subestimacion=float(np.mean(error < 0)), sobreestimacion=float(np.mean(error > 0)),
                              graves_inferiores=float(np.mean(error <= -2)), graves_superiores=float(np.mean(error >= 2)))
                metrics.append({**common, 'regla': rule, 'n': len(y), **scores})
                for label in range(k):
                    support = int(np.sum(y == label))
                    recalls.append({**common, 'nivel': 'fold' if by_fold else 'oof', 'regla': rule,
                                    'clase_indice': label, 'clase_original': classes[identity['objetivo']][label],
                                    'soporte': support, 'recall': float(np.mean(pred[y == label] == label)) if support else np.nan})
            comparisons = [('moda', 'mediana')]
            if 'original' in rules:
                comparisons += [('original', 'moda'), ('original', 'mediana')]
            for before, after in comparisons:
                a, b = part[before].to_numpy(dtype=int), part[after].to_numpy(dtype=int)
                ea, eb = np.abs(a - y), np.abs(b - y)
                pairs.append({**common, 'comparacion': f'{after}-{before}', 'n': len(y),
                              'n_cambios': int(np.sum(a != b)), 'mejoras_mae': int(np.sum(eb < ea)),
                              'empeora_mae': int(np.sum(eb > ea)), 'igual_mae': int(np.sum(eb == ea)),
                              'delta_mae': float(np.mean(eb - ea)),
                              'delta_accuracy': float(np.mean(b == y) - np.mean(a == y)),
                              'delta_errores_graves': float(np.mean(eb >= 2) - np.mean(ea >= 2))})
    return pd.DataFrame(metrics), pd.DataFrame(recalls), pd.DataFrame(pairs)


def figures(paired, oof, profiles, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = ['#2563a0', '#3a8b72', '#a77424', '#8654a2']
    with plt.rc_context({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False}):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), layout='constrained')
        for ax, target in zip(axes.flat, TARGET_COLUMNS):
            ax.axhline(0, color='#8a929b', lw=1)
            for i, model in enumerate(MODELS):
                p = paired.loc[(paired.objetivo == target) & (paired.modelo == model)
                               & (paired.grupo == 'todos') & (paired.comparacion == 'mediana-moda')]
                ax.scatter(i + np.linspace(-.13, .13, len(p)), p.delta_mae, color=colors[i], alpha=.6, s=30)
                ax.plot([i-.24, i+.24], [p.delta_mae.mean()]*2, color=colors[i], lw=3)
            ax.set_title(target, fontweight='bold')
            ax.set_xticks(range(4), NAMES, rotation=15)
            ax.set_ylabel('Δ MAE: mediana − moda')
            if target == 'GDS_R3':
                ax.set_ylim(-.01, .01)
            ax.grid(axis='y', alpha=.15)
        fig.suptitle('Cambiar la decisión sobre las mismas probabilidades\nPuntos: folds externos · Trazo: media · Negativo: menor MAE con mediana', fontsize=14)
        fig.supxlabel('Escalas verticales independientes; las reglas no se seleccionaron con estos resultados.', fontsize=10)
        fig.savefig(out / 'decisiones.png', dpi=180)
        plt.close(fig)
        fig, axes = plt.subplots(2, 3, figsize=(14, 8.5), layout='constrained')
        for ax, target in zip(axes.flat, TARGET_COLUMNS):
            tags = profiles.loc[profiles.objetivo == target]
            ax.set_title(f'{target} · vistos {int(tags.visto.sum())} / nuevos {int((~tags.visto).sum())}', fontweight='bold', fontsize=11)
            for offset, cohort, color, label in [(-.18, 'visto', '#2563a0', 'Vistos'), (.18, 'nuevo', '#d27a35', 'Nuevos')]:
                p = oof.loc[(oof.objetivo == target) & (oof.grupo == cohort) & (oof.regla == 'mediana')].set_index('modelo')
                ax.bar(np.arange(4) + offset, p.loc[MODELS, 'mae_ordinal'], width=.34, color=color, label=label)
            ax.set_xticks(range(4), NAMES, rotation=15)
            ax.set_ylabel('MAE OOF')
            ax.grid(axis='y', alpha=.15)
        axes[0, 0].legend(frameon=False)
        fig.suptitle('Errores en perfiles vistos y nuevos durante el entrenamiento\nRegla mediana fijada para los cuatro modelos · Agregado de filas OOF', fontsize=14)
        fig.supxlabel('Cada soporte cuenta filas, no perfiles únicos. Escalas verticales independientes.', fontsize=10)
        fig.savefig(out / 'perfiles.png', dpi=180)
        plt.close(fig)


def discussion(paired, paired_oof, oof, recalls, out):
    lines = ['# Decisiones y perfiles de respuesta', '',
             'Se reutilizan probabilidades externas de Softmax, CORAL, CORAL ponderada y el predictor por perfil, sin entrenar redes. '
             'Las reglas se comparan sobre las mismas filas, probabilidades y folds. La selección interna de las redes se hizo con '
             'su decisión original; no se volvió a optimizar para cada regla.', '',
             '## Cambios de decisión', '',
             'Diferencia = mediana − moda. Las medias y desviaciones (ddof=0) corresponden a folds externos; no son intervalos de confianza. '
             'Los cambios cuentan filas OOF. Las cifras no seleccionan una regla ganadora para futuros datos.', '',
             '| Objetivo | Modelo | Δ MAE externo | Δ errores graves externo | Cambios OOF |',
             '|---|---|---:|---:|---:|']
    for target in TARGET_COLUMNS:
        for model, name in zip(MODELS, NAMES):
            p = paired.loc[(paired.objetivo == target) & (paired.modelo == model) & (paired.grupo == 'todos') & (paired.comparacion == 'mediana-moda')]
            aggregate = paired_oof.loc[(paired_oof.objetivo == target) & (paired_oof.modelo == model) & (paired_oof.grupo == 'todos') & (paired_oof.comparacion == 'mediana-moda')].iloc[0]
            lines.append(f'| {target} | {name} | {p.delta_mae.mean():.4f} ± {p.delta_mae.std(ddof=0):.4f} | {p.delta_errores_graves.mean():.4f} | {int(aggregate.n_cambios)} |')
    lines += ['', '![Diferencias pareadas de MAE](decisiones.png)', '',
              'Los archivos pareados también incluyen mediana−original y moda−original para las redes. '
              'El error firmado se define como predicción−verdad: valores negativos indican subestimación. '
              'Las tablas conservan las tasas de subestimación, sobreestimación y errores graves por dirección.', '']
    direction = oof.loc[(oof.objetivo == 'GDS_R2') & (oof.grupo == 'todos') & (oof.modelo == 'softmax')].set_index('regla')
    lines += [f'En R2, Softmax pasa de MAE OOF {direction.loc["moda", "mae_ordinal"]:.4f} con moda '
              f'a {direction.loc["mediana", "mae_ordinal"]:.4f} con mediana. La subestimación pasa de '
              f'{direction.loc["moda", "subestimacion"]:.1%} a {direction.loc["mediana", "subestimacion"]:.1%}, '
              f'y la sobreestimación de {direction.loc["moda", "sobreestimacion"]:.1%} a '
              f'{direction.loc["mediana", "sobreestimacion"]:.1%}.', '',
              '## Perfiles vistos y nuevos', '',
              'Se usa la mediana como regla común prefijada para describir el MAE, sin elegirla por el test. '
              'Visto significa que las 15 respuestas aparecieron en el entrenamiento del fold; no identifica una persona. '
              'Para un perfil nuevo, el predictor por perfil recurre a la distribución global suavizada del entrenamiento. '
              'Las tablas también incluyen moda y las decisiones originales, con soportes por clase.', '',
              'Ejemplo en GDS_R2: valores agregados OOF, distintos del promedio de folds.', '',
              '| Modelo | MAE vistos | MAE nuevos | Graves vistos | Graves nuevos |', '|---|---:|---:|---:|---:|']
    for model, name in zip(MODELS, NAMES):
        p = oof.loc[(oof.objetivo == 'GDS_R2') & (oof.modelo == model) & (oof.regla == 'mediana')].set_index('grupo')
        lines.append(f'| {name} | {p.loc["visto", "mae_ordinal"]:.4f} | {p.loc["nuevo", "mae_ordinal"]:.4f} | {p.loc["visto", "errores_graves"]:.4f} | {p.loc["nuevo", "errores_graves"]:.4f} |')
    lines += ['', '![MAE por perfiles vistos y nuevos](perfiles.png)', '',
              '### Recall por clase en GDS_R2, regla mediana', '',
              '| Modelo | Grupo | Soportes (clases 1/2/3) | Recalls (clases 1/2/3) |', '|---|---|---|---|']
    for model, name in zip(MODELS, NAMES):
        for cohort in ('visto', 'nuevo'):
            p = recalls.loc[(recalls.objetivo == 'GDS_R2') & (recalls.modelo == model) & (recalls.regla == 'mediana')
                            & (recalls.grupo == cohort) & (recalls.nivel == 'oof')].sort_values('clase_indice')
            supports = ' / '.join(str(int(n)) for n in p.soporte)
            values = ' / '.join(f'{r:.3f}' if np.isfinite(r) else 'no estimable' for r in p.recall)
            lines.append(f'| {name} | {cohort} | {supports} | {values} |')
    lines += ['', '## Alcance', '',
              'Cambiar moda por mediana modifica la decisión, no el aprendizaje. Comparar el predictor por perfil con una red '
              'sí cambia el estimador y el aprendizaje; sus diferencias no se atribuyen solo a la regla. '
              'Los grupos visto/nuevo tienen distinta composición y soporte: una diferencia de errores no demuestra por sí sola un efecto causal de novedad.', '',
              'Los perfiles con etiquetas contradictorias describen ambigüedad empírica del archivo completo; no estiman el límite poblacional '
              'ni una cota que deba respetar el conjunto de modelos OOF. Los seis objetivos usan los mismos registros y los entrenamientos '
              'de los folds se solapan. No se aplican pruebas que traten folds u objetivos como muestras independientes. '
              'Los recalls sin soporte se dejan como no estimables, no como ceros.', '',
              'El mapa de evidencia está en [resultados](../../../docs/resultados.md#decisiones-y-perfiles).', '']
    (out / 'discusion.md').write_text('\n'.join(lines), encoding='utf-8')


def main(decisions_path='results/g5_decisiones/filas.csv', profiles='results/g5_2_perfiles',
         unweighted='results/g4_2_central_unweighted', weighted='results/g4_2_central_weighted',
         out='results/resumen/analisis'):
    out = Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f'Use una carpeta de salida nueva o vacía: {out}')
    rows, classes, sources, data_hash = load_rows(decisions_path, profiles, unweighted, weighted)
    folds, recall_folds, paired_folds = tables(rows, classes, True)
    oof, recall_oof, paired_oof = tables(rows, classes, False)
    recalls = pd.concat([recall_folds, recall_oof], ignore_index=True)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in [('metricas_folds', folds), ('resumen_folds', fold_summary(folds, GROUPS)),
                        ('metricas_oof', oof), ('recall_clases', recalls),
                        ('pareadas_folds', paired_folds), ('pareadas_oof', paired_oof)]:
        frame.to_csv(out / f'{name}.csv', index=False)
    figures(paired_folds, oof, rows.loc[rows.modelo == 'perfil'], out)
    discussion(paired_folds, paired_oof, oof, recalls, out)
    root = Path(__file__).resolve().parents[1]
    for name in ['experimentos/analisis.py', 'experimentos/decisiones.py', 'experimentos/perfiles.py', 'src/evaluation/metrics.py']:
        sources[name] = sha256(root / name)
    audit = {'status': 'terminado', 'local_only': True, 'data_sha256': data_hash,
             'filas_pareadas_redes': int(np.sum(rows.modelo != 'perfil')), 'filas_perfil': int(np.sum(rows.modelo == 'perfil')),
             'objetivos': classes, 'fuentes_sha256': sources,
             'regla_figura_perfiles': 'mediana prefijada; sin selección con test',
             'error_firmado': 'predicción - verdad', 'ddof': 0}
    (out / 'auditoria.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Análisis completo: {len(rows)} predicciones, dos figuras y discusión en {out}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--decisions-path', default='results/g5_decisiones/filas.csv')
    parser.add_argument('--profiles', default='results/g5_2_perfiles')
    parser.add_argument('--unweighted', default='results/g4_2_central_unweighted')
    parser.add_argument('--weighted', default='results/g4_2_central_weighted')
    parser.add_argument('--out', default='results/resumen/analisis')
    main(**vars(parser.parse_args()))
