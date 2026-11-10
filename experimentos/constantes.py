"""Reconstruye moda y mediana constantes en los seis objetivos, sin entrenar redes."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experimentos.controles import ConstantPredictor
from experimentos.decisiones import decisions, fold_summary, summarize
from experimentos.perfiles import sha256, validated_splits
from src.config import TARGET_COLUMNS
from src.data.loader import DataFrameLoader
from src.data.preprocessing import prepare_experiment_data


def predict_fold(data, train, test, fold):
    model = ConstantPredictor(classes=np.arange(data.num_classes)).fit(data.y[train])
    probs = model.predict_proba(data.X[test])
    mode, median = decisions(probs)
    return pd.DataFrame(dict(objetivo=data.target_name, modelo='constante', fold=fold,
                             row_id=test, verdad=data.y[test], k=data.num_classes,
                             probs=list(probs), moda=mode, mediana=median))


def main(data_path='dataset/15 atributos R0-R5.sav', splits_dir='results/g4_2_central_unweighted',
         out='results/resumen/constantes'):
    out, splits_dir = Path(out), Path(splits_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Use una salida nueva o vacía: {out}')
    manifest = splits_dir / 'manifest.json'
    central = json.loads(manifest.read_text())
    if central['status'] != 'terminado' or sha256(data_path) != central['identity']['data_sha256']:
        raise ValueError('Los datos no corresponden a la corrida central terminada.')
    loader = DataFrameLoader()
    frame = loader.load(data_path)
    frames, hashes = [], {}
    for target in TARGET_COLUMNS:
        data = prepare_experiment_data(frame, target, loader)
        path = splits_dir / target / 'splits.npz'
        pairs = validated_splits(path, data)
        frames.extend(predict_fold(data, tr, te, fold) for fold, (tr, te) in enumerate(pairs, 1))
        hashes[target] = sha256(path)
    rows = pd.concat(frames, ignore_index=True)
    groups = ['objetivo', 'modelo']
    folds = summarize(rows, out, groups + ['fold'])
    folds.to_csv(out / 'folds.csv', index=False)
    fold_summary(folds, groups + ['regla']).to_csv(out / 'resumen_folds.csv', index=False)
    summarize(rows, out, groups).to_csv(out / 'oof.csv', index=False)
    root = Path(__file__).resolve().parents[1]
    sources = ['experimentos/constantes.py', 'experimentos/controles.py', 'experimentos/decisiones.py',
               'experimentos/perfiles.py', 'src/data/loader.py', 'src/data/preprocessing.py',
               'src/config.py', 'src/evaluation/metrics.py']
    audit = dict(filas_oof=len(rows), folds=len(folds) // 2, data_sha256=sha256(data_path),
                 central_manifest_sha256=sha256(manifest), splits_sha256=hashes,
                 sources_sha256={name: sha256(root / name) for name in sources},
                 ddof=0, conteos='Solo entrenamiento externo, sin suavizado; empate hacia la clase menor.')
    (out / 'auditoria.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False) + '\n')
    print(f'Constantes: {len(rows)} filas OOF y {audit["folds"]} folds; solo agregados exportados.')
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', default='dataset/15 atributos R0-R5.sav')
    parser.add_argument('--splits-dir', default='results/g4_2_central_unweighted')
    parser.add_argument('--out', default='results/resumen/constantes')
    main(**vars(parser.parse_args()))
