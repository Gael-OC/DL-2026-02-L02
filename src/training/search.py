"""Busqueda de hiperparametros en el loop interno de la validacion anidada."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from src.config import ModelConfig, TrainingConfig, apply_overrides
from src.data.preprocessing import StratifiedSplitter
from src.training.trainer import Trainer


@dataclass
class CandidateScore:
    """MAE y QWK promedio de una configuracion sobre los folds internos."""

    overrides: dict[str, Any]
    model_config: ModelConfig
    training_config: TrainingConfig
    mae_mean: float = float("nan")
    mae_std: float = float("nan")
    qwk_mean: float = float("nan")
    qwk_std: float = float("nan")

    def selection_key(self) -> tuple[float, float]:
        """Menor MAE primero; empate: mayor QWK. NaN queda al final."""

        mae = math.inf if math.isnan(self.mae_mean) else round(self.mae_mean, 10)
        qwk = -math.inf if math.isnan(self.qwk_mean) else self.qwk_mean
        return mae, -qwk


@dataclass
class SearchResult:
    """Configuracion ganadora y puntajes de todas las candidatas."""

    best: CandidateScore
    candidates: list[CandidateScore] = field(default_factory=list)
    skipped: bool = False
    candidate_records: list[dict] = field(default_factory=list)


class GridSearch:
    """
    Recorre un grid de configuraciones sobre folds internos estratificados.

    Reglas del laboratorio:
    - Solo se usan datos del entrenamiento externo.
    - Se elige la configuracion con menor MAE interno promedio.
    - Empate: mayor QWK interno promedio.
    - Nunca se mira el fold externo de prueba.

    Si candidates es [{}], se evalua solo la configuracion base (--no-grid):
    sirve para leer el MAE interno sin buscar.
    """

    def __init__(
        self,
        algorithm: str,
        candidates: list[dict[str, Any]],
        base_model_config: ModelConfig,
        base_training_config: TrainingConfig,
        inner_folds: int,
        device: torch.device,
    ) -> None:
        if not candidates:
            raise ValueError("El grid debe tener al menos una configuracion.")
        self.algorithm = algorithm
        self.candidates = candidates
        self.base_model_config = base_model_config
        self.base_training_config = base_training_config
        self.inner_folds = inner_folds
        self.device = device

    def _candidate(self, overrides: dict[str, Any]) -> CandidateScore:
        model_config, training_config = apply_overrides(
            self.base_model_config, self.base_training_config, overrides
        )
        return CandidateScore(overrides, model_config, training_config)

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        num_classes: int,
        seed: int,
        init_seed: int | None = None,
        splits=None,
        skip: bool = False,
        records=None,
        context=None,
    ) -> SearchResult:
        """Evalua cada candidata y devuelve la ganadora."""

        splitter = StratifiedSplitter(self.inner_folds, random_state=seed)
        if skip or not splitter.can_split(y):
            # Clase demasiado rara: se usa la configuracion base sin buscar.
            from dataclasses import asdict
            import json
            best = self._candidate({})
            row = {'candidata': 0, 'configuracion': json.dumps({'model': asdict(best.model_config), 'training': asdict(best.training_config)}),
                   'resultados_internos': '[]', 'mae_mean': float('nan'), 'mae_std': float('nan'),
                   'qwk_mean': float('nan'), 'qwk_std': float('nan'), 'seleccionada': True}
            return SearchResult(best=best, skipped=True, candidate_records=[row])

        splits = splitter.split(y) if splits is None else splits
        init_seed = seed if init_seed is None else init_seed
        scores = []
        candidate_records = []
        for candidate_index, overrides in enumerate(self.candidates, start=1):
            candidate = self._candidate(overrides)
            mae_values, qwk_values, internal_results = [], [], []
            for fold_index, (train_idx, val_idx) in enumerate(splits):
                run_seed = init_seed + fold_index
                trainer = Trainer.build(
                    self.algorithm,
                    candidate.model_config,
                    candidate.training_config,
                    num_features=X.shape[1],
                    num_classes=num_classes,
                    device=self.device,
                    seed=run_seed,
                )
                if records is None:
                    result = trainer.fit_and_evaluate(X[train_idx], y[train_idx], X[val_idx], y[val_idx], seed=run_seed)
                else:
                    from src.training.records import recorded_fit
                    result = recorded_fit(records, trainer, X[train_idx], y[train_idx], X[val_idx], y[val_idx], run_seed,
                                          {**context, 'phase': 'inner', 'candidate': candidate_index, 'inner_fold': fold_index + 1})
                internal_results.append(result.metrics)
                mae_values.append(result.metrics["mae_ordinal"])
                qwk_values.append(result.metrics["qwk"])

            candidate.mae_mean = float(np.mean(mae_values))
            candidate.mae_std = float(np.std(mae_values))
            candidate.qwk_mean = float(np.mean(qwk_values))
            candidate.qwk_std = float(np.std(qwk_values))
            scores.append(candidate)
            from dataclasses import asdict
            import json
            candidate_records.append({'candidata': candidate_index,
                'configuracion': json.dumps({'model': asdict(candidate.model_config), 'training': asdict(candidate.training_config)}),
                'resultados_internos': json.dumps(internal_results), 'mae_mean': candidate.mae_mean,
                'mae_std': candidate.mae_std, 'qwk_mean': candidate.qwk_mean, 'qwk_std': candidate.qwk_std,
                'seleccionada': False})

        best = min(scores, key=CandidateScore.selection_key)
        candidate_records[scores.index(best)]["seleccionada"] = True
        return SearchResult(best=best, candidates=scores, candidate_records=candidate_records)
