"""Validacion cruzada anidada: el loop interno elige, el externo reporta."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from src.config import ModelConfig, TrainingConfig, describe_configs
from src.data.preprocessing import ExperimentData, StratifiedSplitter
from src.evaluation.metrics import METRIC_KEYS, MetricsCalculator
from src.training.search import CandidateScore, GridSearch
from src.training.trainer import Trainer


@dataclass
class OuterFoldResult:
    """Resultado de un fold externo."""

    outer_fold: int
    selected: CandidateScore
    inner_skipped: bool
    metrics: dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray
    final_train_loss: float
    row_ids: np.ndarray | None = None
    logits: np.ndarray | None = None
    probabilities: np.ndarray | None = None


@dataclass
class ExperimentResult:
    """
    Resultado de un experimento completo (algoritmo x objetivo).

    summary contiene mean_<metrica> y std_<metrica> sobre los folds externos:
    es el unico numero que se reporta en el informe.
    """

    algorithm: str
    target_name: str
    classes: list[int]
    num_samples: int
    num_features: int
    outer_folds: int
    inner_folds: int
    use_grid: bool
    device: str
    epochs: int
    folds: list[OuterFoldResult] = field(default_factory=list)
    summary: dict[str, float] = field(default_factory=dict)

    @property
    def last_fold(self) -> OuterFoldResult:
        return self.folds[-1]

    def last_fold_report(self) -> str:
        return MetricsCalculator.classification_report(
            self.last_fold.y_true, self.last_fold.y_pred, class_names=self.classes
        )

    def last_fold_confusion(self) -> np.ndarray:
        return MetricsCalculator.confusion_matrix(
            self.last_fold.y_true, self.last_fold.y_pred, num_classes=len(self.classes)
        )

    def most_selected(self) -> CandidateScore:
        """Configuracion elegida con mas frecuencia entre los folds externos."""

        labels = [
            describe_configs(f.selected.model_config, f.selected.training_config)
            for f in self.folds
        ]
        winner, _ = Counter(labels).most_common(1)[0]
        return self.folds[labels.index(winner)].selected


class NestedCrossValidator:
    """
    Validacion anidada outer x inner con busqueda de hiperparametros.

    Para cada fold externo:
    1. GridSearch sobre folds internos del entrenamiento externo.
    2. Reentrenar la configuracion ganadora con todo el entrenamiento externo.
    3. Evaluar una sola vez en el fold externo de prueba.
    """

    def __init__(
        self,
        algorithm: str,
        base_model_config: ModelConfig,
        base_training_config: TrainingConfig,
        outer_folds: int,
        inner_folds: int,
        grid: list[dict[str, Any]] | None,
        seed: int,
        device: torch.device,
        verbose: bool = True,
        split_seed: int | None = None,
        init_seed: int | None = None,
        split_path=None,
        records=None,
    ) -> None:
        self.algorithm = algorithm
        self.base_model_config = base_model_config
        self.base_training_config = base_training_config
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.use_grid = bool(grid)
        self.candidates = list(grid) if grid else [{}]
        self.split_seed = seed if split_seed is None else split_seed
        self.init_seed = seed if init_seed is None else init_seed
        self.split_path = split_path
        self.records = records
        self.device = device
        self.verbose = verbose

    def _search(self) -> GridSearch:
        return GridSearch(
            algorithm=self.algorithm,
            candidates=self.candidates,
            base_model_config=self.base_model_config,
            base_training_config=self.base_training_config,
            inner_folds=self.inner_folds,
            device=self.device,
        )

    def run(self, data: ExperimentData) -> ExperimentResult:
        """Ejecuta la validacion anidada sobre un objetivo."""

        X, y = data.X, data.y
        outer_splits = StratifiedSplitter(self.outer_folds, self.split_seed).split(y)
        splits_to_save = {"split_seed": np.array(self.split_seed), "target": np.array(data.target_name),
                          "labels": y, "classes": np.asarray(data.classes)}
        inner_splits = []
        for outer_index, (train_idx, test_idx) in enumerate(outer_splits, start=1):
            prefix = f"outer_{outer_index}"
            splits_to_save[f"{prefix}_train"] = train_idx
            splits_to_save[f"{prefix}_test"] = test_idx
            splitter = StratifiedSplitter(self.inner_folds, self.split_seed + outer_index * 100)
            local = [] if data.target_name == "GDS" or not splitter.can_split(y[train_idx]) else splitter.split(y[train_idx])
            inner_splits.append(local)
            splits_to_save[f"{prefix}_inner_skipped"] = np.array(not bool(local))
            for inner_index, (inner_train, inner_val) in enumerate(local, start=1):
                splits_to_save[f"{prefix}_inner_{inner_index}_train"] = train_idx[inner_train]
                splits_to_save[f"{prefix}_inner_{inner_index}_val"] = train_idx[inner_val]
        if self.split_path is not None:
            from pathlib import Path
            path = Path(self.split_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                with np.load(path, allow_pickle=False) as saved:
                    if set(saved.files) != set(splits_to_save) or any(
                        not np.array_equal(saved[key], value) for key, value in splits_to_save.items()
                    ):
                        raise ValueError(f"Las particiones existentes no coinciden: {path}. Use otra carpeta.")
            else:
                np.savez_compressed(path, **splits_to_save)
        search = self._search()
        result = ExperimentResult(
            algorithm=self.algorithm,
            target_name=data.target_name,
            classes=data.classes,
            num_samples=len(y),
            num_features=data.num_features,
            outer_folds=self.outer_folds,
            inner_folds=self.inner_folds,
            use_grid=self.use_grid,
            device=str(self.device),
            epochs=self.base_training_config.epochs,
        )

        for outer_index, (train_idx, test_idx) in enumerate(outer_splits, start=1):
            variant = ("coral_weighted" if self.algorithm == "conv1d_coral" and self.base_training_config.use_class_weights
                       else "coral" if self.algorithm == "conv1d_coral" else "softmax")
            if self.records is not None:
                saved = self.records.load_fold(data.target_name, variant, outer_index)
                if saved is not None:
                    result.folds.append(saved)
                    continue
            context = {'target': data.target_name, 'variant': variant, 'outer_fold': outer_index}
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            search_result = search.run(
                X_train,
                y_train,
                num_classes=data.num_classes,
                seed=self.split_seed + outer_index * 100,
                init_seed=self.init_seed + outer_index * 100,
                splits=inner_splits[outer_index - 1],
                skip=not inner_splits[outer_index - 1],
                records=self.records,
                context=context,
            )
            if search_result.skipped:
                if data.target_name == "GDS":
                    print(f"GDS fold externo {outer_index}: busqueda interna omitida por protocolo.")
                else:
                    print(
                    f"Aviso: el fold externo {outer_index} de {data.target_name} no "
                    f"admite {self.inner_folds} folds internos estratificados. "
                    "Se usa la configuracion base sin busqueda."
                )

            selected = search_result.best
            final_seed = self.init_seed + outer_index * 1000
            trainer = Trainer.build(
                self.algorithm,
                selected.model_config,
                selected.training_config,
                num_features=data.num_features,
                num_classes=data.num_classes,
                device=self.device,
                seed=final_seed,
            )
            if self.records is None:
                fit = trainer.fit_and_evaluate(X_train, y_train, X_test, y_test, final_seed)
            else:
                from src.training.records import recorded_fit
                fit = recorded_fit(self.records, trainer, X_train, y_train, X_test, y_test, final_seed,
                                   {**context, 'phase': 'refit', 'candidate': None})
            logits = probabilities = None
            if self.records is not None:
                logits = trainer.predict_logits(X_test)
                with torch.no_grad():
                    probabilities = trainer.task.probabilities(torch.from_numpy(logits).to(self.device)).cpu().numpy()
                fit.metrics['nll'] = MetricsCalculator.nll(y_test, probabilities)
                fit.metrics['rps'] = MetricsCalculator.rps(y_test, probabilities)
            result.folds.append(
                OuterFoldResult(
                    outer_fold=outer_index,
                    selected=selected,
                    inner_skipped=search_result.skipped,
                    metrics=fit.metrics,
                    y_true=fit.y_true,
                    y_pred=fit.y_pred,
                    final_train_loss=fit.final_train_loss,
                    row_ids=test_idx if self.records is not None else None,
                    logits=logits, probabilities=probabilities,
                )
            )

            if self.records is not None:
                self.records.save_fold(data.target_name, variant, outer_index, result.folds[-1], search_result.candidate_records)

            if self.verbose:
                print(
                    f"  fold {outer_index}/{self.outer_folds}: "
                    f"{describe_configs(selected.model_config, selected.training_config)} "
                    f"| MAE interno={selected.mae_mean:.4f} "
                    f"| test F1_m={fit.metrics['f1_macro']:.4f} "
                    f"MAE={fit.metrics['mae_ordinal']:.4f}"
                )

        for metric in METRIC_KEYS:
            values = np.asarray([f.metrics[metric] for f in result.folds], dtype=float)
            valid = values[np.isfinite(values)]
            result.summary[f"valid_{metric}"] = len(valid)
            result.summary[f"mean_{metric}"] = float(valid.mean()) if len(valid) else float("nan")
            result.summary[f"std_{metric}"] = float(valid.std()) if len(valid) else float("nan")
        return result
