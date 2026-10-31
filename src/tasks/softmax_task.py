"""Tarea multiclase categorica: CrossEntropyLoss + argmax."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.config import TrainingConfig
from src.tasks.base import ClassificationTask, LossFunction


class SoftmaxTask(ClassificationTask):
    """
    Clasificacion multiclase que ignora el orden de las clases.

    Formas:
        logits: (B, K)
        labels: (B,) int64 en 0 .. K-1
    """

    name = "softmax"

    def build_criterion(
        self,
        y_train: np.ndarray,
        config: TrainingConfig,
        device: torch.device,
    ) -> LossFunction:
        return nn.CrossEntropyLoss()

    def probabilities(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 2 or logits.shape[1] != self.num_classes or not torch.isfinite(logits).all():
            raise ValueError("Se requieren logits finitos con forma (B,K).")
        return torch.softmax(logits, dim=1)

    @torch.no_grad()
    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=1)
