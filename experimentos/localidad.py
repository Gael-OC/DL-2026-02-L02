"""
Modelos y gestión de órdenes para el estudio de localidad y vecindad.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.models.conv_softmax import Conv1DSoftmaxNet


class MLPCapacityNet(nn.Module):
    """
    MLP de capacidad próxima (15 -> 128 -> 115 -> 3).
    ReLU tras cada capa oculta y dropout (0.15) antes de la capa de salida.
    """
    def __init__(self, input_dim: int = 15, num_classes: int = 3, dropout: float = 0.15) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 115)
        self.relu2 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc3 = nn.Linear(115, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # Asegurar forma (B, 15) si llega como (B, 1, 15)
        if inputs.ndim == 3 and inputs.size(1) == 1:
            inputs = inputs.squeeze(1)
        x = self.relu1(self.fc1(inputs))
        x = self.relu2(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)
        return logits


def build_locality_model(model_name: str, num_features: int = 15, num_classes: int = 3) -> nn.Module:
    """Fábrica para los modelos del panel de localidad."""
    if model_name == "cnn_k3":
        return Conv1DSoftmaxNet(
            num_features=num_features,
            num_classes=num_classes,
            conv_channels=(16, 32),
            kernel_size=3,
            hidden_dim=32,
            dropout=0.15,
        )
    elif model_name == "cnn_k1":
        return Conv1DSoftmaxNet(
            num_features=num_features,
            num_classes=num_classes,
            conv_channels=(16, 32),
            kernel_size=1,
            hidden_dim=32,
            dropout=0.15,
        )
    elif model_name == "mlp_capacity":
        return MLPCapacityNet(
            input_dim=num_features,
            num_classes=num_classes,
            dropout=0.15,
        )
    else:
        raise ValueError(f"Modelo de localidad desconocido: {model_name}")


def get_feature_permutation(order_type: str) -> list[int]:
    """Devuelve los índices exactos según el orden especificado en el protocolo."""
    if order_type == "original":
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    elif order_type == "blocks":
        # Bloques 0:4, 4:8 y 8:15 permutados con default_rng(2026)
        return [3, 1, 0, 2, 4, 5, 6, 7, 9, 13, 14, 12, 8, 10, 11]
    elif order_type == "global":
        # Permutación global con default_rng(2027).permutation(15)
        return [4, 2, 13, 5, 9, 10, 7, 14, 3, 0, 11, 1, 8, 6, 12]
    else:
        raise ValueError(f"Tipo de orden desconocido: {order_type}")


def apply_order_to_features(X: np.ndarray | torch.Tensor, order_type: str) -> np.ndarray | torch.Tensor:
    """Reordena las columnas de la matriz de características X según el orden indicado."""
    indices = get_feature_permutation(order_type)
    if isinstance(X, np.ndarray):
        return X[:, indices]
    elif isinstance(X, torch.Tensor):
        return X[:, indices]
    else:
        raise TypeError("X debe ser un numpy.ndarray o torch.Tensor")
