from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

from src.data.dataset import CognitiveSequenceDataset
from src.evaluation.metrics import MetricsCalculator
from src.config import DEFAULT_RANDOM_SEED
from src.utils import resolve_device, set_seed

# ---------------------------------------------------------------------------
# Predictores constantes y regresión logística
# ---------------------------------------------------------------------------

class ConstantPredictor:
    """Predictor constante para líneas base (moda, mediana, distribución previa)."""
    def __init__(self, strategy='mode', classes=None):
        self.strategy = strategy
        self.classes = classes if classes is not None else np.array([0, 1, 2])
        self.K = len(self.classes)
        self.pred_class = None
        self.probabilities = None

    def fit(self, y_train):
        y_train = np.asarray(y_train)
        counts = np.array([np.sum(y_train == c) for c in self.classes], dtype=float)
        total = len(y_train)

        if total > 0:
            self.probabilities = counts / total
        else:
            self.probabilities = np.ones(self.K) / self.K

        if self.strategy == 'mode':
            max_count = np.max(counts)
            tied_classes = np.where(counts == max_count)[0]
            self.pred_class = tied_classes[0]
        elif self.strategy == 'median':
            cum_probs = np.cumsum(self.probabilities)
            self.pred_class = np.argmax(cum_probs >= 0.5)
        else:
            raise ValueError(f"Estrategia desconocida: {self.strategy}")
        return self

    def predict(self, X):
        n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]
        return np.full(n_samples, self.pred_class, dtype=np.int64)

    def predict_proba(self, X):
        n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]
        return np.tile(self.probabilities, (n_samples, 1))


class LogisticControlPredictor:
    """Control de Regresión Logística (Suma de bits o 15 entradas)."""
    def __init__(self, mode='features', C_candidates=None, random_state=DEFAULT_RANDOM_SEED):
        self.mode = mode
        self.C_candidates = C_candidates if C_candidates is not None else [0.01, 0.1, 1.0, 10.0]
        self.random_state = random_state
        self.best_C = None
        self.model = None
        self.classes_ = None

    def _transform_X(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.mode == 'sum':
            return X.sum(axis=1, keepdims=True)
        elif self.mode == 'features':
            return X
        else:
            raise ValueError(f"Modo desconocido: {self.mode}")

    def fit_grid_search(self, X_train, y_train, X_val, y_val, classes):
        self.classes_ = np.asarray(classes)
        X_tr = self._transform_X(X_train)
        X_va = self._transform_X(X_val)

        best_score_mae = float('inf')
        best_score_qwk = float('-inf')
        best_C = self.C_candidates[0]

        for C in self.C_candidates:
            clf = LogisticRegression(
                C=C, penalty='l2', solver='lbfgs', max_iter=2000, random_state=self.random_state
            )
            clf.fit(X_tr, y_train)
            preds = clf.predict(X_va)

            mae = MetricsCalculator.mae_ordinal(y_val, preds)
            qwk = MetricsCalculator.quadratic_weighted_kappa(y_val, preds)

            if mae < best_score_mae or (np.isclose(mae, best_score_mae) and qwk > best_score_qwk):
                best_score_mae = mae
                best_score_qwk = qwk
                best_C = C

        self.best_C = best_C
        self.model = LogisticRegression(
            C=self.best_C, penalty='l2', solver='lbfgs', max_iter=2000, random_state=self.random_state
        )
        self.model.fit(X_tr, y_train)
        return self

    def predict(self, X):
        return self.model.predict(self._transform_X(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self._transform_X(X))

# ---------------------------------------------------------------------------
# MLP sencilla de referencia (Lab01)
# ---------------------------------------------------------------------------

class ShallowMultiClassNet(nn.Module):
    """Red neuronal poco profunda para clasificación multiclase (Lab01)."""
    def __init__(self, input_dim: int = 15, hidden_dim: int = 32, dropout: float = 0.15, output_dim: int = 3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(inputs)
        hidden = self.relu(hidden)
        hidden = self.dropout(hidden)
        logits = self.fc2(hidden)
        return logits


class MLPControlPredictor:
    """Control MLP sencilla del Lab01."""
    def __init__(self, candidates=None, epochs=20, batch_size=32, lr=1e-3, weight_decay=1e-4, seed=DEFAULT_RANDOM_SEED):
        self.candidates = candidates if candidates is not None else [
            {"hidden_dim": 32, "dropout": 0.15},
            {"hidden_dim": 32, "dropout": 0.30},
            {"hidden_dim": 64, "dropout": 0.15},
            {"hidden_dim": 64, "dropout": 0.30},
        ]
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.best_config = None
        self.best_model = None
        self.num_classes = None

    def _train_model(self, model, X_tr, y_tr, device):
        set_seed(self.seed)

        y_tr_long = np.asarray(y_tr, dtype=np.int64)
        dataset = TensorDataset(torch.as_tensor(X_tr, dtype=torch.float32), torch.as_tensor(y_tr_long))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        model.train()
        for _ in range(self.epochs):
            for inputs, targets in loader:
                inputs = inputs.to(device)
                targets = targets.to(device).long()

                optimizer.zero_grad()
                logits = model(inputs)

                # CORRECCIÓN CLAVE: Eliminar dimensiones extra (ej. transformar [16, 1, 3] a [16, 3])
                if logits.ndim > 2:
                    logits = logits.squeeze(1) # o logits.reshape(-1, self.num_classes)

                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

        return model

    def fit_grid_search(self, X_train, y_train, X_val, y_val, classes):
        self.num_classes = len(classes)
        device = torch.device("cpu")

        best_score_mae = float('inf')
        best_score_qwk = float('-inf')
        best_config = self.candidates[0]

        # Búsqueda interna sobre las cuatro candidatas de hiperparámetros.
        for config in self.candidates:
            set_seed(self.seed)
            model = ShallowMultiClassNet(
                input_dim=15,
                hidden_dim=config["hidden_dim"],
                dropout=config["dropout"],
                output_dim=self.num_classes
            ).to(device)

            model = self._train_model(model, X_train, y_train, device)

            # Evaluación en validación interna
            model.eval()
            val_dataset = TensorDataset(torch.as_tensor(X_val, dtype=torch.float32), torch.zeros(len(X_val), dtype=torch.long))
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

            preds_list = []
            with torch.no_grad():
                for inputs, _ in val_loader:
                    logits = model(inputs.to(device))
                    if logits.ndim > 2:
                        logits = logits.squeeze(1)
                    # Obtener estrictamente los índices de clase enteros (B,)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    preds_list.append(preds)
            val_preds = np.concatenate(preds_list)

            # Ahora ambos (y_val y val_preds) tendrán forma (20,)
            mae = MetricsCalculator.mae_ordinal(y_val, val_preds)
            qwk = MetricsCalculator.quadratic_weighted_kappa(y_val, val_preds)

            # Criterio del laboratorio: menor MAE, empate por mayor QWK
            if mae < best_score_mae or (np.isclose(mae, best_score_mae) and qwk > best_score_qwk):
                best_score_mae = mae
                best_score_qwk = qwk
                best_config = config

        self.best_config = best_config

        # Reentrenamiento final con la mejor configuración seleccionada
        set_seed(self.seed)
        self.best_model = ShallowMultiClassNet(
            input_dim=15,
            hidden_dim=self.best_config["hidden_dim"],
            dropout=self.best_config["dropout"],
            output_dim=self.num_classes
        ).to(device)
        self.best_model = self._train_model(self.best_model, X_train, y_train, device)
        return self

    @torch.no_grad()
    def predict(self, X):
        device = next(self.best_model.parameters()).device
        self.best_model.eval()
        dataset = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.zeros(len(X), dtype=torch.long))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        preds_list = []
        for inputs, _ in loader:
            logits = self.best_model(inputs.to(device))
            if logits.ndim > 2:
                logits = logits.squeeze(1)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds_list.append(preds)
        return np.concatenate(preds_list)

    @torch.no_grad()
    def predict_proba(self, X):
        device = next(self.best_model.parameters()).device
        self.best_model.eval()
        dataset = TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.zeros(len(X), dtype=torch.long))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        probs_list = []
        softmax = nn.Softmax(dim=1)
        for inputs, _ in loader:
            logits = self.best_model(inputs.to(device))
            if logits.ndim > 2:
                logits = logits.squeeze(1)
            probs = softmax(logits).cpu().numpy()
            probs_list.append(probs)
        return np.concatenate(probs_list)


# ---------------------------------------------------------------------------
# Boosting tabular (HistGradientBoosting)
# ---------------------------------------------------------------------------

class HistGradientBoostingControlPredictor:
    """Control HistGradientBoosting con candidatas de max_leaf_nodes (3, 7, 15, 31)."""
    def __init__(self, leaf_candidates=None, random_state=DEFAULT_RANDOM_SEED):
        self.leaf_candidates = leaf_candidates if leaf_candidates is not None else [3, 7, 15, 31]
        self.random_state = random_state
        self.best_leaves = None
        self.model = None

    def fit_grid_search(self, X_train, y_train, X_val, y_val, classes):
        best_score_mae = float('inf')
        best_score_qwk = float('-inf')
        best_leaves = self.leaf_candidates[0]

        for max_leaves in self.leaf_candidates:
            clf = HistGradientBoostingClassifier(
                max_leaf_nodes=max_leaves,
                max_iter=100,
                learning_rate=0.1,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=self.random_state
            )
            clf.fit(X_train, y_train)
            preds = clf.predict(X_val)

            mae = MetricsCalculator.mae_ordinal(y_val, preds)
            qwk = MetricsCalculator.quadratic_weighted_kappa(y_val, preds)

            if mae < best_score_mae or (np.isclose(mae, best_score_mae) and qwk > best_score_qwk):
                best_score_mae = mae
                best_score_qwk = qwk
                best_leaves = max_leaves

        self.best_leaves = best_leaves
        self.model = HistGradientBoostingClassifier(
            max_leaf_nodes=self.best_leaves,
            max_iter=100,
            learning_rate=0.1,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=self.random_state
        )
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)
