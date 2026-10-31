"""Metricas multiclase, ordinales y probabilisticas del laboratorio."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
)

METRIC_KEYS = [
    "accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro",
    "mae_ordinal", "qwk", "accuracy_pm1", "errores_graves",
]


class MetricsCalculator:
    """Metricas sobre indices ordinales 0..K-1; num_classes fija el alfabeto."""

    @staticmethod
    def _as_labels(y_true, y_pred):
        y_true = np.asarray(y_true).ravel()
        y_pred = np.asarray(y_pred).ravel()
        if y_true.shape != y_pred.shape or not len(y_true):
            raise ValueError("y_true e y_pred deben tener igual forma y no estar vacios.")
        if (not np.issubdtype(y_true.dtype, np.integer)
                or not np.issubdtype(y_pred.dtype, np.integer)
                or np.any(y_true < 0) or np.any(y_pred < 0)):
            raise ValueError("Las etiquetas deben ser indices enteros no negativos.")
        return y_true, y_pred

    @classmethod
    def _labels(cls, y_true, y_pred, num_classes=None):
        y_true, y_pred = cls._as_labels(y_true, y_pred)
        # Solo por compatibilidad; los folds deben pasar el K del objetivo.
        if num_classes is None:
            num_classes = int(max(y_true.max(), y_pred.max())) + 1
        if not isinstance(num_classes, (int, np.integer)) or num_classes < 1:
            raise ValueError("num_classes debe ser un entero positivo.")
        if np.any(y_true >= num_classes) or np.any(y_pred >= num_classes):
            raise ValueError("Etiqueta fuera del alfabeto 0..K-1.")
        return y_true, y_pred, np.arange(num_classes)

    @classmethod
    def accuracy(cls, y_true, y_pred):
        y_true, y_pred = cls._as_labels(y_true, y_pred)
        return float(np.mean(y_true == y_pred))

    @classmethod
    def balanced_accuracy(cls, y_true, y_pred):
        """Recall medio solo sobre clases con soporte real."""
        y_true, y_pred = cls._as_labels(y_true, y_pred)
        # Evita advertencias de sklearn cuando se predice una clase sin soporte.
        labels = np.unique(y_true)
        return float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))

    @classmethod
    def precision_macro(cls, y_true, y_pred, num_classes=None):
        y_true, y_pred, labels = cls._labels(y_true, y_pred, num_classes)
        return float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))

    @classmethod
    def recall_macro(cls, y_true, y_pred, num_classes=None):
        y_true, y_pred, labels = cls._labels(y_true, y_pred, num_classes)
        return float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))

    @classmethod
    def f1_macro(cls, y_true, y_pred, num_classes=None):
        y_true, y_pred, labels = cls._labels(y_true, y_pred, num_classes)
        return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))

    @classmethod
    def mae_ordinal(cls, y_true, y_pred):
        y_true, y_pred = cls._as_labels(y_true, y_pred)
        return float(mean_absolute_error(y_true, y_pred))

    @classmethod
    def quadratic_weighted_kappa(cls, y_true, y_pred, num_classes=None):
        """QWK con distancias de indices originales; NaN si el desacuerdo esperado es cero."""
        y_true, y_pred, labels = cls._labels(y_true, y_pred, num_classes)
        observed = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
        expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / len(y_true)
        weights = (labels[:, None] - labels[None, :]) ** 2
        denominator = np.sum(weights * expected)
        return float(1 - np.sum(weights * observed) / denominator) if denominator else float("nan")

    @classmethod
    def accuracy_pm1(cls, y_true, y_pred):
        y_true, y_pred = cls._as_labels(y_true, y_pred)
        return float(np.mean(np.abs(y_true - y_pred) <= 1))

    @classmethod
    def errores_graves(cls, y_true, y_pred):
        y_true, y_pred = cls._as_labels(y_true, y_pred)
        return float(np.mean(np.abs(y_true - y_pred) >= 2))

    @classmethod
    def compute(cls, y_true, y_pred, num_classes=None):
        y_true, y_pred, labels = cls._labels(y_true, y_pred, num_classes)
        k = len(labels)
        return {
            "accuracy": cls.accuracy(y_true, y_pred),
            "balanced_accuracy": cls.balanced_accuracy(y_true, y_pred),
            "precision_macro": cls.precision_macro(y_true, y_pred, k),
            "recall_macro": cls.recall_macro(y_true, y_pred, k),
            "f1_macro": cls.f1_macro(y_true, y_pred, k),
            "mae_ordinal": cls.mae_ordinal(y_true, y_pred),
            "qwk": cls.quadratic_weighted_kappa(y_true, y_pred, k),
            "accuracy_pm1": cls.accuracy_pm1(y_true, y_pred),
            "errores_graves": cls.errores_graves(y_true, y_pred),
        }

    @classmethod
    def _probabilities(cls, y_true, probabilities):
        probabilities = np.asarray(probabilities)
        if (probabilities.ndim != 2 or probabilities.shape[1] < 2
                or not np.issubdtype(probabilities.dtype, np.floating)):
            raise ValueError("Las probabilidades deben ser una matriz flotante (N,K), K >= 2.")
        y_true, _, _ = cls._labels(y_true, y_true, probabilities.shape[1])
        if (len(y_true) != len(probabilities) or not np.isfinite(probabilities).all()
                or np.any(probabilities < 0) or np.any(probabilities > 1)
                or not np.allclose(probabilities.sum(axis=1), 1, rtol=1e-6, atol=1e-7)):
            raise ValueError("Probabilidades invalidas: deben ser finitas, no negativas y sumar uno.")
        return y_true, probabilities

    @classmethod
    def nll(cls, y_true, probabilities):
        """NLL media; solo el logaritmo se protege con el menor positivo del dtype."""
        y_true, probabilities = cls._probabilities(y_true, probabilities)
        selected = probabilities[np.arange(len(y_true)), y_true]
        smallest = np.nextafter(probabilities.dtype.type(0), probabilities.dtype.type(1))
        return float(-np.log(np.maximum(selected, smallest)).mean())

    @classmethod
    def rps(cls, y_true, probabilities):
        """RPS medio normalizado por K-1, incluidas las tareas binarias."""
        y_true, probabilities = cls._probabilities(y_true, probabilities)
        cumulative = np.cumsum(probabilities, axis=1)[:, :-1]
        observed = y_true[:, None] <= np.arange(probabilities.shape[1] - 1)
        return float(np.mean((cumulative - observed) ** 2))

    @classmethod
    def classification_report(cls, y_true, y_pred, class_names=None, num_classes=None):
        if class_names is not None:
            if num_classes is not None and num_classes != len(class_names):
                raise ValueError("class_names debe contener K nombres.")
            num_classes = len(class_names)
        y_true, y_pred, labels = cls._labels(y_true, y_pred, num_classes)
        return classification_report(
            y_true, y_pred, labels=labels,
            target_names=[str(name) for name in class_names] if class_names is not None else None,
            zero_division=0, digits=4,
        )

    @classmethod
    def confusion_matrix(cls, y_true, y_pred, num_classes=None):
        """Filas = real, columnas = predicho; conserva K clases."""
        y_true, y_pred, labels = cls._labels(y_true, y_pred, num_classes)
        return confusion_matrix(y_true, y_pred, labels=labels)
