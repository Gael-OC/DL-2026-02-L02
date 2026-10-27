import numpy as np
from sklearn.linear_model import LogisticRegression
from src.evaluation.metrics import MetricsCalculator  # O la ruta correspondiente en tu proyecto para calcular MAE/QWK

class ConstantPredictor:
    """
    Predictor constante para líneas base (moda, mediana, distribución previa).
    """
    def __init__(self, strategy='mode', classes=None):
        self.strategy = strategy
        self.classes = classes if classes is not None else np.array([0, 1, 2])
        self.K = len(self.classes)
        self.pred_class = None
        self.probabilities = None

    def fit(self, y_train):
        """
        Ajusta el predictor utilizando exclusivamente el conjunto de entrenamiento.
        """
        y_train = np.asarray(y_train)
        
        # Calcular frecuencias absolutas para cada clase del alfabeto completo
        counts = np.array([np.sum(y_train == c) for c in self.classes], dtype=float)
        total = len(y_train)
        
        # Probabilidades empíricas (conteo / total)
        if total > 0:
            self.probabilities = counts / total
        else:
            self.probabilities = np.ones(self.K) / self.K

        if self.strategy == 'mode':
            # Moda con desempate hacia el índice menor (np.argmax selecciona el primer índice en empates)
            max_count = np.max(counts)
            tied_classes = np.where(counts == max_count)[0]
            self.pred_class = tied_classes[0]
            
        elif self.strategy == 'median':
            # Mediana basada en la distribución acumulada empírica
            cum_probs = np.cumsum(self.probabilities)
            self.pred_class = np.argmax(cum_probs >= 0.5)
            
        else:
            raise ValueError(f"Estrategia desconocida: {self.strategy}")
            
        return self

    def predict(self, X):
        """
        Retorna la predicción constante para cada muestra de X.
        """
        n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]
        return np.full(n_samples, self.pred_class, dtype=np.int64)

    def predict_proba(self, X):
        """
        Retorna la distribución de probabilidad constante (N, K) alineada con el alfabeto.
        """
        n_samples = len(X) if hasattr(X, '__len__') else X.shape[0]
        return np.tile(self.probabilities, (n_samples, 1))

class LogisticControlPredictor:
    """
    Control de Regresión Logística para la ficha J1.2.
    Soporta modo 'sum' (suma de bits como 1 sola variable) o 'features' (15 entradas).
    """
    def __init__(self, mode='features', C_candidates=None, random_state=42):
        self.mode = mode
        self.C_candidates = C_candidates if C_candidates is not None else [0.01, 0.1, 1.0, 10.0]
        self.random_state = random_state
        self.best_C = None
        self.model = None
        self.classes_ = None

    def _transform_X(self, X):
        X = np.asarray(X, dtype=np.float32)
        if self.mode == 'sum':
            # Suma de bits a lo largo de las columnas -> forma (N, 1)
            return X.sum(axis=1, keepdims=True)
        elif self.mode == 'features':
            # Las 15 entradas directas sin estandarización
            return X
        else:
            raise ValueError(f"Modo desconocido para LogisticControlPredictor: {self.mode}")

    def fit_grid_search(self, X_train, y_train, X_val, y_val, classes):
        """
        Realiza la búsqueda de la mejor candidata C usando validación interna (train/val del fold).
        Criterio de selección: menor MAE, empate por mayor QWK.
        """
        self.classes_ = np.asarray(classes)
        X_tr = self._transform_X(X_train)
        X_va = self._transform_X(X_val)
        
        best_score_mae = float('inf')
        best_score_qwk = float('-inf')
        best_C = self.C_candidates[0]
        
        metrics_calc = MetricsCalculator()

        for C in self.C_candidates:
            # Configuración fija exigida por el protocolo
            clf = LogisticRegression(
                C=C,
                penalty='l2',
                solver='lbfgs',
                max_iter=2000,
                random_state=self.random_state
            )
            clf.fit(X_tr, y_train)
            
            # Predicción en validación interna
            preds = clf.predict(X_va)
            
            # Calcular métricas internas para selección (menor MAE, mayor QWK)
            mae = metrics_calc.mae_ordinal(y_val, preds)
            qwk = metrics_calc.quadratic_weighted_kappa(y_val, preds)
            
            # Criterio de selección: menor MAE, ante empate mayor QWK
            if mae < best_score_mae or (np.isclose(mae, best_score_mae) and qwk > best_score_qwk):
                best_score_mae = mae
                best_score_qwk = qwk
                best_C = C

        self.best_C = best_C
        
        # Refit final combinando train y val (o reentrenando con todo el train externo según flujo)
        # Aquí reentrenamos con X_train completo para el despliegue final en test externo
        self.model = LogisticRegression(
            C=self.best_C,
            penalty='l2',
            solver='lbfgs',
            max_iter=2000,
            random_state=self.random_state
        )
        self.model.fit(X_tr, y_train)
        return self

    def predict(self, X):
        X_t = self._transform_X(X)
        return self.model.predict(X_t)

    def predict_proba(self, X):
        X_t = self._transform_X(X)
        return self.model.predict_proba(X_t)