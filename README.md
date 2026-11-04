# DL-2026-02-L02

Avance del 04/11: corrida central completa y auditada (981 ajustes),
[seis tablas comparativas](results/resumen/central/comparacion.md) y
[borrador del informe](docs/informe.md). Las salidas por fila permanecen locales.

**Controles R2 históricos:** las cifras siguientes requieren revisión antes de
integrarlas a la comparación final. El ejecutor actual usa un solo fold interno,
omite el refit externo completo y entrena la MLP diez épocas; la llamada a las
constantes también mezcla etiquetas originales con índices. Se conserva el
trabajo de Johan y se registra esta dependencia para H5.2.

## Tabla Consolidada: Rendimiento OOF promedio de Controles en $GDS\_R2$
(resultados/controles/metricas_controles_GDS_R2.csv)

| Modelo de Control    | Estrategia / Configuración         | Accuracy (↑)  | Balanced Acc (↑) | MAE Ordinal (↓) | QWK (↑)       | Errores Graves (↓) |
|----------------------|------------------------------------|---------------|------------------|-----------------|---------------|--------------------|
| Constante (Moda)     | Moda global (Clase mayoritaria)    | 0.5800±0.0025 |         0.3333   | 0.5737±0.0044   |             0 | 0.1537±0.0023      |
| Constante (Mediana)  | Mediana de la distribución         | 0.5800±0.0025 |         0.3333   | 0.5737±0.0044   |             0 | 0.1537±0.0023      |
| Logística (Suma)     | Regresión Logística (Suma de bits) | 0.7363±0.0402 | 0.6486±0.0513    | 0.2816±0.0426   | 0.7023±0.0482 | 0.0179±0.0032      |
| Logística (15 feats) | Regresión Logística (C por grid)   | 0.7256±0.0422 | 0.6425±0.0453    | 0.2914±0.0444   | 0.6908±0.0545 | 0.0170±0.0037      |
| MLP Sencilla (Ref.)  | Red Neuronal Lab01 (15→H→K)        | 0.7238±0.0330 | 0.6203±0.0284    | 0.2985±0.0416   | 0.6633±0.0609 | 0.0224±0.0100      |
| Hist. Grad. Boosting | Árboles con gradiente potenciado   | 0.7211±0.0405 | 0.6417±0.0424    | 0.2941±0.0442   | 0.6928±0.0547 | 0.0152±0.0040      |

## Análisis y Conclusiones del Comportamiento de los Controles
**1. Establecimiento de la Cota Inferior (Predictores Constantes):**
Los modelos basados en moda y mediana alcanzan una exactitud aproximada del $58\%$, pero con un QWK de $0.0$ y un MAE ordinal elevado ($\approx 0.57$), lo que evidencia que una estrategia ingenua es totalmente incapaz de capturar la estructura ordenada de las clases en $GDS\_R2$, incurriendo en tasas de errores graves cercanas al $15.3\%$. Esto confirma que cualquier arquitectura neuronal profunda posterior tiene una valla clara que superar obligatoriamente.

<br>

**2. Competitividad Sorprendente del Modelo Lineal Simple (logistica_suma):**
Resulta sumamente notable que la Regresión Logística operando sobre la simple suma de bits de las entradas alcance el mejor desempeño general del bloque de control, con un QWK de $0.7023$ y un MAE de $0.2816$. Esto demuestra que la acumulación lineal de características cognitivas posee una correlación directa y muy fuerte con el objetivo $GDS\_R2$.

<br>

**3. Paridad entre Modelos Lineales, Neuronales y Tabulares:**
La Regresión Logística sobre las 15 características completas, la MLP sencilla de referencia y el HistGradientBoosting se mueven en un rango de rendimiento muy cerrado (QWK entre $0.663$ y $0.693$). El HistGradientBoosting registra menos errores graves que la MLP (1.52\% frente a 2.24\%); los tres modelos mantienen resultados cercanos en el resto de las métricas.

Organización del equipo: [plan, tareas y registro de avances](PLAN_LABORATORIO.md).

Proyecto base del **Laboratorio 02 de Deep Learning** (segundo semestre 2026):
redes con una **capa convolucional 1D** para predecir la etapa de deterioro
cognitivo (escala GDS) a partir de 15 respuestas binarias de un test de
orientacion.

Es la continuacion del Laboratorio 01: se mantiene el dataset, los seis
experimentos (`GDS`, `GDS_R1` ... `GDS_R5`), la validacion cruzada anidada y
las metricas multiclase y ordinales. El cambio principal es la arquitectura:
cada fila de 15 atributos se trata como una secuencia de largo 15 y se procesa
con `nn.Conv1d` antes del MLP.

## Experimentos

| Algoritmo | Arquitectura | Perdida | Prediccion | Estado |
|---|---|---|---|---|
| `conv1d_softmax` | Conv1D + MLP + `Linear(K)` | `CrossEntropyLoss` | `argmax` | Implementado |
| `conv1d_coral` | Conv1D + MLP + `CoralLayer(K-1)` | BCE sobre umbrales | conteo de umbrales | Implementado; integración corta comprobada |

Ambos comparten el mismo tronco (`Conv1DBackbone`). Solo cambia la cabeza y la
estrategia de entrenamiento (`ClassificationTask`).

## Que esta implementado

- Carga de datos `csv` / `sav` y recodificacion del objetivo a `0 .. K-1`.
- `Dataset` que entrega cada fila como tensor `(1, 15)` para `nn.Conv1d`.
- Bloques `Conv1DBlock` (Conv1d + BatchNorm1d + ReLU), extractor convolucional,
  MLP y la red completa `Conv1DSoftmaxNet`.
- Cabeza CORAL, red `Conv1DCoralNet`, pérdida ordinal y pesos por número efectivo.
- Fabricas `ModelFactory` y `TaskFactory` para registrar algoritmos.
- `Trainer` con Adam, `GridSearch` interna y `NestedCrossValidator` (5x3).
- **Busqueda de hiperparametros en el loop interno**: recorre
  `HYPERPARAMETER_GRID`, elige por menor MAE (empate: mayor QWK), reentrena en
  todo el entrenamiento externo y evalua una sola vez en el test externo.
- Metricas: accuracy, balanced accuracy, precision/recall/F1 macro, MAE, QWK,
  accuracy ±1 y errores graves.
- Reportes: CSV, Markdown, rankings, graficos de barras, matrices de confusion
  y la configuracion elegida en cada fold (`seleccion_hp.csv`).

## Próximos entregables

- Integrar las extensiones de decisiones, perfiles y robustez con los controles
  y el estudio de localidad; revisar conjuntamente el informe y producir el PDF.
- La comparación obligatoria de tres variantes y seis objetivos ya está
  disponible en `results/resumen/central/`.

Detalle en [docs/04_extension_ordinal.md](docs/04_extension_ordinal.md).

## Inicio rapido

```bash
conda activate lab_pytorch
cp "/ruta/al/archivo/15 atributos R0-R5.sav" dataset/

# Un experimento rapido (sin grid) para verificar el entorno
python main.py --data-path "dataset/15 atributos R0-R5.sav" --epochs 5 --no-grid

# Experimento completo en GDS_R2 (grid interno + 5x3 folds)
python main.py --data-path "dataset/15 atributos R0-R5.sav" --target-name GDS_R2

# Los seis objetivos con tablas, rankings y graficos en results/
python main.py --data-path "dataset/15 atributos R0-R5.sav" --all-targets
```

Para revisar los datos y la arquitectura antes de entrenar, consulta
[el problema](docs/01_problema.md) y [la arquitectura](docs/02_arquitectura.md).
Las pruebas se ejecutan con `python -m unittest discover -s tests -v`.

## Estructura

```text
DL-2026-02-L02/
|-- main.py                  # CLI: arma ExperimentConfig y llama a ExperimentRunner
|-- src/
|   |-- config.py            # constantes, dataclasses de configuracion y HYPERPARAMETER_GRID
|   |-- utils.py             # semillas y dispositivo
|   |-- experiment.py        # ExperimentRunner: algoritmos x objetivos
|   |-- data/                # DataFrameLoader, TargetEncoder, StratifiedSplitter, Dataset (1, 15)
|   |-- models/              # Conv1DBlock, Conv1DBackbone, redes Softmax y CORAL, ModelFactory
|   |-- tasks/               # ClassificationTask, SoftmaxTask, CoralTask, TaskFactory
|   |-- training/            # Trainer, GridSearch, NestedCrossValidator
|   |-- evaluation/          # MetricsCalculator
|   `-- reporting/           # ReportWriter, rankings y graficos
|-- docs/                    # documentacion por partes
|-- dataset/                 # datos locales (no versionados)
|-- results/                 # salidas de main.py (no versionadas)
|-- environment.yml
`-- README.md
```

## Documentacion

1. [El problema](docs/01_problema.md): GDS, reagrupaciones, desbalance y por que
   usar Conv1D sobre los 15 items.
2. [Arquitectura](docs/02_arquitectura.md): convolucion 1D, trazado de formas,
   clases y responsabilidades.
3. [Uso](docs/03_uso.md): entorno, argumentos, ejemplos de comandos y salidas.
4. [Extension ordinal](docs/04_extension_ordinal.md): esqueletos CORAL y orden
   sugerido de implementacion.

## Plan sugerido para alumnos

1. Leer [docs/01_problema.md](docs/01_problema.md) y revisar el dataset.
2. Ejecutar `conv1d_softmax` en `GDS_R2` con `--no-grid` y luego con grid.
3. Correr `--all-targets` y revisar `results/` (tablas, rankings, `seleccion_hp.csv`).
4. Implementar los esqueletos CORAL (ver [docs/04_extension_ordinal.md](docs/04_extension_ordinal.md)).
5. Ejecutar `--algorithm all` con y sin `--use-weights`.
6. Completar el informe comparando Softmax y CORAL en los seis objetivos.
