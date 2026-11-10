# 3. Uso

## Entorno

El entorno Conda `lab_pytorch` es el mismo del Laboratorio 01.

```bash
conda activate lab_pytorch
```

Si hay que recrearlo en otro equipo:

```bash
conda env create -f environment.yml
conda activate lab_pytorch
```

En VS Code: `Ctrl + Shift + P` → `Python: Select Interpreter` → `lab_pytorch`.

## Datos

Copiar el archivo del laboratorio a `dataset/` (no se versiona):

```bash
cp "/ruta/15 atributos R0-R5.sav" dataset/
```

Todos los comandos se ejecutan desde la raiz del repositorio.

## Argumentos

### Datos y experimentos

| Argumento | Defecto | Descripcion |
|---|---|---|
| `--data-path` | (obligatorio) | Archivo `.csv` o `.sav` |
| `--target-name` | `GDS_R2` | Objetivo de un experimento suelto |
| `--all-targets` | off | Ejecuta `GDS`, `GDS_R1` ... `GDS_R5` |
| `--algorithm` | `conv1d_softmax` | `conv1d_softmax`, `conv1d_coral` o `all` |

### Modelo (configuracion base)

| Argumento | Defecto | Descripcion |
|---|---|---|
| `--conv-channels` | `16 32` | Canales de cada bloque Conv1D (uno o mas enteros) |
| `--kernel-size` | `3` | Tamaño del kernel (impar) |
| `--hidden-dim` | `32` | Neuronas de la capa oculta del MLP |
| `--dropout` | `0.15` | Dropout del MLP |

### Entrenamiento

| Argumento | Defecto | Descripcion |
|---|---|---|
| `--learning-rate` | `1e-3` | Adam |
| `--weight-decay` | `1e-4` | Adam |
| `--batch-size` | `32` | Tamaño de batch |
| `--epochs` | `20` | Epocas por cada entrenamiento |
| `--use-weights` | off | Pesos por numero efectivo (solo `conv1d_coral`) |
| `--beta` | `0.99` | Beta del numero efectivo |

### Validacion

| Argumento | Defecto | Descripcion |
|---|---|---|
| `--outer-folds` | `5` | Folds externos (reporte) |
| `--inner-folds` | `3` | Folds internos (seleccion) |
| `--gds-outer-folds` | `2` | Folds externos de `GDS` con `--all-targets` |
| `--gds-inner-folds` | `2` | Argumento heredado; la búsqueda interna de GDS se omite por protocolo |
| `--no-grid` | off | No recorre el grid: usa solo la configuracion base |

### Ejecucion y reportes

| Argumento | Defecto | Descripcion |
|---|---|---|
| `--seed` | `42` | Semilla de referencia para particion y entrenamiento |
| `--split-seed` | `--seed` | Semilla de particion externa; internas usan semilla + 100 × fold externo |
| `--init-seed` | `--seed` | Semilla de entrenamiento; iguales folds internos usan la misma semilla entre candidatas |
| `--device` | `cpu` | `cpu`, `cuda` o `auto` |
| `--output-dir` | `results` | Carpeta de CSV, Markdown y PNG |
| `--rank-metric` | `f1_macro` | Metrica del ranking generico |
| `--rank-mode` | `auto` | `auto` minimiza MAE y errores graves; maximiza el resto |

### Grid frente a configuracion fija

Con el grid activo (por defecto), cada entrada de `HYPERPARAMETER_GRID` en
`src/config.py` **sobrescribe** la configuracion base. Los flags que el grid
no toca (por ejemplo `--learning-rate`) siguen aplicando. Con `--no-grid` se
entrena solo la configuracion de la linea de comandos; el loop interno igual
se ejecuta para informar el MAE interno.

## Ejemplos de comandos

```bash
# 1. Verificar el entorno (rapido, sin grid)
python main.py --data-path "dataset/15 atributos R0-R5.sav" --algorithm all --epochs 1 --no-grid --outer-folds 2 --inner-folds 2 --output-dir results/prueba_entorno

# 2. Experimento completo en GDS_R2 (grid + 5x3)
python main.py --data-path "dataset/15 atributos R0-R5.sav" --target-name GDS_R2

# 3. GDS suelto: hay que bajar los folds
python main.py --data-path "dataset/15 atributos R0-R5.sav" --target-name GDS \
  --outer-folds 2 --inner-folds 2

# 4. Probar otra arquitectura fija (tres bloques, kernel 5)
python main.py --data-path "dataset/15 atributos R0-R5.sav" --no-grid \
  --conv-channels 8 16 32 --kernel-size 5 --hidden-dim 64 --dropout 0.3

# 5. Los seis objetivos con reportes
python main.py --data-path "dataset/15 atributos R0-R5.sav" --all-targets

# 6. Ranking por MAE
python main.py --data-path "dataset/15 atributos R0-R5.sav" --all-targets \
  --rank-metric mae_ordinal --rank-mode min

# 7. Experimento ordinal
python main.py --data-path "dataset/15 atributos R0-R5.sav" --algorithm conv1d_coral

# 8. Ordinal con pesos por numero efectivo
python main.py --data-path "dataset/15 atributos R0-R5.sav" --algorithm conv1d_coral \
  --use-weights --beta 0.99

# 9. Comparacion completa Softmax vs CORAL en los seis objetivos
python main.py --data-path "dataset/15 atributos R0-R5.sav" --all-targets --algorithm all
```

Softmax, CORAL y CORAL ponderada están implementados. Cada comando de
entrenamiento necesita su propio `--output-dir results/NOMBRE_CORRIDA`; no usar
la raíz `results/`, que contiene documentación y agregados.

## Que se ve en consola

Por cada experimento:

- una linea por fold externo con la configuracion elegida, el MAE interno y
  el F1 / MAE en el test externo,
- media ± std externa de todas las metricas,
- configuracion elegida con mas frecuencia,
- reporte por clase y matriz de confusion del ultimo fold externo.

Al final: rankings por F1, por MAE y por `--rank-metric`, y la lista de
archivos generados.

Cada objetivo guarda `<output-dir>/<objetivo>/splits.npz`: `outer_N_train`,
`outer_N_test`, `outer_N_inner_M_train` y `outer_N_inner_M_val` contienen
índices globales de fila, con folds numerados desde 1. Incluye `split_seed`,
`target`, `labels`, `classes` e `outer_N_inner_skipped`. Se comprueba que un
archivo existente coincida antes de reutilizar la carpeta. Las redes usan esas
mismas particiones; para compararlas con los controles de R2 usar 5×3 y
`--split-seed 42`. En GDS se omite siempre la búsqueda interna por protocolo.
Cada corrida nueva necesita una carpeta vacía. `manifest.json` guarda la
configuración, hashes de datos y código, versiones, dispositivo, semillas y
estado. Al repetir el mismo comando se recuperan los folds completados; si el
manifiesto no coincide se rechaza la reanudación. Se vuelve a ejecutar el fold
incompleto. `ajustes.jsonl` conserva cada intento, incluso los fallidos o
repetidos tras una interrupción; `candidatas.csv` y `metricas_folds.csv` se
regeneran desde los folds completos. Los archivos contienen datos individuales y
permanecen fuera de Git. Cada variante guarda `predicciones_oof.csv`, `confusion_oof.csv` y
`reporte.json` en `<objetivo>/<variante>/`. El reporte separa las métricas de
cada fold del agregado OOF, incluye NLL y RPS, soportes y omisiones de búsqueda.
Para regenerarlos sin entrenar, ejecutar `python -m experimentos.reportes CARPETA`.
Las predicciones actuales corresponden a la repetición 1.

## Convenciones de métricas

Las métricas macro y la QWK usan las K clases del objetivo, incluso si una clase
no aparece en un fold; una clase sin definición aporta cero a las macro. La
balanced accuracy promedia recall solo sobre clases con soporte real. El reporte
por clase indica el soporte y la confusión del último fold usa etiquetas originales.
Si el desacuerdo esperado de QWK es cero, se informa `NaN` y la causa en
`seleccion_hp.csv`. Los resúmenes muestran cuántos folds son válidos, calculan
media y desviación (`ddof=0`) sobre ellos y conservan `NaN` si no hay ninguno.

`MetricsCalculator.nll(y, probabilidades)` calcula la pérdida logarítmica media;
solo protege el logaritmo con el menor positivo representable del tipo de la
matriz. `MetricsCalculator.rps(y, probabilidades)` calcula RPS medio normalizado
por K−1. Ambas funciones exigen una matriz válida (N, K). Las redes exponen `Trainer.predict_logits(X)`, `predict_proba(X)` y la decisión
original con `predict(X)`. Las métricas NLL y RPS se incluyen en `metricas_folds.csv` y en los
reportes regenerables de cada variante.

## Salidas en `results/`

| Archivo | Contenido |
|---|---|
| `resultados.csv` / `.md` | Tabla algoritmo x objetivo x metricas |
| `seleccion_hp.csv` | Configuracion elegida en cada fold externo |
| `ranking_f1.*`, `ranking_mae.*`, `ranking_<metrica>.*` | Rankings |
| `barras_f1.png`, `barras_mae.png` | Barras por objetivo y algoritmo |
| `confusion_<algoritmo>_<objetivo>.png` | Matriz del ultimo fold externo |

## Reconstrucción de los estudios

El [README principal](../README.md#reproducir-la-comparación-obligatoria) contiene
los comandos centrales. La [guía de resultados](../results/README.md) cubre
controles, constantes, decisiones, perfiles, localidad y robustez, distinguiendo
regeneración de agregados y reentrenamiento.

Los manifiestos conservan las versiones originales de cada corrida.
La reproducción desde otro entorno se revisará en H6.2; no se promete igualdad
bit a bit entre dispositivos. Para limitar CPU usar
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.
