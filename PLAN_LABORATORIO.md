# Plan de trabajo — Laboratorio 02

**Equipo:** Gael y Johan.

**Entrega final:** 12 de noviembre de 2026, a las **23:59**, zona horaria
**America/Santiago**.

**Objetivo interno:** PDF y código revisados el 10 de noviembre; revisión final
el día 11. Todas las fechas de este documento corresponden a 2026.

## 1. Cómo usar este documento

Este documento reúne el protocolo, las tareas y el registro de avances. La
primera etapa organiza el trabajo; el desarrollo y los experimentos avanzan por
fichas pequeñas. Las duraciones son estimaciones de trabajo activo, no tiempos
de cómputo. Las fechas internas son plazos máximos: se puede avanzar antes si
las dependencias están completas.

Para mantener el repositorio fácil de leer, actualizar las guías existentes
cuando corresponda; no crear un documento por cada ficha. Guardar las salidas
generadas en `results/` y compartir solo tablas o figuras finales seleccionadas.

1. Elegir una ficha cuyas dependencias estén completas.
2. Cambiar su estado a **en curso** y realizar los pasos indicados.
3. Ejecutar su comprobación y guardar la evidencia.
4. Pasarla a **en revisión**. Al verificar su aceptación, marcar la casilla y
   cambiar el estado a **terminada**.
5. Registrar lo que sigue o un bloqueo concreto. Una tarea bloqueada conserva
   el comando, el error y lo que se intentó.

Estados permitidos: **pendiente, en curso, bloqueada, en revisión, terminada**.
Cambiar responsables o fechas mediante un acuerdo registrado. El informe se
redacta entre ambos, con partes repartidas y revisión final conjunta.

**Estado al 28/10/2026:** H0.1, H0.2, H0.3, G1.1, G1.2, J1.1, J1.2, G2.1, G2.2 y G2.3 terminadas. Reparto de trabajo
confirmado por ambos; las demás tareas y fechas del tablero se mantienen.

### Tablero de tareas

La casilla indica aceptación completa. El estado detallado y la evidencia se
actualizan aquí; no es necesario repetirlos en cada ficha.

| Hecho | Ficha | Responsable | Fecha interna | Depende de | Estado | Evidencia de cierre |
|---|---|---|---|---|---|---|
| [x] | H0.1 Organización documental | Gael | 24/10 | — | terminada | Este plan, enlace en README y comprobación documental |
| [x] | H0.2 Entorno y arranque | Gael | 24/10 | H0.1 | terminada | 25/10: prueba Softmax/R2 correcta; log y reportes en results/h0_arranque/ |
| [x] | H0.3 Acuerdo de trabajo | Gael y Johan | 25/10 | H0.1 | terminada | 25/10: revisión técnica y reparto confirmado por ambos |
| [x] | G1.1 Auditoría de datos | Gael | 26/10 | H0.2 | terminada | Script y pruebas; resumen en `docs/01_problema.md` |
| [x] | G1.2 Formas y parámetros | Gael | 26/10 | H0.2 | terminada | Inspección y pruebas; explicación en `docs/02_arquitectura.md` |
| [x] | J1.1 Constantes | Johan | 27/10 | H0.2, H0.3 | terminada | Implementado ConstantPredictor (moda con desempate a índice menor, mediana y distribución previa) y pruebas en tests/test_controles.py. |
| [x] | J1.2 Controles logísticos | Johan | 27/10 | J1.1 | terminada | Implementado LogisticControlPredictor para suma de bits y 15 entradas, barriendo las 4 candidatas de $C$ con selección MAE/QWK y pruebas aprobadas. |
| [x] | G2.1 Niveles e inferencia | Gael | 29/10 | G1.1 | terminada | Funciones y pruebas en `tests/test_coral.py`; suite completa aprobada el 27/10 |
| [x] | G2.2 Cabeza CORAL | Gael | 29/10 | G1.2 | terminada | Cabeza y red implementadas; pruebas de formas, orden y gradientes aprobadas el 28/10 |
| [x] | G2.3 Pérdida CORAL | Gael | 29/10 | G2.1, G2.2 | terminada | BCE con logits, reducción y estabilidad comprobadas el 28/10 |
| [ ] | G2.4 Pesos por clase | Gael | 29/10 | G2.3 | pendiente | — |
| [ ] | G2.5 Integración CORAL | Gael | 29/10 | G2.4 | pendiente | — |
| [ ] | J2.1 MLP de referencia | Johan | 31/10 | J1.2 | pendiente | — |
| [ ] | J2.2 Boosting | Johan | 31/10 | J1.2 | pendiente | — |
| [ ] | J2.3 Comparación de controles | Johan | 31/10 | J2.1, J2.2 | pendiente | — |
| [ ] | G3.1 Métricas y alfabeto | Gael | 31/10 | G1.1 | pendiente | — |
| [ ] | G3.2 Particiones y semillas | Gael | 31/10 | G1.1, H0.3 | pendiente | — |
| [ ] | G3.3 Probabilidades | Gael | 31/10 | G2.5 | pendiente | — |
| [ ] | G3.4 Registros de ajustes | Gael | 31/10 | G3.2 | pendiente | — |
| [ ] | G3.5 Predicciones y reportes | Gael | 31/10 | G3.1, G3.3, G3.4 | pendiente | — |
| [ ] | G4.1 Piloto de cómputo | Gael | 01/11 | G3.5 | pendiente | — |
| [ ] | G4.2 Corrida obligatoria | Gael | 02/11 | G4.1 | pendiente | — |
| [ ] | G4.3 Tablas y borrador | Gael | 02/11 | G4.2 | pendiente | — |
| [ ] | J3.1 Modelos de localidad | Johan | 03/11 | J2.1 | pendiente | — |
| [ ] | J3.2 Órdenes y prueba corta | Johan | 03/11 | J3.1, G3.2 | pendiente | — |
| [ ] | G5.1 Moda y mediana | Gael | 05/11 | G4.2 | pendiente | — |
| [ ] | G5.2 Predictor por perfil | Gael | 05/11 | G3.2, G3.3 | pendiente | — |
| [ ] | G5.3 Análisis de decisiones y perfiles | Gael | 05/11 | G5.1, G5.2 | pendiente | — |
| [ ] | J4.1 Ejecución de localidad | Johan | 06/11 | J3.2, G4.1 | pendiente | — |
| [ ] | J4.2 Análisis de localidad | Johan | 06/11 | J4.1 | pendiente | — |
| [ ] | H5.1 Robustez | Gael | 08/11 | G4.3 | pendiente | — |
| [ ] | H5.2 Integración de resultados | Gael y Johan | 08/11 | J2.3, G5.3, J4.2, H5.1 | pendiente | — |
| [ ] | H6.1 Informe completo | Gael y Johan | 10/11 | H5.2 | pendiente | — |
| [ ] | H6.2 Reproducción | Gael y Johan | 10/11 | H6.1 | pendiente | — |
| [ ] | H6.3 Archivos de entrega | Gael y Johan | 10/11 | H6.2 | pendiente | — |
| [ ] | F1 Revisión final | Gael y Johan | 11/11 | H6.3 | pendiente | — |
| [ ] | F2 Entrega | Gael y Johan | 12/11, 23:59 | F1 | pendiente | — |

## 2. Qué se busca demostrar

### Entrega académica

Completar las tres variantes exigidas en los seis objetivos, responder las siete
preguntas del enunciado y entregar código modular, repositorio e informe PDF.

### Preguntas del estudio

1. ¿Cómo cambian las métricas al usar Softmax, CORAL y CORAL ponderada?
2. ¿Cuánto cambia el resultado al decidir por moda o mediana sobre las mismas
   probabilidades?
3. ¿Qué obtiene la CNN frente a controles más simples y qué ocurre con perfiles
   de respuestas vistos o nuevos durante el entrenamiento?
4. ¿La vecindad impuesta por Conv1D y el orden original aportan una ventaja estable?

La mediana Softmax y el diagnóstico de ambigüedad continúan el Lab01. La
ampliación principal estudia localidad y controles. Un resultado negativo bien
comprobado también responde las preguntas. No se afirmará novedad científica
general, causalidad ni utilidad clínica a partir de este estudio.

### Referencias iniciales sobre datos y código

Estos valores se verificaron en la revisión inicial y serán reproducidos en G1:

- 1119 registros; ID, 15 entradas binarias y seis objetivos.
- Sin faltantes observados; ID únicos y cinco recodificaciones consistentes.
- 321 perfiles de entrada; 97 con etiquetas GDS diferentes, que abarcan 851 filas.
- GDS contiene dos registros de etapa 7. El ID no entra como predictor.
- La plantilla incluye Softmax, grid, entrenamiento y validación. Faltan CORAL
  y las funciones ordinales señaladas como TODO.
- Los reportes iniciales muestran la confusión del último fold. Será necesario
  exportar probabilidades y agregar todos los folds externos.

Un ID único no demuestra que cada fila sea una persona distinta. Tampoco se
eliminarán filas por compartir respuestas: un mismo perfil puede corresponder
a personas diferentes.

La correspondencia de las últimas siete variables con memoria de nombre y
dirección del ACE-III es una hipótesis respaldada por coincidencias con el
instrumento, no una procedencia confirmada del archivo. Mantener los nombres y
el orden docente y reconocer que el texto exacto, los sufijos y la codificación
no están completamente documentados. Llamar inicialmente **suma de bits** al
agregado de las entradas.

## 3. Protocolo congelado antes de evaluar

### 3.1 Comparación obligatoria

| Variante | Entrenamiento | Decisión original |
|---|---|---|
| Softmax | CrossEntropyLoss | Argmax, con empate hacia el índice menor |
| CORAL | BCE acumulativa | Conteo de probabilidades mayores que 0.5 |
| CORAL ponderada | Misma BCE, ponderada por clase real | Mismo conteo |

- Objetivos: GDS y GDS_R1–GDS_R5, con sus recodificaciones docentes.
- Tronco y cuatro candidatas del grid docente sin cambios en esta comparación.
- Adam, 20 épocas, batch 32, learning rate 0.001 y weight decay 0.0001.
- CORAL ponderada con beta 0.99; conteos obtenidos solo del entrenamiento de
  cada ajuste, recalculados al reentrenar sobre todo el entrenamiento externo.
- Cinco folds externos y tres internos en las cinco recodificaciones.
- GDS: dos externos, configuración base prefijada y búsqueda interna omitida.
  Registrar `inner_skipped`; no describirlo como validación anidada efectiva 2×2.
- Selección interna por menor MAE promedio; empate por mayor QWK. NaN no gana.
- Referencia central: semilla de partición 42 y semilla de entrenamiento 42.
- Evaluar el test externo después de seleccionar y reentrenar; nunca usarlo para
  elegir épocas, columnas, candidatas, umbrales o semillas.
- Reportar media ± desviación estándar externa, con `ddof=0` para conservar la
  convención de la plantilla. La desviación estándar no es un intervalo de confianza.

Las nueve métricas son accuracy, balanced accuracy, precision macro, recall
macro, F1 macro, MAE, QWK, accuracy ±1 y errores de distancia al menos dos.
Comparar dentro de cada objetivo. Las seis recodificaciones usan los mismos
registros y no son seis muestras independientes. En R3, las etiquetas originales
1/3 se recodifican 0/1: MAE=1−accuracy, accuracy ±1=1 y errores graves=0.

### 3.2 Aprendizaje frente a decisión

Para cada distribución externa de Softmax, CORAL y CORAL ponderada, obtener
moda y mediana inferior. La mediana es el primer índice cuya acumulada alcanza
0.5. Comparar MAE, accuracy, errores graves y recall por clase por pares, usando
exactamente las mismas filas y probabilidades.

No hay nuevos ajustes. Las configuraciones se seleccionaron con la regla
original; este estudio no representa una búsqueda optimizada para cada regla
adicional. Para R3 comprobar la coincidencia de moda y mediana con ese desempate.

### 3.3 Controles e información de perfiles

En los seis objetivos, calcular moda y mediana constantes con las etiquetas de
entrenamiento. Los empates eligen la clase menor. El predictor por perfil usa
frecuencias de etiquetas del perfil en entrenamiento, con suavizado aditivo de
uno por clase; para perfiles nuevos usa las frecuencias globales de entrenamiento
con el mismo suavizado. Exportar probabilidades, moda, mediana y si el perfil fue
visto en entrenamiento.

En R2 comparar estos cuatro controles, con búsqueda interna MAE/QWK:

| Control | Cuatro candidatas | Valores fijos |
|---|---|---|
| Suma de bits + logística | C: 0.01, 0.1, 1, 10 | L2, lbfgs, max_iter=2000, sin pesos de clase |
| 15 entradas + logística | C: 0.01, 0.1, 1, 10 | Mismos valores; sin estandarización de entradas binarias |
| MLP sencilla del Lab01 | hidden 32/64 × dropout 0.15/0.30 | 15→hidden→K, ReLU, Adam y duración docente |
| HistGradientBoosting | max_leaf_nodes: 3, 7, 15, 31 | max_iter=100, lr=0.1, l2=1, early_stopping=False |

La logística recibe la suma como una sola variable numérica, sin ajuste global
previo. Usar las mismas filas, folds y alfabeto que las redes. Volver a entrenar
la MLP bajo el protocolo Lab02; sus números antiguos son antecedentes.

### 3.4 Localidad y orden

Panel con configuración prefijada en R2:

| Modelo | Arquitectura |
|---|---|
| CNN-k3 | Base docente: canales 16/32, hidden 32, dropout 0.15, kernel 3 |
| CNN-k1 | Misma configuración con kernel 1 |
| MLP de capacidad próxima | 15→128→115→3, ReLU tras cada oculta y dropout 0.15 antes de la salida |

Todos usan cabeza Softmax, CE, argmax y entrenamiento docente. La MLP tiene
17231 parámetros y la CNN-k3 17219 para K=3; reportar también los de CNN-k1.
La comparación no iguala todas las propiedades de optimización o regularización.

Órdenes, con índices sobre `FEATURE_COLUMNS` y base cero:

| Orden | Procedimiento | Índices exactos |
|---|---|---|
| Original | Orden docente | 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14 |
| Dentro de bloques | `default_rng(2026)`, permutar consecutivamente bloques 0:4, 4:8 y 8:15 | 3,1,0,2,4,5,6,7,9,13,14,12,8,10,11 |
| Global | `default_rng(2027).permutation(15)` | 4,2,13,5,9,10,7,14,3,0,11,1,8,6,12 |

Aplicar cada orden tanto al entrenamiento como a evaluación y entrenar desde
cero. No permutar solo en inferencia. Conservar la lista de nombres de columnas.

- Particiones externas: 42, 314, 2718.
- Inicializaciones: 7, 17, 29.
- Cinco folds externos por condición. Sin búsqueda ni bucle interno informativo
  en este panel fijo; no confundirlo con `--no-grid`, que en la plantilla aún
  ejecuta evaluación interna.
- Total: 3 modelos × 3 órdenes × 3 particiones × 3 inicializaciones × 5 folds
  = 405 ajustes. No seleccionar la mejor permutación como resultado principal.

### 3.5 Robustez, métricas probabilísticas y límites

Repetir las tres variantes obligatorias de R1 y R2 con particiones 314 y 2718,
semilla de entrenamiento 42 y búsqueda interna completa. Conservar separado
el efecto de partición y el de inicialización.

Añadir NLL y RPS normalizado por K−1 como métricas probabilísticas. Para NLL,
proteger únicamente el logaritmo con el menor positivo representable del dtype
y registrar esa convención. No corregir acumulativas inválidas silenciosamente.
RPS puede calcularse con las acumuladas y es evaluable también en R3.

Reportar siempre condiciones previstas, incluidas las que no mejoren. Mostrar
diferencias pareadas y dispersión; no aplicar un t-test que trate los folds como
independientes. No usar el test externo para ampliar o seleccionar el estudio.
La ambigüedad calculada con todo el archivo es descriptiva, no un predictor ni
una cota poblacional o una cota que deba respetar un agregado OOF de varios modelos.

No se incluyen CORN, pérdidas adicionales, calibración, conformalidad, modelos
de texto ni generación sintética en esta ejecución. Primero cerrar el estudio
acordado. Las incertidumbres del instrumento se exponen como limitaciones.

## 4. Propuesta de trabajo

El taller consiste en comparar modelos que predicen una categoría a partir de
15 respuestas. El reparto general se confirmó el 25/10:

| Quién | De qué se encarga |
|---|---|
| Gael | Revisar los datos, completar el modelo CORAL y preparar la evaluación común de los modelos. |
| Johan | Probar modelos más sencillos para comparar y estudiar si cambiar el orden de las 15 preguntas afecta los resultados. |
| Ambos | Repartir la redacción, explicar sus resultados y juntar y revisar el informe final entre los dos. |

El informe sigue la plantilla del laboratorio. Cada uno prepara borradores de
sus partes; ambos revisan el documento completo. El reparto por sección está
en la sección 7.

### Cómo vamos a trabajar

- Usaremos los mismos datos y la misma separación entre filas de entrenamiento
  y de evaluación para todos los modelos. Así podremos comparar sus resultados
  en las mismas condiciones.
- Cada uno guardará sus predicciones y resultados. Gael preparará las funciones
  comunes para juntarlos y construir las tablas y gráficos del informe.
- Los resultados completos se guardan en carpetas separadas de `results/`.
  Al repositorio subimos las tablas y gráficos elegidos para el informe.
- Cada uno trabaja en sus archivos. Si hace falta cambiar una parte compartida
  del código, lo acordamos antes.

### Próximos pasos

- **Gael, 26/10:** revisar los datos y comprobar cómo funciona la red entregada
  en la plantilla (G1.1 y G1.2).
- **Johan, 27/10:** preparar los primeros modelos de comparación: uno que siempre
  responde lo mismo y otros que usan regresión logística (J1.1 y J1.2).
- **Gael, 31/10:** dejar listas las funciones y archivos comunes para evaluar
  los modelos y juntar sus resultados (G3). Hasta entonces usamos las funciones
  que ya tiene la plantilla.

H0.3 cerrada el 25/10 tras la confirmación de Johan al reparto general y al
trabajo conjunto en el informe. Las tareas y fechas del tablero se mantienen;
cualquier ajuste posterior se anota en el registro de avances.

### Archivos de cada integrante

Gael mantiene `main.py`, `src/`, validación y reportes comunes. Johan desarrolla
sus controles y localidad en `experimentos/controles.py` y
`experimentos/localidad.py`; sus pruebas se llaman `test_controles.py` y
`test_localidad.py`. Estos archivos son entregables futuros, no implementaciones
ya disponibles. Acordar cualquier cambio de los módulos compartidos.

### Datos, particiones e interfaces

- Entrada: X `(N,15)` float32, en el orden docente; y `(N,)` int64 en `0..K−1`.
- `row_id`: posición original de fila, base cero, ligada al hash del SAV. El ID
  original nunca entra en X. Conservar el mapa de clases originales e índices.
- Para avanzar antes de G3, reutilizar `prepare_experiment_data`, que devuelve
  `ExperimentData` con `X`, `y` y `classes`, y `StratifiedSplitter` de la plantilla.
  No crear otra recodificación ni otro protocolo de partición.
- En los controles de R2, conservar los cinco folds externos con semilla 42.
  Los tres internos usan `42 + 100 * outer_fold`, numerando externos desde 1,
  como en la plantilla. Sus índices son locales al entrenamiento externo:
  convertirlos a filas globales mediante `train_idx[inner_idx]`.
- G3 exportará índices en la numeración global de filas. Cada inner fold estará
  contenido en el entrenamiento externo. Los controles consumirán esas mismas
  particiones y verificarán que coincidan con las que usaron al comenzar.
- Separar `split_seed` e `init_seed`, manteniendo `--seed` como referencia de
  compatibilidad cuando no se especifiquen por separado.
- Fijar el mismo conjunto de semillas de entrenamiento por fold y candidata;
  no usar `candidate_index` para cambiar inicializaciones. Igual semilla no
  equivale a igual inicialización entre arquitecturas diferentes.
- Ampliar las tareas con `probabilities(logits)` → `(B,K)` y el Trainer con
  `predict_proba(X)` → `(N,K)`, preservando `predict(X)` → `(N,)`.
- Los controles clásicos usan `fit` y `predict_proba`; alinear sus columnas con
  el alfabeto completo. Exponer un fallo de ajuste, no rellenarlo con una métrica.
- Mantener identificadores de variante: `softmax`, `coral`, `coral_weighted`.

### Salidas previstas

Cada corrida usa una carpeta nueva en `results/`, sin sobrescribir otras. Mantener
allí datos individuales y artefactos completos, ignorados por Git.

| Artefacto | Contenido mínimo |
|---|---|
| `manifest.json` | Configuración efectiva, hashes de datos y código, versiones, semillas, dispositivo y estado |
| `splits.npz` | Índices externos e internos en numeración global; orden y semilla identificados en el manifiesto |
| `metricas_folds.csv` | Variante, objetivo, repetición, fold y todas las métricas externas |
| `candidatas.csv` | Configuración, resultados internos por fold, medias, criterio y selección |
| `ajustes.jsonl` | Identificador de ajuste, semillas, soportes, pesos, épocas, pérdidas, tiempos y estado |
| `predicciones_oof.csv` por objetivo/variante | row_id, repetición, fold, verdad, clase original, logits, probabilidades, decisiones y perfil visto cuando corresponda |
| Tablas y confusiones | Resúmenes externos y agregados OOF etiquetados por separado; alfabeto completo |

Guardar cada fold completado y retomar solo si coincide el manifiesto. Conservar
errores como ejecuciones fallidas; no transformar ausencias en resultados cero.
No convertir que falte CORAL en una corrida completa con salida exitosa.

Las tablas y figuras agregadas elegidas se copiarán a `docs/resultados/` para
versionarlas. Registrar su corrida de origen. Las confusiones principales suman
todos los folds, con etiquetas originales y soporte por clase.

Para métricas macro y QWK usar el alfabeto completo. Balanced accuracy promedia
recall sobre clases con soporte real; mostrar su soporte. Si QWK no es estimable,
registrar NaN y la causa, sin sustituirlo por acuerdo perfecto. Los reportes deben
mostrar cuántos folds tienen una métrica válida, sin omitirlos silenciosamente.

## 5. Fichas de trabajo

Los nombres de módulos, pruebas y scripts que aún no existen son entregables
de su ficha. Sus comandos de comprobación estarán disponibles al cerrarla.
Ejecutar los comandos desde la raíz, con el entorno del laboratorio activo.

### H0.1 — Organización documental

**Responsable:** Gael. **Fecha:** 24/10. **Duración:** 45–90 min. **Insumos:**
plan acordado, README y estructura de la plantilla.

1. Preparar este protocolo, el tablero y las fichas.
2. Añadir un enlace al plan desde README y comprobar el formato y los nombres.

**Entregable:** plan compartido accesible desde README. **Comprobación:**
`git diff --check` y revisión de documentos nuevos. **Aceptación:** responsables,
fechas, dependencias, entregables y comprobaciones presentes; ninguna tarea técnica
futura marcada como completada.

### H0.2 — Entorno y ejecución corta

**Responsable:** Gael. **Fecha:** 24/10. **Duración:** 60–90 min. **Insumos:**
`environment.yml`, Python y SAV local.

1. Activar o crear el entorno y registrar ejecutable, versiones y hash del SAV.
2. Ejecutar Softmax corto en R2, conservando el log y los reportes.
3. Confirmar que las salidas son legibles y que esta prueba no se usa como resultado final.

**Comando actual:**

```bash
python main.py --data-path 'dataset/15 atributos R0-R5.sav' --target-name GDS_R2 --algorithm conv1d_softmax --epochs 1 --no-grid --outer-folds 2 --inner-folds 2 --output-dir results/h0_arranque
```

**Entregable:** registro de entorno y log bajo `results/h0_arranque/`.
**Aceptación:** código de salida cero, dos folds externos, métricas finitas y
CSV/Markdown/PNG creados. Registrar el comando completo en la bitácora.

### H0.3 — Acordar el trabajo del equipo

**Responsables:** Gael y Johan. **Fecha:** 25/10. **Duración:** 45–60 min.
**Insumos:** protocolo de la sección 3 y propuesta de trabajo de la sección 4.

1. Revisar qué hará cada uno y para cuándo.
2. Confirmar que usaremos los mismos datos y la misma forma de evaluar, y cómo
   guardaremos y juntaremos los resultados.
3. Anotar la confirmación de ambos o los cambios que necesite la propuesta.

**Entregable:** reparto y fechas confirmados en el registro de avances.
**Comprobación:** revisar que la forma de trabajar propuesta encaje con el código
de la plantilla y con las comparaciones del taller.
**Aceptación:** ambos saben cuál es su siguiente tarea, qué necesitan para
empezar y qué deben entregar.

### G1.1 — Auditoría reproducible de datos

**Responsable:** Gael. **Fecha:** 26/10. **Duración:** 90–120 min. **Insumos:**
SAV, `FEATURE_COLUMNS`, objetivos y agrupaciones docentes.

1. Crear `experimentos/auditoria_datos.py` para comprobar dominios, finitud,
   metadatos de ausencia, ID, recodificaciones y perfiles repetidos.
2. Exportar solo un resumen agregado y comprobar los conteos de referencia.

**Entregable:** script, `test_datos.py` y resumen agregado.
**Comprobación:** ejecutar el script y
`python -m unittest discover -s tests -p 'test_datos.py' -v`.
**Aceptación:** 1119 filas, 15 entradas binarias, 321 perfiles, 97 ambiguos en
GDS y cero discrepancias de recodificación; X excluye ID y todos los objetivos.

### G1.2 — Formas, parámetros y convolución

**Responsable:** Gael. **Fecha:** 26/10. **Duración:** 60–90 min. **Insumos:**
tronco y Softmax entregados, documentación de arquitectura.

1. Inspeccionar con código entradas `(B,15)` y `(B,1,15)`, salidas y capas.
2. Comprobar longitud 15, conteo de parámetros y campo receptivo; generar un
   ejemplo ejecutable de ventana convolucional con primitivas PyTorch.

**Entregable:** `test_arquitectura.py` y una explicación breve de resultados.
**Comprobación:** `python -m unittest discover -s tests -p 'test_arquitectura.py' -v`.
**Aceptación:** Softmax K=3 tiene 17219 parámetros, formas válidas y salidas
finitas. Explicar Flatten, posición y concentración de parámetros mediante las
salidas del programa; no se requiere cálculo manual.

### J1.1 — Predictores constantes

**Responsable:** Johan. **Fecha:** 27/10. **Duración:** 60–90 min. **Insumos:**
datos y splitter existentes, contrato y métricas.

1. Implementar moda, mediana y distribución de entrenamiento en su módulo de controles.
2. Probar empates y que cambiar las etiquetas de evaluación no modifica el predictor.

**Entregable:** controles constantes y primeras pruebas en `test_controles.py`.
**Comprobación:** `python -m unittest discover -s tests -p 'test_controles.py' -v`.
**Aceptación:** decisiones dentro del alfabeto, probabilidades válidas y conteos
obtenidos solo de train. Cubrir el caso binario y los empates mediante código.

### J1.2 — Controles logísticos

**Responsable:** Johan. **Fecha:** 27/10. **Duración:** 90–120 min. **Insumos:**
J1.1, R2 y candidatas de la sección 3.3.

1. Implementar logística sobre suma de bits y sobre las 15 entradas.
2. Recorrer las cuatro candidatas dentro del primer entrenamiento externo de R2.
3. Exportar puntuaciones internas, ganadora y evaluación externa de prueba corta.

**Entregable:** ambos controles y log de un fold, identificado como prueba corta.
**Comprobación:** pruebas de controles y ejecución del módulo sobre ese fold.
**Aceptación:** cuatro candidatas registradas por control; selección MAE/QWK;
probabilidades `(N,3)` alineadas. Registrar y resolver cualquier no convergencia.

### G2.1 — Niveles acumulativos e inferencia

**Responsable:** Gael. **Fecha:** 29/10. **Duración:** 45–90 min. **Insumos:**
auditoría y funciones ordinales pendientes.

1. Implementar `labels_to_levels` y `logits_to_ordinal_predictions`.
2. Probar K=2,3,7, extremos de etiqueta, umbral 0.5 y formas/dtypes.

**Entregable:** funciones y casos de `test_coral.py`.
**Comprobación:** `python -m unittest discover -s tests -p 'test_coral.py' -v`.
**Aceptación:** niveles equivalentes a `y>k`; predicciones int64 entre 0 y K−1;
el logit cero no suma un umbral. Sin sigmoide previa a una pérdida con logits.

### G2.2 — Cabeza CORAL

**Responsable:** Gael. **Fecha:** 29/10. **Duración:** 60–90 min. **Insumos:**
tronco y esqueletos CORAL.

1. Implementar puntaje lineal compartido sin sesgo y sesgos ordenados con softplus.
2. Conectar la cabeza al tronco y comprobar monotonía antes y después de un paso de optimización.

**Entregable:** `CoralLayer`, `Conv1DCoralNet` y pruebas.
**Comprobación:** suite `test_coral.py` sobre K=2,3,7 y ambas formas de entrada.
**Aceptación:** logits `(B,K−1)`, orden decreciente y gradientes finitos; el
caso K=2 no depende de que existan diferencias entre varios sesgos.

### G2.3 — Pérdida CORAL

**Responsable:** Gael. **Fecha:** 29/10. **Duración:** 60–90 min. **Insumos:**
niveles y cabeza implementados.

1. Implementar BCE con logits sin reducción, sumar umbrales y promediar muestras.
2. Contrastar automáticamente con BCE binaria en K=2 y con la reducción de referencia PyTorch.

**Entregable:** `coral_loss` y pruebas de reducción/estabilidad.
**Comprobación:** suite `test_coral.py`, incluidos logits de gran magnitud.
**Aceptación:** escalar y gradientes finitos; reducción sum/mean correcta. No
copiar una reducción distinta del Lab01 sin advertir la diferencia.

### G2.4 — Pesos por número efectivo

**Responsable:** Gael. **Fecha:** 29/10. **Duración:** 60–90 min. **Insumos:**
pérdida CORAL y etiquetas de entrenamiento.

1. Implementar pesos de Cui con beta 0.99, ausentes a cero y media del vector igual a uno.
2. Ponderar cada pérdida por el peso de su clase real, antes de promediar el batch.
3. Probar conteos iguales, clases ausentes y equivalencia con pesos unitarios.

**Entregable:** función y pruebas de origen de los conteos y ponderación.
**Comprobación:** suite `test_coral.py`.
**Aceptación:** sin NaN; pesos por clase de tamaño K, no pesos por umbral;
no consultar etiquetas de validación/test ni añadir muestreo ponderado.

### G2.5 — Integración corta de CORAL

**Responsable:** Gael. **Fecha:** 29/10. **Duración:** 60–90 min. **Insumos:**
G2.1–G2.4 y CLI actual.

1. Ejecutar R2 corto con CORAL sin pesos y con pesos, en carpetas separadas.
2. Verificar que fábricas, tarea, entrenamiento e inferencia se conectan correctamente.

**Comprobación:** usar el comando H0.2 cambiando algoritmo a `conv1d_coral`,
salida a `results/g2_coral`; repetir con `--use-weights` y salida
`results/g2_coral_weighted`. **Entregable:** logs y pruebas de integración.
**Aceptación:** ambas corridas terminan, no se omite CORAL y hay métricas
finitas. No presentar estas corridas de una época como resultados finales.

### J2.1 — MLP sencilla de referencia

**Responsable:** Johan. **Fecha:** 31/10. **Duración:** 90–120 min. **Insumos:**
arquitectura sencilla Lab01 y entrenamiento docente.

1. Adaptar 15→hidden→K con ReLU y dropout; registrar las cuatro candidatas.
2. Ejecutar una prueba corta y comprobar probabilidad, etiquetas y separación de folds.

**Entregable:** control MLP en su módulo y pruebas.
**Comprobación:** suite `test_controles.py` y ejecución corta R2.
**Aceptación:** logits sin Softmax en forward, CE correcta, cuatro candidatas
exactas y ningún resultado antiguo reutilizado como evaluación actual.

### J2.2 — Boosting tabular

**Responsable:** Johan. **Fecha:** 31/10. **Duración:** 60–90 min. **Insumos:**
R2, parámetros fijos y contrato de controles.

1. Añadir HistGradientBoosting y sus cuatro tamaños de hojas.
2. Comprobar alineación de probabilidades y registrar una búsqueda de prueba corta.

**Entregable:** control y pruebas.
**Comprobación:** suite `test_controles.py` y ejecución sobre un fold.
**Aceptación:** cuatro candidatas, early stopping desactivado, alfabeto completo
y ninguna selección basada en test externo.

### J2.3 — Ejecutar y analizar controles

**Responsable:** Johan. **Fecha:** 31/10. **Duración:** 90–120 min de preparación
y análisis, más cómputo. **Insumos:** J2.1, J2.2 y logística.

1. Evaluar cuatro controles en R2 con 5×3, partición 42 e inicialización 42 donde corresponda.
2. Evaluar constantes en los seis objetivos con los folds externos previstos.
3. Escribir tabla, soporte por clase y una interpretación breve de capacidad e información.

**Entregable:** resultados en `results/controles/` y borrador de su sección.
**Comprobación:** pruebas de controles, integridad OOF y regeneración de la tabla.
**Aceptación:** 260 ajustes de selección/refit R2, sin contar constantes;
candidatas, probabilidades y métricas trazables. La comparación final con CNN
puede completarse al estar disponibles sus resultados, sin detener este estudio.

### G3.1 — Métricas con alfabeto completo

**Responsable:** Gael. **Fecha:** 31/10. **Duración:** 90–120 min. **Insumos:**
mapas de clases y módulo actual de métricas.

1. Fijar K y etiquetas de referencia en QWK, métricas macro y confusiones.
2. Añadir RPS/NLL y explicitar ausencias, NaN y normalizaciones.
3. Probar clases intermedias ausentes y las identidades de R3.

**Entregable:** métricas y `test_metricas.py`.
**Comprobación:** `python -m unittest discover -s tests -p 'test_metricas.py' -v`.
**Aceptación:** QWK no comprime distancias, confusión K×K, RPS/NLL evaluables y
convenciones de la sección 4 reflejadas en reportes.

### G3.2 — Particiones y semillas reutilizables

**Responsable:** Gael. **Fecha:** 31/10. **Duración:** 90–120 min. **Insumos:**
splitter, grid y contrato técnico.

1. Persistir splits externos/internos y separar partición de entrenamiento.
2. Eliminar la dependencia de semillas respecto al orden de las candidatas.
3. Conservar el tratamiento de GDS y verificar compatibilidad de controles.

**Entregable:** interfaz y `test_particiones.py`.
**Comprobación:** `python -m unittest discover -s tests -p 'test_particiones.py' -v`.
**Aceptación:** disjunción y cobertura externa, internos contenidos en train,
misma asignación al reordenar candidatas y `inner_skipped` verdadero en GDS.

### G3.3 — Probabilidades por clase

**Responsable:** Gael. **Fecha:** 31/10. **Duración:** 90–120 min. **Insumos:**
Softmax, CORAL e interfaces acordadas.

1. Añadir conversión Softmax y conversión de acumulativas CORAL a masas por clase.
2. Exponer logits, probabilidades y decisiones conservando la API de etiquetas.

**Entregable:** interfaces y `test_probabilidades.py`.
**Comprobación:** `python -m unittest discover -s tests -p 'test_probabilidades.py' -v`.
**Aceptación:** probabilidades `(N,K)`, finitas, no negativas y suma uno; las
decisiones originales se conservan; acumulativas inválidas producen un fallo visible.

### G3.4 — Registros por ajuste

**Responsable:** Gael. **Fecha:** 31/10. **Duración:** 90–120 min. **Insumos:**
grid, fit y particiones.

1. Registrar candidatas completas, soportes, pesos, pérdidas y selección.
2. Guardar manifiesto, tiempos y estado por ajuste/fold para retomar sin mezclar corridas.

**Entregable:** registros de la sección 4 y casos de `test_registros.py`.
**Comprobación:** prueba sintética de selección y corrida corta interrumpida
entre folds y retomada. **Aceptación:** selección solo interna, archivos
completos identificables y rechazo de una reanudación con manifiesto distinto.

### G3.5 — OOF y reportes regenerables

**Responsable:** Gael. **Fecha:** 31/10. **Duración:** 90–120 min. **Insumos:**
métricas, probabilidades y registros.

1. Exportar predicciones por fila, repetición y variante, sin perder el mapa original.
2. Agregar confusiones de todos los folds y generar reportes desde archivos guardados.
3. Diferenciar resumen externo, agregado OOF y búsqueda interna omitida.

**Entregable:** exportación y `experimentos/reportes.py`, con pruebas de registros.
**Comprobación:** `python -m unittest discover -s tests -p 'test_registros.py' -v`
y regeneración desde la corrida corta. **Aceptación:** una predicción por fila
y variante/repetición; métricas y confusiones reproducidas sin entrenar; variantes
ponderada y no ponderada nunca se sobrescriben ni se combinan por nombre.

### G4.1 — Piloto de cómputo

**Responsable:** Gael. **Fecha:** 01/11. **Duración:** 60–120 min más cómputo.
**Insumos:** servidor y un ajuste representativo de 20 épocas en R2.

1. Registrar entorno y recursos reales del servidor.
2. Comparar CPU y GPU con tres repeticiones; incluir carga, ajuste y evaluación.
3. Comparar pocos procesos independientes, limitando hilos, y elegir por ajustes/tiempo.

**Entregable:** tabla de tiempos y configuración de ejecución documentada.
**Comprobación:** mismos datos/configuración de referencia y métricas válidas.
**Aceptación:** elección respaldada por mediciones. No sumar la memoria de las
dos GPU ni distribuir una sola red pequeña entre ambas por defecto.

### G4.2 — Reproducción obligatoria completa

**Responsable:** Gael. **Fecha:** 02/11. **Duración:** 60–90 min de preparación
y control, más cómputo. **Insumos:** pruebas anteriores aprobadas y piloto.

1. Ejecutar las tres variantes y los seis objetivos con el protocolo congelado.
2. Validar registros, soportes, candidatas, cobertura OOF y excepciones.

**Entregable:** corrida completa identificada y verificación de integridad.
**Comprobación:** pruebas de registros y conteo de ajustes por objetivo.
**Aceptación:** 18 combinaciones completas y 981 ajustes centrales; 27 folds
externos por variante. GDS conserva configuración base y búsqueda omitida.

### G4.3 — Primera versión entregable

**Responsable:** Gael. **Fecha:** 02/11. **Duración:** 90–120 min. **Insumos:**
corrida obligatoria y documentación docente.

1. Preparar seis tablas de tres variantes, con todas las métricas y notas GDS.
2. Redactar datos, convolución, CORAL, metodología y respuestas mínimas provisionales.

**Entregable:** tablas agregadas y borrador del informe.
**Comprobación:** regeneración de tablas y lista de requisitos de la sección 7.
**Aceptación:** comparación obligatoria completa y explicable, sin depender de
que estén terminadas las extensiones.

### J3.1 — Modelos para localidad

**Responsable:** Johan. **Fecha:** 03/11. **Duración:** 90–120 min. **Insumos:**
tronco docente y arquitectura de capacidad próxima.

1. Preparar CNN-k3, CNN-k1 y la MLP de dos ocultas en su módulo de localidad.
2. Comprobar formas, parámetros y entrenamiento breve de cada modelo.

**Entregable:** modelos y `test_localidad.py`.
**Comprobación:** `python -m unittest discover -s tests -p 'test_localidad.py' -v`.
**Aceptación:** tres modelos Softmax con formas correctas, MLP de 17231
parámetros y CNN-k3 de 17219; configuración restante fijada según protocolo.

### J3.2 — Órdenes y prueba del panel

**Responsable:** Johan. **Fecha:** 03/11. **Duración:** 60–120 min. **Insumos:**
modelos de localidad, permutaciones exactas y particiones exportadas.

1. Implementar órdenes de la sección 3.4 y guardar nombres/índices.
2. Generar lista de 405 ajustes y ejecutar una prueba corta de los nueve pares modelo/orden.

**Entregable:** configuración, lista de trabajos y prueba corta.
**Comprobación:** suite `test_localidad.py` y verificación de aplicación de orden.
**Aceptación:** ninguna columna se pierde/duplica, mismo orden en train y test,
semillas independientes y ausencia de búsquedas internas en este panel fijo.

### G5.1 — Decidir por moda y mediana

**Responsable:** Gael. **Fecha:** 05/11. **Duración:** 60–90 min. **Insumos:**
probabilidades externas de las tres variantes obligatorias.

1. Añadir decisiones alternativas sobre los mismos archivos OOF.
2. Calcular cambios pareados por fila y fold y comprobar el caso binario.

**Entregable:** análisis reproducible y `test_decisiones.py`.
**Comprobación:** `python -m unittest discover -s tests -p 'test_decisiones.py' -v`.
**Aceptación:** cero nuevos entrenamientos, probabilidades idénticas entre
reglas y decisiones equivalentes en R3 con el desempate definido.

### G5.2 — Predictor por perfil

**Responsable:** Gael. **Fecha:** 05/11. **Duración:** 90–120 min. **Insumos:**
datos y particiones comunes, regla de suavizado y respaldo.

1. Implementar frecuencias suavizadas por perfil y respaldo global de entrenamiento.
2. Probar perfiles conocidos, nuevos y cambio de etiquetas de test.
3. Evaluar los seis objetivos con los folds externos centrales.

**Entregable:** predictor en `experimentos/perfiles.py` y `test_perfiles.py`.
**Comprobación:** `python -m unittest discover -s tests -p 'test_perfiles.py' -v`.
**Aceptación:** probabilidades válidas, moda/mediana y marca visto/nuevo;
ningún conteo depende de etiquetas externas. Los perfiles no se eliminan.

### G5.3 — Interpretación de decisiones y perfiles

**Responsable:** Gael. **Fecha:** 05/11. **Duración:** 90–120 min. **Insumos:**
G5.1, G5.2 y resultados obligatorios.

1. Comparar decisiones, errores graves, dirección del error y recalls.
2. Describir errores y soporte en perfiles vistos/nuevos sin seleccionar reglas con test.

**Entregable:** dos figuras regenerables y borrador de discusión.
**Comprobación:** reconstruir análisis desde los artefactos, con pares de filas completos.
**Aceptación:** separar cambios de decisión de cambios de aprendizaje y señalar
que la ambigüedad empírica no es una estimación del límite poblacional.

### J4.1 — Ejecutar localidad

**Responsable:** Johan. **Fecha:** 06/11. **Duración:** 60–90 min de preparación
y revisión, más cómputo. **Insumos:** J3.2 y configuración del piloto.

1. Ejecutar los 405 ajustes con órdenes y semillas prefijados.
2. Registrar resultados completos y verificar cobertura OOF por condición.

**Entregable:** corrida identificada del panel.
**Comprobación:** lista prevista frente a ejecutada, pruebas de localidad y registros.
**Aceptación:** cada modelo/orden/partición/inicialización tiene cinco folds;
fallos visibles y ninguna condición sustituida por la de mejor rendimiento.

### J4.2 — Interpretar localidad

**Responsable:** Johan. **Fecha:** 06/11. **Duración:** 90–120 min. **Insumos:**
panel completo y conteos de parámetros.

1. Graficar MAE por arquitectura y orden mostrando partición e inicialización.
2. Comparar k3 con k1 y MLP y redactar alcance de las diferencias.

**Entregable:** figura reproducible y sección de metodología/resultados/discusión.
**Comprobación:** regeneración sin entrenamiento y cotejo de todas las condiciones.
**Aceptación:** distinguir localidad, capacidad y optimización; no confundir
reentrenar con permutar en inferencia ni anunciar un orden universalmente óptimo.

### H5.1 — Robustez entre particiones

**Responsable:** Gael. **Fecha:** 08/11. **Duración:** 90–120 min más cómputo.
**Insumos:** corrida central y semillas adicionales previstas.

1. Repetir selección/refit de tres variantes en R1 y R2, con particiones 314/2718 e init 42.
2. Comparar métricas y frecuencia de selección del grid con la referencia 42.

**Entregable:** 780 ajustes adicionales y análisis de estabilidad.
**Comprobación:** integridad de registros, mismas reglas y tres particiones por comparación.
**Aceptación:** no elegir una semilla ganadora ni tratar repeticiones como pacientes nuevos.

### H5.2 — Integrar resultados

**Responsables:** Gael y Johan. **Fecha:** 08/11. **Duración:** 90–120 min.
**Insumos:** estudios y borradores completos.

1. Integrar tablas, figuras y metodología con identificadores de corrida.
2. Revisar afirmaciones contra artefactos y separar hipótesis de hallazgos.

**Entregable:** agregados seleccionados en `docs/resultados/` y discusión unificada.
**Comprobación:** regeneración y revisión de unidades, soportes y leyendas.
**Aceptación:** cada afirmación importante identifica objetivo, métrica,
comparación y límite; incluir resultados desfavorables y no mezclar tareas distintas.

### H6.1 — Completar informe

**Responsables:** Gael y Johan. **Fecha:** 10/11. **Duración:** dos bloques de
90–120 min, uno por integrante. **Insumos:** borradores y agregados finales.

1. Cada integrante termina sus borradores según el reparto de la sección 7.
2. Ambos juntan las secciones, revisan el texto completo, las referencias y el formato.
3. Comprobar requisitos, siete preguntas mínimas y correspondencia de figuras/tablas.

**Entregable:** informe PDF y fuentes.
**Comprobación:** compilación/exportación, inspección del PDF y checklist de sección 7.
**Aceptación:** texto y gráficos legibles, cifras trazables y ninguna pregunta
obligatoria sin respuesta. Describir límites, sin convertir el trabajo en diagnóstico clínico.

### H6.2 — Reproducción

**Responsables:** Gael y Johan. **Fecha:** 10/11. **Duración:** 90–120 min más
cómputo. **Insumos:** código y manifiestos de corridas.

1. Ejecutar pruebas y corrida corta desde entorno limpio con datos locales.
2. Regenerar tablas/figuras finales desde los artefactos completos conservados.
3. Revisar el comando documentado para reproducir los estudios completos.

**Entregable:** registro de reproducción con ejecutable, versiones y comandos.
**Comprobación:** pruebas aprobadas, código de salida y cotejo de agregados.
**Aceptación:** dependencias documentadas y pasos ejecutables; distinguir
regeneración de reportes de reentrenamiento completo. No prometer igualdad bit a bit entre dispositivos.

### H6.3 — Preparar entrega

**Responsables:** Gael y Johan. **Fecha:** 10/11. **Duración:** 60–90 min.
**Insumos:** informe y reproducción aprobados.

1. Preparar ZIP con código y documentación, siguiendo los requisitos del curso.
2. Revisar contenido y nombres, excluyendo datos individuales, entornos y salidas voluminosas.

**Entregable:** ZIP, PDF y repositorio coherentes.
**Comprobación:** listar y abrir el ZIP, abrir el PDF y verificar revisión de código citada.
**Aceptación:** archivos íntegros, autores correctos y ningún resultado de otra corrida mezclado.

### F1 — Revisión final

**Responsables:** Gael y Johan. **Fecha:** 11/11. **Duración:** 60–90 min.
**Insumos:** archivos finales.

1. Corregir erratas, comprobar accesos del repositorio y revisar el checklist final.
2. Conservar una copia identificada de los archivos que se entregarán.

**Entregable:** versión final revisada. **Comprobación:** checklist de sección 7.
**Aceptación:** no quedan correcciones necesarias ni archivos pendientes de generar.

### F2 — Entrega

**Responsables:** Gael y Johan. **Fecha:** 12/11, antes de las 23:59
America/Santiago. **Duración:** 30–60 min. **Insumos:** entrega revisada.

1. Subir los archivos por el mecanismo del curso y comprobar su recepción.
2. Registrar fecha/hora y evidencia de la entrega.

**Entregable:** entrega recibida y comprobante. **Aceptación:** PDF, ZIP y
repositorio disponibles según requisitos, dentro del plazo.

## 6. Cómputo, presupuesto y coordinación

Recursos informados: acceso continuo al servidor, 64 núcleos, aproximadamente
503 GB de RAM y dos A5000 con 24 GB cada una. Verificar el entorno y medir
rendimiento en el piloto antes de fijar concurrencia. Preferir trabajos
independientes por proceso, limitar hilos y evitar sobresuscripción de CPU.

| Bloque | Ajustes previstos |
|---|---:|
| Obligatorio: cinco recodificaciones × tres variantes × 65, más GDS | 981 |
| Cuatro controles R2 con grid 5×3 | 260 |
| Localidad con configuraciones fijas | 405 |
| Dos particiones adicionales R1/R2, tres variantes | 780 |
| Constantes centrales y predictor por perfil | 27 + 27, de bajo costo |
| Total de referencia, antes de pruebas cortas y piloto | 2480 |

Un ajuste de grid viable cuesta `5 × (4 × 3 + 1) = 65` por procedimiento y
objetivo. Reservar inicialmente aproximadamente 3000 ajustes para el estudio
y las comprobaciones, sin convertir el margen en nuevas familias de modelos.
No fijar horas de cómputo sin mediciones. Moda y mediana de una misma
distribución no cuentan como dos entrenamientos.

Esfuerzo humano orientativo: Gael 40–50 horas y Johan 20–30 horas; revisar las
estimaciones después de G2. Los tiempos de las fichas son orientativos.
Después de cada sesión registrar evidencia, siguiente paso y bloqueos. Dividir
una ficha si excede dos horas de trabajo activo.

## 7. Informe y aceptación final

### Reparto de las secciones

Propuesta basada en la plantilla del laboratorio, pendiente de confirmar entre
ambos. La columna de responsable indica quién prepara el primer borrador;
el informe completo se revisa entre los dos.

| Sección | Primer borrador | Máximo según la plantilla |
|---|---|---|
| Resumen | Ambos, al terminar el resto del informe | No indicado |
| Introducción | Johan: problema, objetivos y herramientas utilizadas | 1 página |
| Marco teórico | Gael explica Conv1D y CORAL; Johan, los modelos de comparación | 1 página |
| Metodología | Gael describe datos y evaluación; cada uno explica sus experimentos | 2 páginas |
| Resultados | Cada uno presenta las tablas, gráficos y explicación de sus experimentos | 3 páginas |
| Discusión | Ambos: qué significan los resultados, sus diferencias y limitaciones | 1 página |
| Conclusiones | Ambos: principales hallazgos y aprendizajes | 1 página |
| Código y reproducibilidad | Cada uno explica su código; ambos comprueban cómo repetir el trabajo | 1 página |

En metodología y resultados, Gael cubre CORAL, comparación obligatoria,
decisiones, perfiles y robustez; Johan cubre los modelos de comparación y el
efecto del orden de las preguntas. Incluir las seis tablas obligatorias, las
configuraciones elegidas y su estabilidad, además de responder las siete
preguntas siguientes.

Ambos completan las referencias y la rúbrica de autoevaluación de la plantilla.
Los anexos son opcionales y pueden reunir tablas completas, detalles de pruebas
y gráficos adicionales. Al juntar el informe, ambos revisan que las partes
conecten, se respeten los límites de páginas y el PDF final sea legible.

### Siete preguntas mínimas

- [ ] Qué patrones puede aprender un kernel 3 sobre los 15 ítems.
- [ ] Qué ocurre al permutar columnas en Conv1D y MLP, distinguiendo reentrenamiento e inferencia.
- [ ] Por qué usar Flatten y no pooling global en la referencia.
- [ ] Cuántos parámetros tiene la red y dónde se concentran.
- [ ] Qué configuraciones elige el grid y cuán estable es entre folds.
- [ ] Cómo se garantiza la consistencia ordinal de CORAL.
- [ ] En qué objetivos CORAL cambia MAE y errores graves frente a Softmax.

### Figuras principales

1. Clases y soporte por partición, destacando la etapa 7 de GDS.
2. Arquitectura, formas y distribución de parámetros.
3. Comparación obligatoria con valores externos y dispersión.
4. Cambios de moda a mediana sobre las mismas probabilidades.
5. Localidad: arquitectura, orden, partición e inicialización.
6. Errores y soporte según perfiles vistos/nuevos en entrenamiento.

Las confusiones OOF restantes, métricas probabilísticas y detalles del grid
pueden acompañar las figuras o ir en anexos; no reemplazan las tablas obligatorias.

### Checklist de entrega

- [ ] Tres variantes × seis objetivos completas, o cualquier fallo claramente identificado y corregido.
- [ ] Todas las métricas externas, con media, desviación y convención documentadas.
- [ ] GDS con configuración base y búsqueda interna omitida, correctamente rotulado.
- [ ] Probabilidades, splits, candidatas y revisión de código conservados.
- [ ] Confusiones agregadas de todos los folds y clases originales identificadas.
- [ ] Estudios previstos y resultados favorables/desfavorables incluidos.
- [ ] Tablas y figuras regenerables desde las corridas conservadas.
- [ ] PDF legible, siete preguntas contestadas y referencias verificadas.
- [ ] Pruebas y reproducción registradas; ZIP y repositorio coherentes con el PDF.
- [ ] Datos individuales y archivos ajenos a la entrega excluidos del ZIP y de Git.
- [ ] Acceso del profesor y ayudantes al repositorio comprobado.
- [ ] Entrega y recepción registradas antes del 12/11/2026, 23:59 America/Santiago.

## 8. Registro de avances

Añadir una fila después de cada sesión. Registrar resultados observables; una
prueba corta no convierte un experimento final en terminado.

| Ficha | Responsable | Avance y evidencia | Falta o siguiente paso |
|---|---|---|---|
| H0.1 | Gael | 23/10: plan de trabajo y enlace desde README revisados | H0.2 |
| H0.2 | Gael | 25/10: entorno verificado y prueba Softmax/R2 completada; dos folds externos, métricas finitas y reportes legibles | H0.3 con Johan |
| H0.3 | Gael y Johan | 25/10: revisión técnica completada y reparto general confirmado por Johan; informe compartido. Se mantienen las tareas y fechas del tablero | Gael: G1.1/G1.2. Johan: J1.1/J1.2 |
| G1.1 | Gael | 26/10: auditoría reproducible y cinco pruebas aprobadas; resumen en la guía de datos | G1.2 |
| G1.2 | Gael | 26/10: formas, parámetros y ventanas comprobadas; cinco pruebas aprobadas | G2.1/G2.2 |
| J1.1 | Johan | 27/10: Implementación de moda con desempate a índice menor, mediana y distribución previa + pruebas | J1.2 |
| J1.2 | Johan | 27/10: Implementación para suma de bits y 15 entradas, barriendo las 4 candidatas de $C$ con selección MAE/QWK y pruebas aprobadas. | J2.1/J2.2/J2.3 |
| G2.1 | Gael | 27/10: niveles acumulativos e inferencia implementados; pruebas para K=2,3,7 y umbrales aprobadas | G2.2 y G2.3 |
| G2.2 | Gael | 28/10: cabeza CORAL y red conectadas; formas, orden antes/después de optimización y gradientes finitos comprobados para K=2,3,7; suite completa aprobada | G2.3 |
| G2.3 | Gael | 28/10: pérdida BCE con logits, suma de umbrales y promedio del batch; equivalencia binaria, pesos unitarios y estabilidad comprobadas | G2.4 |

### Registro de arranque H0.2 — 25/10/2026

Entorno `lab_pytorch`: Python 3.10.21, PyTorch 2.14.0 y NumPy 2.2.6.
Prueba en CPU con semilla 42, una época y dos folds externos e internos.
Terminó sin errores en 9.37 s. Es una prueba de funcionamiento, no un resultado final.
El SHA-256 del SAV coincide con la referencia:
`b454a4e5c997be26edb3a187605c2c10da1147ac1ce6cea76d42adbe68b17474`.

Comando desde la raíz del repositorio, con el entorno activo:

```bash
python main.py --data-path 'dataset/15 atributos R0-R5.sav' --target-name GDS_R2 --algorithm conv1d_softmax --epochs 1 --no-grid --outer-folds 2 --inner-folds 2 --output-dir results/h0_arranque
```

Log, versiones y reportes conservados localmente en `results/h0_arranque/`.
El comando exacto ejecutado está en `comando.txt` de esa carpeta.
`--no-grid` mantiene la evaluación interna. La confusión muestra solo el último fold.

## 9. Fuentes de referencia

- Enunciado del Laboratorio 02, diapositivas 30–40 para protocolo, entregables y evaluación.
- [Problema y datos](docs/01_problema.md), [arquitectura](docs/02_arquitectura.md),
  [uso](docs/03_uso.md) y [extensión ordinal](docs/04_extension_ordinal.md) de la plantilla.
- [CORAL: Cao, Mirjalili y Raschka](https://arxiv.org/html/1901.07884v7).
- [Número efectivo: Cui y colaboradores](https://arxiv.org/html/1901.05555v1).
- [Selección y sesgo de evaluación: Cawley y Talbot](https://www.jmlr.org/papers/v11/cawley10a.html).
- [ACE-III e información de ítems: Calderón y colaboradores](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0251137).
- [Manual de INECO](https://www.fundacionineco.org/wp-content/uploads/Manual-de-buenas-practicas-para-el-diagnostico-de-demencias.pdf),
  como referencia de las coincidencias con los elementos de memoria. No acredita
  por sí solo la procedencia del SAV.
