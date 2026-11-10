# Evidencia para preparar el informe

Este documento organiza el material técnico del laboratorio. La redacción,
la estructura y la maquetación del informe se harán después con el formato
específico del profesor. No se incorpora el PDF ni la fuente IEEE del respaldo.

## Cumplimiento del enunciado

Referencia: `DL_2026_02_L02.pdf`, diapositivas 7 y 30–40. El objetivo es
implementar y comparar modelos categóricos y ordinales con Conv1D en los seis
objetivos; no exige que CORAL supere a Softmax. Los controles y los estudios
adicionales amplían esa comparación, sin sustituir las tablas obligatorias.

| Requisito | Evidencia disponible | Pendiente para la entrega |
|---|---|---|
| Representar las 15 respuestas como secuencia | [Datos](01_problema.md), [arquitectura](02_arquitectura.md), `src/data/dataset.py` | Explicarlo en el informe |
| Softmax y CORAL sobre el mismo tronco | `src/models/blocks.py`, `src/models/conv_softmax.py`, `src/models/coral.py` | Presentar arquitectura y formas |
| Niveles, pérdida, inferencia y pesos ordinales | [Implementación CORAL](04_extension_ordinal.md), `src/tasks/ordinal_task.py`, `tests/test_coral.py` | Justificar consistencia y ponderación |
| Selección interna MAE/QWK y refit externo | `src/training/search.py`, `src/training/nested_cv.py`, registros auditados | Explicar protocolo y excepción GDS |
| Tres variantes × seis objetivos y nueve métricas externas | [Tablas centrales](../results/resumen/central/comparacion.md) | Llevar las seis tablas al formato docente |
| Análisis del grid | [Configuraciones elegidas](../results/resumen/central/configuraciones.csv), [robustez](../results/resumen/robustez/selecciones.csv) | Sintetizar variación entre folds |
| Siete preguntas mínimas | Mapa de la sección siguiente | Redactar y revisar las respuestas finales |
| Código modular, ZIP, repositorio e informe PDF | Código, pruebas, agregados y comandos en este repositorio | Informe, paquete, acceso docente y recepción |

La parte técnica obligatoria tiene implementación y resultados. El laboratorio
no está entregado ni se considera terminado como documento académico.

## Siete preguntas: dónde buscar la evidencia

| Pregunta de la diapositiva 37 | Material para responder |
|---|---|
| 1. Patrones locales de un kernel 3 | [Ventanas y campo receptivo](02_arquitectura.md); `experimentos/inspeccionar_arquitectura.py` |
| 2. Efecto de permutar columnas en CNN y MLP | [Panel de localidad](../results/resumen/localidad/completo/interpretacion.md), [comparaciones por semillas](../results/resumen/localidad/completo/comparacion_pareada.csv) |
| 3. Flatten frente a pooling global | [Explicación de arquitectura](02_arquitectura.md#por-que-flatten-y-no-pooling-global) y prueba de conservación de posiciones |
| 4. Parámetros y su concentración | [Desglose de la red](02_arquitectura.md); base Softmax K=3: 17219, de los cuales 15392 están en la capa 480→32 |
| 5. Configuración elegida y estabilidad | [Frecuencias centrales](../results/resumen/central/configuraciones.csv), [selecciones entre particiones](../results/resumen/robustez/selecciones.csv) |
| 6. Consistencia ordinal CORAL | [Cabeza y umbrales](04_extension_ordinal.md), `CoralLayer` y pruebas de orden/gradientes |
| 7. En qué objetivos cambia MAE y errores graves | [Seis tablas](../results/resumen/central/comparacion.md), con las decisiones originales de cada variante |

Dos capas k=3 tienen un campo receptivo local de cinco posiciones antes de la
capa densa. Flatten conserva las posiciones y la parte densa combina todas.
Permutar solo en inferencia cambia las entradas esperadas; el panel, en cambio,
reentrena desde cero y aplica cada orden tanto a train como a test. La capacidad
de una MLP permite reparametrizar una permutación de entradas, pero su
entrenamiento concreto no tiene por qué producir resultados idénticos.

## Comparación obligatoria

- 981 ajustes: 654 de Softmax/CORAL y 327 de CORAL ponderada.
- 18 combinaciones y 81 folds externos; una predicción OOF por fila y variante.
- Cinco recodificaciones con validación 5×3 y cuatro candidatas docentes.
- GDS: dos externos, configuración base y búsqueda interna omitida por soporte
  mínimo de dos casos. No llamarlo validación anidada 2×2.
- Tablas: media y desviación externas con `ddof=0`; confusiones agregadas de
  todos los folds y NLL/RPS complementarios.

En la referencia central, Softmax tiene menor MAE externo medio en los seis
objetivos. CORAL tampoco reduce los errores graves frente a Softmax; hay
igualdad cuando ambas tasas son cero. Es un resultado de estos datos y protocolo,
no una demostración de superioridad universal.

## Controles y constantes

Los [controles corregidos de R2](../results/resumen/controles/comparacion.md)
usan las particiones centrales, las tres validaciones internas, selección
MAE/QWK y refit con todo el entrenamiento externo. Son 240 ajustes internos
más 20 refits. La MLP usa 20 épocas, y las constantes reciben índices 0..K−1.
La tabla del antiguo ejecutor no es la comparación de referencia.

La [moda y mediana constantes](../results/resumen/constantes/resumen_folds.csv)
están disponibles para los seis objetivos. Se calculan solo con las etiquetas
de entrenamiento externo, sin suavizado y con empate hacia la clase menor.
Estas distribuciones no son el respaldo suavizado del predictor por perfil.

## Decisiones y perfiles

La [discusión de decisiones y perfiles](../results/resumen/analisis/discusion.md)
reúne pares sobre las mismas probabilidades, recalls con soporte y errores
firmados. El [predictor por perfil](../results/resumen/perfiles/oof.csv) usa
frecuencias de entrenamiento con suavizado aditivo uno; los perfiles nuevos
recurren a las frecuencias globales de ese entrenamiento.

En Softmax/R2, cambiar de moda a mediana reduce el MAE OOF de 0.2895 a 0.2833,
con 33 decisiones distintas. El cambio no mejora todos los objetivos. En CORAL,
la mediana coincide con la decisión original en estas predicciones. NLL/RPS
no cambian al mantener las probabilidades y cambiar solo la decisión.

Las 6714 predicciones de perfil se alinean con 20142 de las redes. Visto/nuevo
indica presencia del perfil de respuestas en entrenamiento; no identifica
personas nuevas. Los soportes y la composición de clases deben acompañar
cualquier comparación de grupos.

## Localidad y robustez

El [panel de localidad](../results/resumen/localidad/completo/interpretacion.md)
compara CNN-k3, CNN-k1 y MLP con tres órdenes, tres particiones, tres
inicializaciones y cinco folds: 405 ajustes, 81 condiciones OOF.
Los parámetros para K=3 son 17219, 16163 y 17231, respectivamente.
Las diferencias cambian de signo según la condición: no hay evidencia aquí
para afirmar una ventaja universal del kernel 3 ni del orden original.
La [síntesis de localidad](../analisis_de_localidad_y_ordenes.md) usa estos agregados.

La [robustez de R1/R2](../results/resumen/robustez/robustez.md) añade 780 ajustes
con particiones 314/2718 e inicialización base 42. En R1, Softmax conserva menor
MAE en las tres particiones; en R2, la comparación cambia de signo en 314.
NLL y RPS favorecen a Softmax en ambas tareas y las tres particiones.

## Comprobación y límites

Comprobaciones realizadas sobre el código integrado:

La suite actual conserva 14 pruebas esenciales con datos sintéticos. Se
retiraron los casos repetidos y la infraestructura de pruebas de portabilidad
y reanudación. Las auditorías de corridas siguen disponibles como comandos.

- Regeneración de centrales, robustez, perfiles y decisiones con igualdad de CSV.
- Controles: 260 ajustes y selección/refit auditados; localidad: 405 folds,
  81 condiciones, cobertura, probabilidades y pérdidas de 20 épocas verificadas.
- Constantes: 6714 decisiones y probabilidades coincidentes con la referencia
  local, recalculadas exclusivamente desde entrenamiento externo.
- Enlaces locales y columnas de los nuevos agregados revisados.


La [guía de resultados](../results/README.md) describe cómo auditar y regenerar
cada bloque. La integración conserva el núcleo `main.py`, `src/` y
`experimentos/reportes.py` para mantener la identidad de las corridas centrales.
Los ejecutores de controles y localidad coinciden con los hashes de sus
artefactos locales. Las auditorías se ejecutan explícitamente al verificar
las corridas; no forman parte de la suite reducida. La reproducción final se
revisará en H6.2.

Los folds y particiones comparten observaciones; su dispersión no es un intervalo
de confianza. Las seis etiquetas provienen de las mismas filas y no son seis
muestras independientes. No se atribuye causalidad, capacidad diagnóstica ni
procedencia confirmada del instrumento a estos resultados. Las clases extremas
y los perfiles ambiguos limitan su interpretación.

La revisión del texto, las referencias, la autoevaluación y los límites de
páginas quedan para el informe conforme a la plantilla vigente del profesor.
