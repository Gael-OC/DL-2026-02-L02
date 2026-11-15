# Laboratorio 02: redes convolucionales 1D

**Deep Learning · Universidad Católica del Norte · 2026-02**

**Equipo:** Gael Ortega y Johan Piñones.

Comparación de Conv1D + Softmax, Conv1D + CORAL y Conv1D + CORAL ponderada
para predecir GDS y sus cinco recodificaciones a partir de 15 respuestas
binarias. Los modelos comparten el tronco convolucional; cambian la cabeza
y la pérdida de entrenamiento.

## Resultados principales

La comparación obligatoria está completa: tres variantes en seis objetivos, con
981 ajustes y las nueve métricas multiclase y ordinales solicitadas. Medias de
los folds externos (S = Softmax, C = CORAL, P = CORAL ponderada); las
desviaciones y las demás métricas están en las
[seis tablas](results/resumen/central/comparacion.md).

| Objetivo | MAE S | MAE C | MAE P | QWK S | QWK C | QWK P | BalAcc S | BalAcc C | BalAcc P |
|---|---|---|---|---|---|---|---|---|---|
| `GDS` | 0.474 | 0.689 | 0.789 | 0.705 | 0.685 | 0.678 | 0.346 | 0.368 | 0.442 |
| `GDS_R1` | 0.105 | 0.111 | 0.130 | 0.690 | 0.631 | 0.635 | 0.735 | 0.686 | 0.649 |
| `GDS_R2` | 0.290 | 0.293 | 0.298 | 0.694 | 0.688 | 0.692 | 0.639 | 0.647 | 0.643 |
| `GDS_R3` | 0.092 | 0.099 | 0.099 | 0.587 | 0.569 | 0.584 | 0.753 | 0.751 | 0.768 |
| `GDS_R4` | 0.171 | 0.294 | 0.372 | 0.278 | 0.411 | 0.385 | 0.506 | 0.667 | 0.690 |
| `GDS_R5` | 0.232 | 0.374 | 0.437 | 0.414 | 0.477 | 0.476 | 0.503 | 0.618 | 0.648 |

- Softmax tiene el menor MAE en los seis objetivos, pero en `GDS_R4` y `GDS_R5`
  casi no predice la clase menos frecuente del extremo bajo.
- CORAL y CORAL ponderada mejoran balanced accuracy y QWK en esos dos objetivos,
  a costa de MAE.
- En `GDS_R2`, una regresión logística sobre la suma de respuestas incorrectas
  iguala a la red, y ni el kernel de tamaño 3 ni el orden de las columnas dan
  una ventaja estable.
- Frente a la MLP del Laboratorio 01, la capa convolucional no reduce el MAE de
  forma sistemática.

Los resultados son descriptivos: no constituyen un diagnóstico clínico.
GDS tiene solo dos casos de etapa 7: usa dos folds externos, configuración
base y búsqueda interna omitida. Las cinco recodificaciones usan validación
anidada 5×3 y cuatro candidatas seleccionadas por MAE, con desempate por QWK.

## Estudios adicionales

| Estudio | Material disponible |
|---|---|
| Comparación obligatoria | [Seis tablas, confusiones OOF y selección del grid](results/resumen/central/comparacion.md) |
| Controles en R2 | [Logísticas, MLP, boosting y constantes; 260 ajustes de selección/refit](results/resumen/controles/comparacion.md) |
| Constantes en los seis objetivos | [Moda y mediana de entrenamiento externo](results/resumen/constantes/resumen_folds.csv) |
| Decisiones y perfiles | [Comparaciones pareadas, soporte, recalls y figuras](results/resumen/analisis/discusion.md) |
| Localidad y orden | [CNN-k3, CNN-k1 y MLP; 405 ajustes](results/resumen/localidad/completo/interpretacion.md) |
| Robustez | [R1 y R2 con tres particiones; 780 ajustes adicionales](results/resumen/robustez/robustez.md) |

El análisis y las respuestas a las siete preguntas del enunciado se entregan en
el informe técnico (PDF). El [mapa de evidencia](docs/resultados.md) relaciona cada requisito y pregunta con
su código, tablas y comprobaciones.

## Instalación y comprobación

La suite conserva 14 comprobaciones esenciales de CORAL, particiones, selección
con refit, métricas y cobertura OOF, con casos puntuales de perfiles, constantes
y permutaciones. Usa datos sintéticos y no necesita el SAV ni corridas privadas.

```bash
conda env create -f environment.yml
conda activate lab_pytorch
# Copiar el SAV docente a dataset/ antes de ejecutar experimentos.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m unittest discover -s tests -v

# Prueba corta de ambos algoritmos; no produce resultados finales.
python main.py --data-path "dataset/15 atributos R0-R5.sav" --algorithm all --epochs 1 --no-grid --outer-folds 2 --inner-folds 2 --output-dir results/prueba_entorno
```

Cada configuración debe usar una carpeta de salida propia. El mismo comando
puede recuperar folds completos si su manifiesto coincide. El SAV y las
predicciones individuales se conservan localmente y no se incluyen en Git.

## Reproducir la comparación obligatoria

Desde la raíz, con `lab_pytorch` activo:

```bash
python main.py --data-path "dataset/15 atributos R0-R5.sav" --all-targets --algorithm all --output-dir results/g4_2_central_unweighted
python main.py --data-path "dataset/15 atributos R0-R5.sav" --all-targets --algorithm conv1d_coral --use-weights --output-dir results/g4_2_central_weighted

# Verificar las corridas conservadas y regenerar agregados sin reentrenar.
python -m experimentos.auditar_corrida results/g4_2_central_unweighted --eventos 654 --data-path "dataset/15 atributos R0-R5.sav"
python -m experimentos.auditar_corrida results/g4_2_central_weighted --eventos 327 --data-path "dataset/15 atributos R0-R5.sav"
python -m experimentos.comparacion results/g4_2_central_unweighted results/g4_2_central_weighted --output results/resumen/central
```

Las redes usan Adam, 20 épocas, batch 32, tasa de aprendizaje 0.001 y
weight decay 0.0001. CORAL ponderada utiliza beta 0.99 y conteos exclusivos
del entrenamiento de cada ajuste. El test externo se evalúa después de la
selección y el refit; no decide hiperparámetros, épocas ni semillas.

Los pasos para reconstruir los demás estudios, trasladar datos y regenerar
figuras están en [uso](docs/03_uso.md) y [resultados](results/README.md).
Una copia de Git contiene código y agregados; las corridas individuales deben
reconstruirse con el SAV o trasladarse de forma privada.

## Organización

| Ruta | Contenido |
|---|---|
| `main.py` | CLI y configuración de los experimentos centrales |
| `src/` | Datos, modelos, tareas, entrenamiento, métricas y reportes |
| `experimentos/` | Controles, estudios adicionales y auditorías |
| `tests/` | Pruebas del contrato de datos, modelos y evaluación |
| `results/resumen/` | Tablas, figuras y auditorías agregadas para consulta |
| `docs/` | Explicaciones técnicas, reproducción y mapa de evidencia |
| `dataset/` | SAV local, excluido de Git |

## Documentación

- [Problema, datos y recodificaciones](docs/01_problema.md).
- [Convolución, formas y parámetros](docs/02_arquitectura.md).
- [Entorno, comandos y reproducción](docs/03_uso.md).
- [Implementación ordinal CORAL](docs/04_extension_ordinal.md).
- [Mapa de evidencia: requisitos, preguntas y comprobaciones](docs/resultados.md).
