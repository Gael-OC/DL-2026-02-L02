# Resultados

Las corridas completas se conservan localmente y quedan ignoradas por Git:
manifiestos, ajustes, splits, predicciones individuales, probabilidades y logs.
No añadir archivos por fila ni el SAV al repositorio. Se versionan únicamente
los agregados seleccionados de `resumen/`; el CSV histórico de controles ya
versionado se conserva y requiere revisión del protocolo antes de integrarlo.

## Comparación central

`g4_2_central_unweighted/` contiene Softmax y CORAL; `g4_2_central_weighted/`,
CORAL ponderada. Entre ambas hay 18 combinaciones y 981 ajustes. Las seis
tablas, confusiones OOF y frecuencias del grid están en
[resumen/central/](resumen/central/), con [tablas legibles](resumen/central/comparacion.md).

```bash
python -m experimentos.auditar_corrida results/g4_2_central_unweighted --eventos 654
python -m experimentos.auditar_corrida results/g4_2_central_weighted --eventos 327
python -m experimentos.comparacion results/g4_2_central_unweighted results/g4_2_central_weighted --output results/resumen/central
```

La auditoría es de solo lectura. La comparación genera únicamente agregados;
no vuelve a entrenar. El resumen externo usa media y desviación con `ddof=0`;
no equivale al agregado de todas las filas OOF. GDS usa dos folds externos y
configuración base con búsqueda omitida. Las cinco recodificaciones usan 5×3
folds y cuatro candidatas. NLL y RPS complementan las nueve métricas exigidas.

## Decisiones por moda y mediana

`python -m experimentos.decisiones` reutiliza las probabilidades de las
corridas centrales sin entrenar modelos. Guarda agregados por fold y OOF en
`resumen/decisiones/`; las predicciones individuales permanecen locales.

## Regeneración y entorno

Activar `lab_pytorch` y ejecutar desde la raíz del repositorio. Para limitar
recursos, usar `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.
Las corridas se pueden retomar con el comando original y el mismo manifiesto;
no mezclar código/configuración distintos dentro de una carpeta existente.
La auditoría requiere acceso al SAV original cuyo hash figura en el manifiesto.

`python -m experimentos.reportes CARPETA` reconstruye exportaciones OOF y
reportes de una corrida completa, sin entrenamiento. Las confusiones OOF
agregadas son la referencia; las imágenes históricas de `main.py` rotuladas
como último fold no representan todos los folds.
