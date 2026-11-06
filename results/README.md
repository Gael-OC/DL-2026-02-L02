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

## Predictor por perfil

El cómputo y los artefactos se mantienen en local.
`experimentos.perfiles` usa las 15 respuestas en su orden docente y no
elimina perfiles repetidos. Cuenta etiquetas solo del entrenamiento externo:
`p(c|perfil) = (n_c + 1) / (n_perfil + K)`. Un perfil nuevo usa los conteos
globales de ese entrenamiento con el mismo suavizado. Moda y mediana inferior
reutilizan las reglas de decisiones; los empates eligen el índice menor.

Corrida completada el 06/11/2026, desde la raíz y con `lab_pytorch` activo:

```bash
python -m experimentos.perfiles --data-path 'dataset/15 atributos R0-R5.sav' --splits-dir results/g4_2_central_unweighted --local results/g5_2_perfiles --out results/resumen/perfiles --reference results/g5_perfiles/filas.csv
python -m unittest discover -s tests -p 'test_perfiles.py' -v
python -m unittest discover -s tests -v
```

Se verificaron 6714 predicciones (1119 por objetivo), 27 folds externos y
coincidencia exacta con las filas previas del modelo `perfil`. Siete pruebas
específicas y la suite de 53 pruebas aprobaron. No se entrenan redes: GDS
reutiliza dos folds y las cinco recodificaciones, cinco cada una, con semilla 42.
La marca visto/nuevo corresponde a cada entrenamiento externo: 5406 filas
vistas y 1308 nuevas entre los seis objetivos; no son personas adicionales.

- `g5_2_perfiles/filas.csv`: predicciones individuales, probabilidades, decisiones
  y marca visto/nuevo; permanece ignorado por Git junto al manifiesto local.
- `g5_2_perfiles/manifest.json`: clases originales, orden de entradas, versiones,
  rutas y hashes de datos, fuentes, manifiesto central, splits y referencia.
- [folds.csv](resumen/perfiles/folds.csv): métricas por objetivo/fold/regla.
- [resumen_folds.csv](resumen/perfiles/resumen_folds.csv): media y desviación
  externa `ddof=0`, con cantidad de folds válidos por métrica.
- [oof.csv](resumen/perfiles/oof.csv): métricas sobre todas las filas externas
  de cada objetivo, diferenciadas del promedio de folds.
- [auditoria.json](resumen/perfiles/auditoria.json): cobertura, comparación
  con referencia y hash del manifiesto local de origen; solo agregados.

Para repetir, elegir carpetas de salida nuevas o vacías con `--local` y
`--out`; el ejecutor rechaza carpetas con archivos. `--reference` es opcional
para una reproducción sin el CSV histórico; al especificarlo exige igualdad
exacta antes de escribir salidas. El SAV y los splits deben corresponder a la
corrida central. Las salidas anteriores se conservan.

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
