# Resultados y reproducción

`resumen/` contiene tablas, figuras y auditorías agregadas. Las corridas
completas, el SAV, splits, logits y predicciones individuales se conservan
localmente y están excluidos de Git. El [mapa de evidencia](../docs/resultados.md)
relaciona estos resultados con los requisitos del laboratorio.

Todos los comandos se ejecutan desde la raíz con `lab_pytorch` activo y
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.

## Comparación central

- `g4_2_central_unweighted/`: Softmax y CORAL; 654 ajustes.
- `g4_2_central_weighted/`: CORAL ponderada; 327 ajustes.
- [Seis tablas](resumen/central/comparacion.md), nueve métricas y NLL/RPS.
- [Configuraciones elegidas](resumen/central/configuraciones.csv) y confusiones
  OOF de todos los folds, con clases originales identificadas.

```bash
python -m experimentos.auditar_corrida results/g4_2_central_unweighted --eventos 654 --data-path "dataset/15 atributos R0-R5.sav"
python -m experimentos.auditar_corrida results/g4_2_central_weighted --eventos 327 --data-path "dataset/15 atributos R0-R5.sav"
python -m experimentos.comparacion results/g4_2_central_unweighted results/g4_2_central_weighted --output results/regeneracion/central
```

La auditoría es de solo lectura y valida el hash del SAV, incluidos traslados
con `--data-path`. La comparación escribe agregados sin entrenar. El resumen
principal usa media y desviación externa (`ddof=0`), separadas del agregado de
filas OOF. GDS usa dos externos y configuración base sin búsqueda interna;
las cinco recodificaciones usan 5×3 y cuatro candidatas.

Para reconstruir las corridas desde cero, usar los dos comandos de
[entrenamiento central](../README.md#reproducir-la-comparación-obligatoria).
`python -m experimentos.reportes CARPETA` regenera las exportaciones de una
corrida conservada. Las confusiones históricas rotuladas como último fold no
representan todos los folds; para interpretar resultados usar las OOF agregadas.

## Controles R2 y constantes

Los [controles corregidos](resumen/controles/comparacion.md) reutilizan los
splits centrales. Los cuatro modelos tienen cuatro candidatas evaluadas en
las tres validaciones internas y un refit externo completo: 260 ajustes.
La MLP utiliza 20 épocas. Los [resúmenes completos](resumen/controles/resumen.json)
incluyen las nueve métricas y NLL/RPS, por folds y OOF por separado.

```bash
# Auditar registros y reconstruir agregados; no entrena.
python -m experimentos.ejecutar_controles --report-only --output-dir results/controles_validacion --summary-dir results/regeneracion/controles

# Reconstruir los 260 ajustes si no se conservan sus artefactos.
python -m experimentos.ejecutar_controles --data-path "dataset/15 atributos R0-R5.sav" --output-dir results/nuevos_controles --summary-dir results/nuevos_controles_resumen

# Moda y mediana constantes: solo cuentan etiquetas de train externo.
python -m experimentos.constantes --out results/regeneracion/constantes
```

Las [constantes de seis objetivos](resumen/constantes/resumen_folds.csv) usan
los índices 0..K−1 y no llevan suavizado. El desempate elige la clase menor.
Sus tablas cubren 27 folds; [OOF](resumen/constantes/oof.csv) se informa aparte.
El ejecutor y la tabla histórica de `results/controles/` no son la referencia
para la comparación final; no mezclar esas cifras con los controles corregidos.

## Decisiones por moda y mediana

```bash
python -m experimentos.decisiones --local results/regeneracion/decisiones_local --out results/regeneracion/decisiones
```

Reutiliza las probabilidades centrales, sin entrenamientos. Exporta métricas
por fold, media/std, OOF y diferencias pareadas en `resumen/decisiones/` cuando
se usa la salida predeterminada. `--unweighted` y `--weighted` permiten cambiar
las rutas de las corridas fuente. Las filas individuales quedan en `--local`.

## Predictor por perfil

El predictor cuenta etiquetas de las 15 respuestas exactas en train externo:
`p(c|perfil) = (n_c + 1) / (n_perfil + K)`. Para un perfil nuevo usa los conteos
globales de ese entrenamiento con el mismo suavizado. No elimina respuestas
repetidas ni usa el ID como entrada.

```bash
python -m experimentos.perfiles --data-path "dataset/15 atributos R0-R5.sav" --splits-dir results/g4_2_central_unweighted --local results/regeneracion/perfiles_local --out results/regeneracion/perfiles
```

Usar carpetas nuevas o vacías. La salida individual incluye filas y manifiesto;
la agregada incluye [métricas por fold](resumen/perfiles/folds.csv),
[media/std externa](resumen/perfiles/resumen_folds.csv),
[OOF](resumen/perfiles/oof.csv) y [auditoría](resumen/perfiles/auditoria.json).
Son 6714 predicciones, 27 folds y seis objetivos.

Para cotejar con el CSV histórico conservado se puede añadir
`--reference results/g5_perfiles/filas.csv`. Se exige coincidencia exacta del
modelo `perfil`; este archivo es opcional y no viene en Git.

## Análisis de decisiones y perfiles

```bash
python -m experimentos.analisis --decisions-path results/regeneracion/decisiones_local/filas.csv --profiles results/regeneracion/perfiles_local --out results/regeneracion/analisis
```

Valida 20142 predicciones de redes y 6714 de perfiles contra los folds y
manifiestos fuente, conservando el dtype al comprobar moda/mediana.
`--unweighted` y `--weighted` permiten apuntar a otras corridas compatibles.
No entrena y exige una salida nueva o vacía.

- [Discusión](resumen/analisis/discusion.md): comparaciones por pares y límites.
- [Decisiones](resumen/analisis/decisiones.png): diferencias mediana−moda por fold.
- [Perfiles](resumen/analisis/perfiles.png): MAE OOF con mediana prefijada y soporte.
- [Recall por clase](resumen/analisis/recall_clases.csv): nivel fold/OOF,
  clase original y soporte; sin soporte se declara no estimable.
- [Auditoría](resumen/analisis/auditoria.json): cobertura y hashes de origen.

Error firmado significa predicción−verdad; negativo indica subestimación.
Visto/nuevo indica presencia del perfil en train, no identidad de la persona.
Las diferencias son descriptivas y no permiten elegir retrospectivamente una
regla usando el test.

## Localidad y orden

El [panel completo](resumen/localidad/completo/interpretacion.md) tiene
405 ajustes: tres modelos × tres órdenes × tres particiones × tres
inicializaciones × cinco folds. Cada una de las 81 condiciones conserva
1119 predicciones OOF. No utiliza búsqueda interna.

```bash
# Verificar y regenerar las métricas/figura desde los 405 folds conservados.
python -m experimentos.ejecutar_localidad --report-only --data-path "dataset/15 atributos R0-R5.sav"

# En un equipo sin artefactos: prueba de nueve condiciones y panel completo.
python -m experimentos.ejecutar_localidad --smoke --data-path "dataset/15 atributos R0-R5.sav"
python -m experimentos.ejecutar_localidad --data-path "dataset/15 atributos R0-R5.sav"
```

Las salidas individuales están en `localidad/preflight/` y `localidad/completo/`;
los agregados, en `resumen/localidad/` bajo los mismos nombres. No publicar
las salidas de preflight como resultados finales. El manifiesto coteja código,
datos, órdenes y protocolo antes de reutilizar un fold.

La [comparación pareada](resumen/localidad/completo/comparacion_pareada.csv)
conserva semillas y orden. [OOF por condición](resumen/localidad/completo/oof_condiciones.csv)
y [media/std de sus cinco folds](resumen/localidad/completo/resumen_folds.csv)
son estadísticos diferentes. La figura muestra ambas fuentes de variación.

## Robustez entre particiones

[Resultados](resumen/robustez/robustez.md) de R1/R2 en particiones 42, 314 y
2718, con inicialización base 42. Los 780 ajustes adicionales están en las
ocho carpetas locales `h5_robustez/`. Para reconstruirlas sin artefactos:

```bash
for target in GDS_R1 GDS_R2; do
  for split in 314 2718; do
    python main.py --data-path "dataset/15 atributos R0-R5.sav" --target-name "$target" --algorithm all --split-seed "$split" --init-seed 42 --output-dir "results/h5_robustez/${target}_split${split}_unweighted"
    python main.py --data-path "dataset/15 atributos R0-R5.sav" --target-name "$target" --algorithm conv1d_coral --use-weights --split-seed "$split" --init-seed 42 --output-dir "results/h5_robustez/${target}_split${split}_weighted"
  done
done

# Audita las diez carpetas y regenera resúmenes sin reentrenar.
python -m experimentos.robustez --out results/regeneracion/robustez --data-path "dataset/15 atributos R0-R5.sav"
```

Los folds y particiones comparten observaciones: no son réplicas independientes
ni se selecciona una semilla ganadora. Las frecuencias del grid y las diferencias
OOF se conservan separadas de la dispersión entre folds.
