# Laboratorio 02: comparación categórica y ordinal con Conv1D

**Borrador de las secciones de Gael — 06/11/2026.** Equipo: Gael y Johan.
Sigue las secciones de la plantilla docente. Faltan la integración, el ajuste
de páginas, las referencias finales, la autoevaluación y la exportación a PDF.

## Resumen provisional

Se compararon Softmax, CORAL y CORAL ponderada en seis codificaciones de GDS
sobre 1119 registros con 15 entradas binarias. La corrida central incluye
981 ajustes y una predicción externa por registro y variante. Softmax obtuvo
el menor MAE externo medio en los seis objetivos con el protocolo fijado.
Esto describe estos datos y condiciones; no establece superioridad general ni
utilidad clínica. Las extensiones de decisiones, perfiles y robustez se
presentan por separado para distinguir sus efectos.

## Introducción

El problema consiste en predecir una categoría ordenada de deterioro cognitivo
a partir de respuestas binarias. La comparación estudia si incorporar el orden
en la pérdida mejora los errores de distancia frente a una salida categórica.
Johan completará la introducción del equipo y el contexto de los controles y
del estudio de localidad. Herramientas: Python, NumPy, pandas, PyTorch,
scikit-learn, pyreadstat y Matplotlib; versiones exactas en los manifiestos.

## Marco teórico

La red interpreta cada fila como una secuencia de 15 posiciones. Dos bloques
Conv1D con BatchNorm y ReLU extraen respuestas locales, y una capa densa reúne
sus activaciones. Softmax aprende logits categóricos y minimiza entropía
cruzada. CORAL representa la etiqueta mediante K−1 indicadores acumulativos
`y > k`, optimizados con BCE con logits, sumada sobre umbrales y promediada
sobre muestras. La variante ponderada multiplica cada pérdida por el peso de
su clase real, calculado por número efectivo con beta 0.99 y normalizado a
media uno sobre el alfabeto.

La cabeza CORAL usa un puntaje lineal compartido y sesgos decrecientes. Los
saltos se parametrizan con softplus y suma acumulada: los logits y sus
sigmoides conservan el orden, y las diferencias sucesivas definen probabilidades
de clase no negativas. La decisión original cuenta umbrales que superan 0.5.
Una regla de decisión adicional puede cambiar las clases predichas sin cambiar
el modelo ni su distribución aprendida.

## Metodología

El archivo docente contiene 1119 filas, ID únicos, 15 predictores binarios y
GDS junto con cinco recodificaciones. No se observaron faltantes. El ID y las
otras etiquetas quedan fuera de las entradas. Hay 321 perfiles distintos;
97 presentan etiquetas GDS diferentes y reúnen 851 filas. Esta ambigüedad es
descriptiva del archivo, no un límite poblacional. Un ID único tampoco prueba
que cada fila corresponda a una persona distinta. La procedencia exacta del
instrumento y el significado de algunos nombres no están confirmados.

Se mantuvo el protocolo experimental fijado antes de evaluar:
Adam, 20 épocas, batch 32, learning rate 0.001 y weight decay 0.0001. Las
cinco recodificaciones usan validación anidada de cinco folds externos y tres
internos, con cuatro candidatas de arquitectura. La selección minimiza MAE
interno y desempata por QWK; después se reentrena con todo el entrenamiento
externo. GDS solo tiene dos casos de etapa 7: usa dos folds externos,
configuración base y **búsqueda interna omitida**, no validación anidada 2×2.
Las semillas centrales de partición e inicialización son 42, con las
derivaciones por ajuste registradas; las tres variantes comparten particiones.

Se informan accuracy, balanced accuracy, precision/recall/F1 macro, MAE, QWK,
accuracy ±1 y proporción de errores de distancia al menos dos. Las tablas
principales muestran media ± desviación externa con `ddof=0`; esa desviación
no es un intervalo de confianza. El agregado de todas las predicciones OOF es
un estadístico distinto. NLL protege solo el logaritmo con el menor positivo
del dtype, y RPS se normaliza por K−1. En R3 el alfabeto original 1/3 se
transforma a 0/1, por lo que MAE=1−accuracy y los errores graves son cero.

El análisis adicional aplica moda y mediana inferior a las mismas probabilidades
externas, sin nuevos entrenamientos ni selección de reglas con test. Conserva
la decisión original de cada red como referencia; las redes no se volvieron a
seleccionar para cada regla. El predictor por perfil cuenta etiquetas de las
15 respuestas exactas solo en entrenamiento externo, con suavizado aditivo uno
por clase. Para perfiles nuevos usa las frecuencias globales de entrenamiento
con el mismo suavizado. No elimina respuestas repetidas ni utiliza el ID.

Las 20142 predicciones de las redes se cruzan con 6714 del predictor por perfil
por objetivo, índice global y fold, comprobando etiquetas y probabilidades
contra sus artefactos de origen. Visto/nuevo describe la presencia del perfil
en entrenamiento, no la identidad de la persona. Se conserva soporte por clase
y grupo; recalls sin soporte se declaran no estimables. El error firmado es
predicción−verdad: negativo indica subestimación. La figura de perfiles usa
mediana como regla común prefijada; las tablas conservan también las otras reglas.

## Resultados de la comparación obligatoria

Las [seis tablas completas](../results/resumen/central/comparacion.md) incluyen las
tres variantes y nueve métricas, con NLL/RPS en tablas complementarias. Los
CSV conservan precisión completa; las confusiones corresponden a todos los
folds externos. Las [frecuencias de configuración](../results/resumen/central/configuraciones.csv)
permiten comprobar la variación de la selección, sin elegir retrospectivamente
una configuración por el rendimiento de test.

En R2, el MAE externo fue 0.2896 ± 0.0389 con Softmax, 0.2932 ± 0.0250 con
CORAL y 0.2976 ± 0.0325 con CORAL ponderada. Los errores graves fueron
0.0161 ± 0.0067, 0.0259 ± 0.0066 y 0.0241 ± 0.0045, respectivamente. Las
diferencias pequeñas requieren el análisis de estabilidad previsto. En GDS,
los MAE fueron 0.4737, 0.6890 y 0.7891; el soporte mínimo de la etapa 7 y los
dos folds limitan la interpretación. En R3, los MAE fueron 0.0920, 0.0992 y
0.0992; ninguna variante puede cometer errores de distancia dos en ese alfabeto.

## Resultados de decisiones y perfiles

Las [tablas y discusión completas](../results/resumen/analisis/discusion.md)
se reconstruyen desde los artefactos locales, sin entrenamiento. La
[figura de decisiones](../results/resumen/analisis/decisiones.png) muestra
diferencias mediana−moda por fold y su media; no representa intervalos de confianza.
En Softmax/R2, el cambio medio externo de MAE es −0.0062 ± 0.0073, con 33
decisiones OOF distintas. El MAE OOF pasa de 0.2895 a 0.2833; estos agregados
de filas son distintos de los promedios externos. La tasa OOF de subestimación
pasa de 18.1 % a 17.3 %, y la de sobreestimación, de 9.2 % a 9.5 %.

La mediana no mejora uniformemente: Softmax aumenta ligeramente el MAE en
GDS y R4, y el predictor por perfil aumenta en GDS y R1. En R3 no cambia ninguna
decisión de los cuatro modelos. En CORAL, la mediana coincide con la decisión
original en todas estas predicciones; su ventaja frente a la moda de las
probabilidades de clase no es una mejora respecto de la comparación obligatoria.
NLL y RPS se mantienen al cambiar solo la regla, porque las probabilidades no cambian.

En R2 hay 900 filas con perfiles vistos en entrenamiento y 219 con nuevos.
La [figura de perfiles](../results/resumen/analisis/perfiles.png) muestra MAE
OOF con mediana común en los seis objetivos; el siguiente detalle es R2:

| Modelo | MAE vistos | MAE nuevos | Errores graves vistos | Errores graves nuevos |
|---|---:|---:|---:|---:|
| Softmax | 0.2589 | 0.3836 | 0.0133 | 0.0228 |
| CORAL | 0.2644 | 0.4110 | 0.0200 | 0.0502 |
| CORAL ponderada | 0.2633 | 0.4384 | 0.0189 | 0.0457 |
| Predictor por perfil | 0.2744 | 0.8630 | 0.0078 | 0.2603 |

En los perfiles nuevos de R2, el respaldo global predice siempre la primera
clase. Los soportes de clases 1/2/3 son 87/75/57, con recalls 1/0/0 para ese
predictor; Softmax obtiene 0.747/0.493/0.667. En perfiles vistos, los soportes
son 562/223/115 y el recall de la segunda clase es 0.610 con el predictor por
perfil frente a 0.511 con Softmax. Por tanto, el MAE agregado no describe por sí
solo el comportamiento de cada clase.

La novedad no implica necesariamente mayor error: en R4 los cuatro modelos
tienen menor MAE en perfiles nuevos. La composición de clases y los soportes
difieren; no se atribuye causalidad a visto/nuevo. Comparar moda y mediana
mantiene el aprendizaje, mientras que comparar el predictor por perfil con
las redes cambia el estimador. La ambigüedad empírica del archivo completo no
estima un límite poblacional ni una cota de los modelos OOF.

## Discusión y respuestas mínimas

1. **Kernel 3:** puede combinar tres respuestas contiguas; dos capas alcanzan
   un campo local de cinco posiciones antes de la capa densa. La cercanía
   depende del orden docente de las columnas y no demuestra vecindad clínica.
2. **Permutaciones:** cambiar columnas durante inferencia altera las entradas
   esperadas. Reentrenar con el mismo orden en train/test estudia otra
   representación; una MLP también cambia su optimización con la permutación.
   La conclusión empírica sobre localidad queda pendiente del panel de Johan.
3. **Flatten:** conserva la posición de cada activación, mientras que un
   promedio global pierde esa información. La elección aumenta los parámetros
   de la capa densa y limita una interpretación puramente local del modelo.
4. **Parámetros:** la CNN Softmax base con tres clases tiene 17219; la capa
   480→32 concentra 15392 (89.4 %). Las candidatas del grid cambian estos
   valores; no atribuir el total base a todas las redes seleccionadas.
5. **Selección:** en Softmax/R2 la candidata de canales 8/16 fue elegida en
   tres de cinco folds; las otras dos selecciones fueron distintas. Las
   frecuencias completas están exportadas, incluidas las de CORAL y ponderada.
6. **Consistencia ordinal:** puntaje compartido y sesgos ordenados garantizan
   acumulativas decrecientes. Esto no garantiza menor MAE ni calibración.
7. **Efecto de CORAL:** en la referencia central no redujo MAE frente a Softmax
   en ninguno de los seis objetivos. Tampoco redujo los errores graves: hubo
   igualdad en objetivos donde ambas tasas fueron cero. Los resultados
   desfavorables forman parte de la comparación prevista.

No se consideran independientes las seis etiquetas del mismo archivo ni los
folds externos con entrenamientos solapados. No se aplica un t-test sobre
folds. La ponderación cambia el objetivo de entrenamiento; su efecto sobre
recall de clases minoritarias debe leerse junto con MAE, soporte y errores.

La comparación final con controles R2 requiere revisión de su ejecución:
el script existente utiliza el primer fold interno, no hace el refit con todo
el entrenamiento externo y fija diez épocas para la MLP. Además, las
constantes deben recibir índices de clase, no etiquetas originales. Hasta
resolverlo, sus números históricos no constituyen una comparación bajo el
protocolo congelado. Los módulos y resultados de Johan se conservan.

## Conclusiones provisionales

La implementación ordinal mantiene acumulativas consistentes y permite
comparar entrenamiento y decisión bajo las mismas particiones. En la corrida
central, Softmax obtuvo menores MAE, sin que la estructura ordinal de CORAL
produjera una mejora automática. Las conclusiones del equipo requieren integrar
controles, localidad y las extensiones previstas, además de reconocer la
incertidumbre del instrumento y el escaso soporte de clases extremas.

El análisis de decisiones muestra cambios pequeños y no uniformes en Softmax,
y que CORAL ya usa una decisión coincidente con la mediana en la corrida
central. El predictor por perfil describe respuestas vistas pero depende de
un respaldo global en las nuevas; los recalls y soportes hacen visible ese
límite. Estos resultados distinguen el efecto de decidir del de aprender y no
permiten elegir retrospectivamente una regla o arquitectura para futuros datos.

## Código y reproducibilidad

El núcleo de entrenamiento no se cambió al auditar la corrida central. Se
conservan hashes de código/datos, índices globales, ajustes, pérdidas,
candidatas, selección y probabilidades externas. La auditoría recalcula
soportes, pesos y métricas y detecta artefactos incompatibles. El test externo
no decide épocas, hiperparámetros ni ampliaciones del estudio.

```bash
python -m experimentos.auditar_corrida results/g4_2_central_unweighted --eventos 654
python -m experimentos.auditar_corrida results/g4_2_central_weighted --eventos 327
python -m experimentos.comparacion results/g4_2_central_unweighted results/g4_2_central_weighted --output results/resumen/central
python -m experimentos.analisis --out results/resumen/analisis_reproduccion
python -m unittest discover -s tests -v
```

Activar `lab_pytorch` y limitar hilos a uno por proceso. Los artefactos por
fila y los datos permanecen locales e ignorados; únicamente se versionan
agregados seleccionados. Consultar [uso](03_uso.md) para la ejecución completa.

## Referencias para la integración final

- Enunciado y plantilla docente del Laboratorio 02.
- Cao, Mirjalili y Raschka, [CORAL](https://arxiv.org/abs/1901.07884).
- Cui et al., [Class-Balanced Loss Based on Effective Number of Samples](https://arxiv.org/abs/1901.05555).
- Cawley y Talbot, [On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation](https://www.jmlr.org/papers/v11/cawley10a.html).

La bibliografía, los límites de páginas y la rúbrica se revisarán al integrar
el informe conjunto; este borrador no constituye el PDF final de entrega.
