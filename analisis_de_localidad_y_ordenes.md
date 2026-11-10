# Localidad y orden de características en GDS_R2

## Protocolo y fuente

Panel fijo de tres modelos, tres órdenes, tres particiones (42/314/2718), tres
inicializaciones (7/17/29) y cinco folds: 405 ajustes y 81 condiciones OOF.
Cada orden se aplica tanto al entrenamiento como a evaluación; los modelos se
entrenan desde cero, sin grid interno. Se usan 20 épocas, Adam, batch 32,
learning rate 0.001 y weight decay 0.0001.

Fuente: [métricas por condición](results/resumen/localidad/completo/oof_condiciones.csv).
Las tablas siguientes promedian métricas OOF de condiciones; no son medias
de folds, intervalos de confianza ni comparaciones de observaciones independientes.

## Resumen por arquitectura

Promedio sobre 27 condiciones por arquitectura: tres órdenes × nueve pares de semillas.

| Modelo | Parámetros | Accuracy | Balanced accuracy | MAE | QWK | Errores graves |
|---|---:|---:|---:|---:|---:|---:|
| cnn_k3 | 17219 | 0.7241 | 0.6390 | 0.2936 | 0.6865 | 0.0177 |
| cnn_k1 | 16163 | 0.7308 | 0.6465 | 0.2891 | 0.6899 | 0.0199 |
| mlp_capacity | 17231 | 0.7317 | 0.6511 | 0.2851 | 0.6977 | 0.0167 |

## Resumen por orden

Promedio sobre 27 condiciones por orden: tres modelos × nueve pares de semillas.

| Orden | Accuracy | Balanced accuracy | MAE | QWK | Errores graves |
|---|---:|---:|---:|---:|---:|
| original | 0.7286 | 0.6452 | 0.2897 | 0.6902 | 0.0184 |
| blocks | 0.7285 | 0.6450 | 0.2892 | 0.6927 | 0.0176 |
| global | 0.7295 | 0.6464 | 0.2888 | 0.6912 | 0.0184 |

## Interpretación y límites

Las diferencias agregadas son pequeñas. La [comparación pareada](results/resumen/localidad/completo/comparacion_pareada.csv)
muestra cambios de signo entre condiciones: no permite establecer un ganador
general ni una ventaja universal del orden original o del kernel 3.

La MLP tiene capacidad próxima a CNN-k3, pero no se igualan todas las
propiedades de optimización y regularización. Flatten permite combinar las
posiciones en la capa densa; los resultados no demuestran que el orden sea
irrelevante ni aíslan un efecto causal de la vecindad. La procedencia exacta
del instrumento no está confirmada.

La [interpretación detallada](results/resumen/localidad/completo/interpretacion.md)
y la [figura por semillas](results/resumen/localidad/completo/mae_localidad.png)
mantienen las condiciones separadas. Para regenerar, ver la [guía](results/README.md#localidad-y-orden).
