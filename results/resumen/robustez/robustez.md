# Robustez entre particiones

Tres particiones externas (42, 314 y 2718), inicialización 42 y búsqueda anidada completa. Métricas OOF por partición; los folds y particiones se solapan y no son muestras independientes.

| Objetivo | Partición | Variante | MAE OOF | QWK OOF | NLL OOF | RPS OOF |
|---|---:|---|---:|---:|---:|---:|
| GDS_R1 | 42 | coral | 0.1108 | 0.6372 | 0.3640 | 0.0465 |
| GDS_R1 | 42 | softmax | 0.1046 | 0.6890 | 0.2831 | 0.0407 |
| GDS_R1 | 42 | coral_weighted | 0.1305 | 0.6363 | 0.3729 | 0.0491 |
| GDS_R1 | 314 | coral | 0.1162 | 0.6303 | 0.3594 | 0.0457 |
| GDS_R1 | 314 | softmax | 0.1055 | 0.6826 | 0.2904 | 0.0403 |
| GDS_R1 | 314 | coral_weighted | 0.1233 | 0.6658 | 0.3655 | 0.0471 |
| GDS_R1 | 2718 | coral | 0.1153 | 0.6594 | 0.3728 | 0.0471 |
| GDS_R1 | 2718 | softmax | 0.1081 | 0.6779 | 0.2905 | 0.0411 |
| GDS_R1 | 2718 | coral_weighted | 0.1251 | 0.6621 | 0.3794 | 0.0486 |
| GDS_R2 | 42 | coral | 0.2931 | 0.6890 | 0.7575 | 0.1129 |
| GDS_R2 | 42 | softmax | 0.2895 | 0.6950 | 0.6568 | 0.1049 |
| GDS_R2 | 42 | coral_weighted | 0.2976 | 0.6931 | 0.7516 | 0.1120 |
| GDS_R2 | 314 | coral | 0.3083 | 0.6822 | 0.7568 | 0.1134 |
| GDS_R2 | 314 | softmax | 0.3110 | 0.6692 | 0.6933 | 0.1096 |
| GDS_R2 | 314 | coral_weighted | 0.3003 | 0.6933 | 0.7624 | 0.1147 |
| GDS_R2 | 2718 | coral | 0.3021 | 0.6856 | 0.7637 | 0.1144 |
| GDS_R2 | 2718 | softmax | 0.2976 | 0.6789 | 0.6630 | 0.1068 |
| GDS_R2 | 2718 | coral_weighted | 0.3092 | 0.6851 | 0.7641 | 0.1155 |

Las diferencias pareadas por partición están en `diferencias_oof.csv`; valores negativos de MAE, NLL y RPS favorecen la variante frente a Softmax. Las frecuencias por candidata y configuración están en `selecciones.csv`. La dispersión de folds es descriptiva y se informa aparte en `metricas.csv`. No se selecciona una partición ganadora.

En R1, Softmax tiene menor MAE y mayor QWK que ambas variantes CORAL en las tres particiones. En R2, la comparación de MAE cambia de signo en 314; NLL y RPS favorecen a Softmax en ambas tareas y las tres particiones. Las candidatas seleccionadas varían entre folds y particiones; no hay una candidata elegida en los cinco folds de ninguna condición. Estos resultados describen sensibilidad a la partición, no incertidumbre poblacional ni una prueba de superioridad.
