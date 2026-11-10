# Controles en GDS_R2

Mismos datos y particiones 5×3 de la corrida central sin ponderación; selección por MAE interno y desempate QWK. Los controles se reentrenan con todo el entrenamiento externo. Media ± desviación de cinco folds (ddof=0); OOF se calcula por separado.

| Modelo | MAE media ± DE | QWK media ± DE | MAE OOF | QWK OOF |
|---|---:|---:|---:|---:|
| logistica_suma | 0.2816 ± 0.0381 | 0.7023 ± 0.0431 | 0.2815 | 0.7029 |
| logistica_15 | 0.2824 ± 0.0395 | 0.6959 ± 0.0521 | 0.2824 | 0.6968 |
| mlp_sencilla | 0.2878 ± 0.0405 | 0.6926 ± 0.0503 | 0.2878 | 0.6935 |
| hist_gradient_boosting | 0.2842 ± 0.0291 | 0.6973 ± 0.0409 | 0.2842 | 0.6978 |
| constante_moda | 0.5737 ± 0.0039 | 0.0000 ± 0.0000 | 0.5737 | 0.0000 |
| constante_mediana | 0.5737 ± 0.0039 | 0.0000 ± 0.0000 | 0.5737 | 0.0000 |
| softmax | 0.2896 ± 0.0389 | 0.6940 ± 0.0526 | 0.2895 | 0.6950 |
| coral | 0.2932 ± 0.0250 | 0.6879 ± 0.0402 | 0.2931 | 0.6890 |

La comparación es descriptiva para esta partición; los modelos y sus configuraciones difieren. NLL, RPS y las demás métricas están en el resumen JSON.
