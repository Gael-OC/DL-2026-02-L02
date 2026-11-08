# 07/11: Análisis de Localidad y Orden de Características (J4.1 y J4.2)

## 1. Introducción y Marco del Experimento
El estudio del panel de localidad tuvo como propósito evaluar empíricamente si la vecindad espacial impuesta por las operaciones convolucionales (`Conv1D` con kernel 3 frente a kernel 1) y el orden original de las 15 características cognitivas del ACE-III aportan una ventaja estable en la predicción del objetivo principal $GDS\_R2$. 

Para ello, se ejecutó una validación masiva de **405 ajustes** combinando:
* **Tres arquitecturas:** `CNN-k3` (17,219 parámetros), `CNN-k1` y `MLP-Capacity` (17,231 parámetros).
* **Tres esquemas de ordenamiento:** `original`, `blocks` (permutación interna por bloques con semilla 2026) y `global` (permutación total con semilla 2027).
* **Tres semillas de partición externa:** 42, 314 y 2718 (5 folds externos cada una).
* **Tres semillas de inicialización:** 7, 17 y 29.

---

## 2. Resultados Consolidados por Arquitectura
A continuación se presentan los valores promedio obtenidos OOF (Out-Of-Fold) para cada arquitectura evaluada:

| Modelo | Parámetros | Accuracy ($\uparrow$) | Balanced Acc ($\uparrow$) | MAE Ordinal ($\downarrow$) | QWK ($\uparrow$) | Errores Graves ($\downarrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CNN-k3** | 17,219 | $0.7213$ | $0.6374$ | $0.2946$ | $0.6874$ | **$0.0159$** |
| **CNN-k1** | *N/D* | $0.7247$ | **$0.6465$** | $0.2926$ | **$0.6929$** | $0.0173$ |
| **MLP-Capacity** | 17,231 | **$0.7302$** | $0.6446$ | **$0.2870$** | $0.6923$ | $0.0172$ |

### Discusión sobre Localidad Espacial (`CNN-k3` vs. `CNN-k1`)
El rendimiento prácticamente idéntico (y ligeramente superior en `CNN-k1`) demuestra que **la vecindad de tamaño 3 no introduce una ganancia informacional sustancial** frente a un kernel de tamaño 1. Esto indica que las 15 entradas binarias del instrumento actúan fundamentalmente como un conjunto de indicadores de estado cognitivo más que como una secuencia temporal estricta con dependencias locales contiguas obligatorias.

---

## 3. Resultados Consolidados por Esquema de Órdenes
El impacto de alterar la disposición espacial de los ítems de entrada se resume en la siguiente tabla:

| Esquema de Orden | Accuracy ($\uparrow$) | Balanced Acc ($\uparrow$) | MAE Ordinal ($\downarrow$) | QWK ($\uparrow$) | Errores Graves ($\downarrow$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Original** (Orden docente) | $0.7247$ | $0.6425$ | $0.2919$ | $0.6911$ | $0.0166$ |
| **Dentro de bloques** (`blocks`) | $0.7252$ | **$0.6462$** | **$0.2908$** | **$0.6927$** | **$0.0160$** |
| **Global** (Permutación total) | **$0.7262$** | $0.6438$ | $0.2915$ | $0.6888$ | $0.0177$ |

### Impacto de la Permutación
Los resultados evidencian una **extraordinaria estabilidad métrica** ante la alteración drástica del orden de las preguntas ($QWK$ oscilando estrechamente entre $0.688$ y $0.693$). Esto confirma que las redes convolucionales con *Flatten* y las redes densas aprenden representaciones basadas en la presencia global y sinergias distribuidas de los ítems cognitivos, sin depender de la topología posicional original del cuestionario.

---

## 4. Conclusiones del Panel de Localidad
1. **Ausencia de ventaja topológica estricta:** La cercanía física entre las preguntas en el test clínico no condiciona el éxito predictivo de las arquitecturas convolucionales.
2. **Capacidad vs. Convolución:** La MLP de capacidad próxima iguala o supera levemente a las CNNs, sugiriendo que la no linealidad de capas densas profundas maneja adecuadamente la interacción de los 15 atributos binarios.