# Datos

## `billetes.csv`

*Banknote Authentication* (UCI Machine Learning Repository), 1372 billetes y 5 columnas, sin cabecera:

| Columna | Variable | Descripción |
|---|---|---|
| 0 | varianza | Varianza de la imagen transformada por wavelet |
| 1 | asimetría | Asimetría (*skewness*) de la imagen transformada |
| 2 | curtosis | Curtosis de la imagen transformada |
| 3 | entropía | Entropía de la imagen |
| 4 | clase | 0 = falso, 1 = auténtico |

**Fuente:** V. Lohweg, *Banknote Authentication*, UCI Machine Learning Repository (2013). <https://doi.org/10.24432/C55P57>

**Licencia:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). El archivo se redistribuye sin modificaciones.

El conjunto Iris que usa `svm/svm_dual.py` se carga directamente desde `sklearn.datasets`.
