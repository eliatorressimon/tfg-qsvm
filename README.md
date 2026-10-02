# SVM con kernels cuánticos (QSVM)

Trabajo de Fin de Grado del Grado en Matemáticas, Universidad Complutense de Madrid.

**Autora:** Elia Torres Simón
**Tutor:** Luis Fernando Llana Díaz

## Idea

Una SVM solo necesita productos escalares entre los datos (el *kernel*). Un ordenador cuántico calcula de forma natural productos internos entre estados de un espacio de Hilbert de dimensión 2^n. El método de kernel cuántico (Havlíček et al., 2019) consiste en:

1. **Codificar** cada dato `x` en un estado cuántico `|φ(x)⟩ = U(x)|0…0⟩` mediante un circuito.
2. **Estimar el kernel** `K(x, y) = |⟨φ(x)|φ(y)⟩|²` con un circuito (`U(y)`, después `U(x)†`, y medir).
3. **Resolver la SVM dual** con esa matriz de kernel. La optimización sigue siendo clásica y convexa.

El objetivo es implementarlo de principio a fin, compararlo con kernels clásicos (lineal, RBF) y estudiar sus limitaciones: el error por número finito de disparos y la concentración exponencial.

## Estructura del repositorio

| Carpeta | Contenido |
|---|---|
| `cuaderno-tfg/` | Cuaderno de investigación: teoría de SVM, kernels, hoja de ruta del trabajo |
| `cuaderno-cuantica/` | Apuntes de computación cuántica (curso de IBM, lecciones 1–3), en LaTeX y PDF, con sus figuras |
| `notebooks-qiskit/` | Experimentos en Qiskit de cada lección |
| `svm/` | Implementaciones propias de la SVM (primal y dual) |
| `reuniones/` | Material de las reuniones con el tutor |

## Estado

- [x] Teoría de SVM: primal, dual, KKT, kernels, margen suave
- [x] SVM primal (descenso de subgradiente) y dual de margen duro (`scipy`)
- [x] Fundamentos de computación cuántica: un sistema, varios sistemas, circuitos
- [x] Primeros experimentos en Qiskit, incluida la estimación de un kernel sencillo
- [ ] Dual de margen suave con kernel genérico
- [ ] Kernel cuántico con `ZZFeatureMap` y QSVM completa
- [ ] Experimentos comparativos y limitaciones

## Cómo ejecutar los notebooks

```bash
pip install -r requirements.txt
jupyter notebook
```

Probado con Qiskit 2.2.3.

## Referencias principales

- V. Havlíček et al., *Supervised learning with quantum-enhanced feature spaces*, Nature 567, 209–212 (2019).
- M. Schuld y N. Killoran, *Quantum machine learning in feature Hilbert spaces*, Phys. Rev. Lett. 122, 040504 (2019).
- Y. Liu, S. Arunachalam y K. Temme, *A rigorous and robust quantum speed-up in supervised machine learning*, Nature Physics 17, 1013–1017 (2021).
- S. Thanasilp, S. Wang, M. Cerezo y Z. Holmes, *Exponential concentration in quantum kernel methods*, Nature Communications 15, 5200 (2024).
- J. Watrous, *Understanding Quantum Information and Computation*, IBM Quantum Learning.
