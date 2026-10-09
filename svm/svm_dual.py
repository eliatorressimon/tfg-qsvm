"""
SVM lineal de margen duro resuelta por la vía dual.

Datos: Iris (Setosa frente a Versicolor), con el largo del sépalo y el largo del
pétalo. El dual se resuelve con SLSQP de scipy, imponiendo las dos restricciones
del problema: alpha_i >= 0 y sum_i alpha_i y_i = 0. Teoría en la sección 2.2 de
apuntes/cuaderno_tfg/cuaderno_tfg.tex.

Uso:  python svm/svm_dual.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.optimize import minimize

carpeta_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ruta_figura = os.path.join(carpeta_repo, 'apuntes', 'cuaderno_tfg', 'diagrama2.png')

# 1. CARGA DE DATOS (Iris Setosa vs Versicolor)
iris = load_iris()
X = iris.data[:100, [0, 2]]   # Setosa y Versicolor; largo de sepalo y de petalo
y = iris.target[:100]
y = np.where(y == 0, -1, 1).astype(float)

n_samples, n_features = X.shape

# 2. MATRIZ DE GRAM E INICIALIZACIÓN
# Matriz de Gram y matriz Q del dual
K = X @ X.T
Q = np.outer(y, y) * K

# Punto inicial (factible)
alphas0 = np.zeros(n_samples)

# 3. RESOLUCIÓN DEL DUAL CON RESTRICCIONES
# min 1/2 a^T Q a - 1^T a   s.a.  y^T a = 0,  a >= 0
def objetivo(a):
    return 0.5 * a @ Q @ a - a.sum()

def gradiente(a):
    return Q @ a - np.ones(n_samples)

restricciones = [{'type': 'eq',
                  'fun': lambda a: a @ y,    # sum_i alpha_i y_i = 0
                  'jac': lambda a: y}]
cotas = [(0, None)] * n_samples              # alpha_i >= 0

res = minimize(objetivo, alphas0, jac=gradiente,
               bounds=cotas, constraints=restricciones,
               method='SLSQP',
               options={'maxiter': 1000, 'ftol': 1e-12})
alphas = res.x

# 4. RECUPERACIÓN DE w Y b
# Estacionariedad: w = sum_i alpha_i y_i x_i
w = (alphas * y) @ X

# Vectores de soporte (alpha_i > 0, con tolerancia numerica)
idx = np.where(alphas > 1e-6)[0]

# Holgura complementaria: b = y_s - w^T x_s, promediado
b = np.mean(y[idx] - X[idx] @ w)

# Comprobaciones de las condiciones KKT
print("Optimizacion correcta:", res.success)
print("sum alpha_i y_i =", alphas @ y)                 # ~ 0
print("min y_i f(x_i) =", np.min(y * (X @ w + b)))    # ~ 1
print(f"Vectores de soporte: {len(idx)}")
print(f"b = {b:.4f}, w = {w}")

# 5. GRÁFICO
plt.figure(figsize=(9, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolors='k', s=50, label="Datos")

# Definimos el rango del eje X basándonos en los datos reales
x_plot = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 100)

# Dibujamos las líneas solo si w[1] no es cero
if abs(w[1]) > 1e-9:
    # Despejamos y: y = -(w0*x + b) / w1
    y_plot = -(w[0] * x_plot + b) / w[1]
    y_margen_pos = -(w[0] * x_plot + b - 1) / w[1]
    y_margen_neg = -(w[0] * x_plot + b + 1) / w[1]

    plt.plot(x_plot, y_plot, 'k-', linewidth=2, label='Hiperplano Separador')
    plt.plot(x_plot, y_margen_pos, 'k--', alpha=0.4, label='Margen (+1)')
    plt.plot(x_plot, y_margen_neg, 'k--', alpha=0.4, label='Margen (-1)')

    # Resaltamos los Vectores de Soporte con un círculo azul
    plt.scatter(X[idx][:, 0], X[idx][:, 1], s=150, facecolors='none', edgecolors='blue', label='Vectores de Soporte')

# Ajustamos los límites para que se vea todo bien
plt.ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
plt.title("SVM Hard Margin")
plt.xlabel("Largo del Sépalo")
plt.ylabel("Largo del Pétalo")
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)

# Guardamos la figura que usa el cuaderno del TFG y la mostramos
plt.savefig(ruta_figura, dpi=200, bbox_inches='tight')
plt.show()
