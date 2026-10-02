import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# 1. CARGA DE DATOS (Iris Setosa vs Versicolor)
iris = datasets.load_iris()
# Usamos largo del sépalo y largo del pétalo
X = iris.data[:100, [0, 2]] 
y = iris.target[:100]
y = np.where(y == 0, -1, 1)

# 2. OPTIMIZACIÓN DUAL (Ascenso de Gradiente) 
# Queremos encontrar los multiplicadores de Lagrange (alphas)
n_samples, n_features = X.shape
alphas = np.zeros(n_samples)
eta = 0.0001     # Tasa más pequeña para que no "explote" el cálculo
n_iters = 10000  # Más tiempo para que los alphas encuentren el valor óptimo

# Precalculamos la Matriz de Gram (productos escalares xi * xj)
K = np.dot(X, X.T)

for i in range(n_iters):
    # Gradiente de la función Dual: 1 - y_i * sum(alpha_j * y_j * K_ij)
    gradiente = np.ones(n_samples) - y * np.dot(K, alphas * y)
    alphas += eta * gradiente
    
    # Aplicamos la restricción KKT: alpha >= 0
    alphas[alphas < 0] = 0

# 3. RECONSTRUCCIÓN DE PARÁMETROS
# w = sum(alpha_i * y_i * x_i)
w = np.sum((alphas * y)[:, None] * X, axis=0)

# Buscamos los Vectores de Soporte (donde alpha > 0) para calcular b
idx = np.where(alphas > 1e-5)[0]
if len(idx) > 0:
    # b = y_i - w*x_i 
    b = np.mean(y[idx] - np.dot(X[idx], w))
else:
    b = 0
    print("No se detectaron vectores de soporte. La línea podría no salir.")

# 4. GRÁFICO 
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
plt.show()

print(f"b = {b:.4f}, w = {w}")