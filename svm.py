
import numpy as np
import os
import matplotlib.pyplot as plt

carpeta_actual = os.path.dirname(os.path.abspath(__file__))

ruta_archivo = os.path.join(carpeta_actual, 'billetes.csv')

# Leemos el archivo directamente a una matriz matemática
matriz_datos = np.loadtxt(ruta_archivo, delimiter=',')

# Seleccionamos solo las 2 primeras columnas (Varianza y Asimetría) para poder dibujarlo en 2D
X = matriz_datos[:, [0, 1]] 

# La última columna es la etiqueta (0 = billete falso, 1 = billete verdadero)
y = matriz_datos[:, -1]

# Convertimos a -1 y 1 para nuestra SVM
y = np.where(y == 0, -1, 1)

# Averiguamos cuántas características tenemos (en nuestro caso 2: Varianza y Asimetría)
n_muestras, n_caracteristicas = X.shape

# w es un vector lleno de ceros, del tamaño de nuestras características [0.0, 0.0]
w = np.zeros(n_caracteristicas)

# b es un simple número escalar que empieza en cero
b = 0.0

tasa_aprendizaje = 0.001
iteraciones = 1000
parametro_lambda = 0.01

for epoca in range(iteraciones):
    # Leemos cada billete (x_i) con su etiqueta (y_i) uno por uno
    for i, x_i in enumerate(X):
        
        # Comprobamos si y_i * (w^T * x_i + b) >= 1
        condicion_margen = y[i] * (np.dot(x_i, w) + b) >= 1
        
        if condicion_margen:
            # El punto está bien clasificado y fuera del margen.
            derivada_w = 2 * parametro_lambda * w
            derivada_b = 0
        else:
            # El punto está mal clasificado o pisando el margen 
            derivada_w = 2 * parametro_lambda * w - np.dot(y[i], x_i)
            derivada_b = -y[i]
            
        # Actualizamos las variables restando la derivada (Descenso de Gradiente)
        w = w - tasa_aprendizaje * derivada_w
        b = b - tasa_aprendizaje * derivada_b


# Dibujamos los puntos originales
plt.figure(figsize=(10, 6))
plt.scatter(X[y == -1][:, 0], X[y == -1][:, 1], color='red', label='Falso (-1)', edgecolors='k', alpha=0.7)
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='green', label='Verdadero (1)', edgecolors='k', alpha=0.7)

# Creamos un rango de valores para el eje X (Varianza) que cubra todos nuestros datos
x_recta = np.linspace(np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1, 100)

# Despejamos el eje Y para obtener las tres rectas:
# Hiperplano central (w^T x + b = 0)
y_recta = -(w[0] * x_recta + b) / w[1]

# Margen w^T x + b = 1
y_margen_pos = -(w[0] * x_recta + b - 1) / w[1]

# Margen w^T x + b = -1
y_margen_neg = -(w[0] * x_recta + b + 1) / w[1]

plt.plot(x_recta, y_recta, 'k-', linewidth=2, label='Hiperplano Separador')
plt.plot(x_recta, y_margen_pos, 'k--', alpha=0.5, label='Margen (+1)')
plt.plot(x_recta, y_margen_neg, 'k--', alpha=0.5, label='Margen (-1)')

# 5. Ajustes estéticos del gráfico
plt.ylim(np.min(X[:, 1]) - 2, np.max(X[:, 1]) + 2) # Limitamos la vista para que no se aleje mucho

plt.xlabel("Varianza de la imagen")
plt.ylabel("Asimetría de la imagen")
plt.legend(loc='lower left')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print(f"Ecuación del Hiperplano: ({w[0]:.4f})*x1 + ({w[1]:.4f})*x2 + ({b:.4f}) = 0")
print(f"Vector de Pesos (w):     [{w[0]:.4f}, {w[1]:.4f}]")
print(f"Sesgo / Bias (b):        {b:.4f}")