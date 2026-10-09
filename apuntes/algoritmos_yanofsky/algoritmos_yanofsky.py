# %% [markdown]
# # Algoritmos cuánticos (Yanofsky y Mannucci, secciones 6.1-6.4) en Qiskit
# Experimentos del cuaderno `algoritmos_yanofsky.tex`. Ejecutado con Qiskit 2.2.3.
# Convención de Qiskit: el qubit q0 es el cable de arriba y el factor de la derecha.
# El registro de consulta x ocupa q0..q_{n-1}; el registro de salida y, los siguientes.

# %%
import os
import numpy as np
from math import sqrt, asin, sin, pi, floor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector, Operator
from qiskit.primitives import StatevectorSampler

FIG = os.environ.get("FIGDIR", "figuras_yanofsky")
os.makedirs(FIG, exist_ok=True)
ROSA, SALMON, TINTA, GRIS = "#C2185B", "#E9724C", "#333333", "#9E9E9E"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRIS, "axes.labelcolor": TINTA,
                     "xtick.color": TINTA, "ytick.color": TINTA,
                     "axes.spines.top": False, "axes.spines.right": False})

sampler = StatevectorSampler(seed=np.random.default_rng(2026))  # semilla: resultados reproducibles


def contar(circuito, shots=1000):
    """Ejecuta un circuito con mediciones y devuelve el recuento."""
    resultado = sampler.run([circuito], shots=shots).result()[0]
    return resultado.data.c.get_counts()


def barras(ax, recuento, titulo, color=ROSA, etiquetas=None):
    etiquetas = etiquetas or sorted(recuento)
    valores = [recuento.get(k, 0) for k in etiquetas]
    ax.bar(etiquetas, valores, color=color, width=0.55)
    for i, v in enumerate(valores):
        if v:
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9, color=TINTA)
    ax.set_title(titulo, fontsize=10, color=TINTA)
    ax.set_ylim(0, max(valores) * 1.18)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)


def guardar(fig, nombre):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, nombre), dpi=200, transparent=True)
    plt.close(fig)


def oraculo(f, n, m, etiqueta="            U_f"):
    """Puerta U_f |y>|x> = |y xor f(x)>|x> (x en los n primeros qubits, y en los m siguientes).
    f recibe y devuelve enteros (la cadena de bits de Qiskit leída en binario)."""
    N = 2 ** (n + m)
    U = np.zeros((N, N))
    for x in range(2 ** n):
        for y in range(2 ** m):
            U[(y ^ f(x)) * 2 ** n + x, y * 2 ** n + x] = 1
    return UnitaryGate(U, label=etiqueta)


def de_tabla(tabla):
    """Convierte un diccionario {'01': '1', ...} en una función entera."""
    return lambda x: int(tabla[format(x, "0%db" % len(next(iter(tabla))))], 2)


# %% [markdown]
# ## Experimento 1: el algoritmo de Deutsch con las cuatro funciones

# %%
def oraculo_deutsch(nombre):
    """Oráculos de un bit construidos con puertas: q0 = x (arriba), q1 = y (abajo)."""
    qc = QuantumCircuit(2, name="U_f")
    if nombre == "f2":      # identidad: y <- y xor x
        qc.cx(0, 1)
    elif nombre == "f3":    # NOT: y <- y xor x xor 1
        qc.cx(0, 1)
        qc.x(1)
    elif nombre == "f4":    # constante 1: y <- y xor 1
        qc.x(1)
    return qc               # f1 (constante 0): no hace nada


tablas_deutsch = {"f1": [0, 0], "f2": [0, 1], "f3": [1, 0], "f4": [1, 1]}
for nombre, tabla in tablas_deutsch.items():
    U_puertas = Operator(oraculo_deutsch(nombre))
    U_tabla = Operator(oraculo(lambda x: tabla[x], 1, 1))
    print(nombre, "oráculo con puertas == U_f:", U_puertas.equiv(U_tabla))


def deutsch(nombre):
    qc = QuantumCircuit(2, 1)
    qc.x(1)                 # y = |1>
    qc.h([0, 1])
    qc.barrier()
    qc.compose(oraculo_deutsch(nombre), inplace=True)
    qc.barrier()
    qc.h(0)
    qc.measure(0, 0)
    return qc


fig, axs = plt.subplots(1, 4, figsize=(9, 2.4))
for ax, nombre in zip(axs, tablas_deutsch):
    recuento = contar(deutsch(nombre))
    print(nombre, recuento)
    barras(ax, recuento, nombre + (" (constante)" if nombre in ("f1", "f4") else " (equilibrada)"),
           etiquetas=["0", "1"])
guardar(fig, "deutsch_histogramas.png")

qc = deutsch("f3")
qc.draw("mpl", filename=os.path.join(FIG, "deutsch_circuito.png"), style={"backgroundcolor": "#FFFFFF"})
print(qc.draw("text"))

# Estados intermedios para f3, comparados con el cálculo a mano
qc_sin_medida = deutsch("f3").remove_final_measurements(inplace=False)
psi = Statevector.from_label("00").evolve(qc_sin_medida)
print("Estado final (f3) * sqrt(2):", np.round(psi.data.real * sqrt(2), 3))

# %% [markdown]
# ## Experimento 2: Deutsch-Jozsa

# %%
def deutsch_jozsa(f, n):
    qc = QuantumCircuit(n + 1, n)
    qc.x(n)
    qc.h(range(n + 1))
    qc.barrier()
    qc.append(oraculo(f, n, 1), range(n + 1))
    qc.barrier()
    qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc


funciones_dj = {
    "(6.45) equilibrada": {"00": "1", "01": "1", "10": "0", "11": "0"},
    "(6.47) equilibrada": {"00": "0", "01": "0", "10": "1", "11": "1"},
    "constante 1": {"00": "1", "01": "1", "10": "1", "11": "1"},
    "AND (ni lo uno ni lo otro)": {"00": "0", "01": "0", "10": "0", "11": "1"},
}
fig, axs = plt.subplots(1, 4, figsize=(10, 2.7))
for ax, (nombre, tabla) in zip(axs, funciones_dj.items()):
    qc = deutsch_jozsa(de_tabla(tabla), 2)
    recuento = contar(qc)
    sv = Statevector.from_label("000").evolve(qc.remove_final_measurements(inplace=False))
    print(nombre, recuento, "| exacto:", {k: round(v, 4) for k, v in sv.probabilities_dict(qargs=[0, 1]).items()})
    barras(ax, recuento, nombre, etiquetas=["00", "01", "10", "11"])
guardar(fig, "dj_histogramas.png")
deutsch_jozsa(de_tabla(funciones_dj["(6.45) equilibrada"]), 2).draw(
    "mpl", filename=os.path.join(FIG, "dj_circuito.png"), style={"backgroundcolor": "#FFFFFF"})

# Una función equilibrada de 3 bits elegida al azar
rng = np.random.default_rng(7)
valores = rng.permutation([0] * 4 + [1] * 4)
print("f equilibrada aleatoria (n=3):", {format(x, "03b"): int(v) for x, v in enumerate(valores)})
print("   recuento:", contar(deutsch_jozsa(lambda x: int(valores[x]), 3)))
print("   constante 0 (n=3):", contar(deutsch_jozsa(lambda x: 0, 3)))

# Estado final exacto para (6.45): debe ser |->  (x)  (-|10>)
qc = deutsch_jozsa(de_tabla(funciones_dj["(6.45) equilibrada"]), 2).remove_final_measurements(inplace=False)
esperado = Statevector(np.kron(np.array([1, -1]) / sqrt(2), -np.eye(4)[2]))
print("Estado final (6.45) == |-> (x) (-|10>):", Statevector.from_label("000").evolve(qc).equiv(esperado))

# %% [markdown]
# ## Experimento 3: el algoritmo de Simon

# %%
def simon(f, n):
    qc = QuantumCircuit(2 * n, n)
    qc.h(range(n))
    qc.barrier()
    qc.append(oraculo(f, n, n), range(2 * n))
    qc.barrier()
    qc.h(range(n))
    qc.measure(range(n), range(n))
    return qc


def resolver_F2(ecuaciones, n):
    """Soluciones c != 0 de z.c = 0 (mod 2) para todas las z, por eliminación gaussiana."""
    filas, pivotes = [], []
    for z in ecuaciones:
        v = int(z, 2)
        for fila, p in zip(filas, pivotes):
            if v >> p & 1:
                v ^= fila
        if v:
            p = v.bit_length() - 1
            for i in range(len(filas)):          # forma reducida
                if filas[i] >> p & 1:
                    filas[i] ^= v
            filas.append(v)
            pivotes.append(p)
    libres = [b for b in range(n) if b not in pivotes]
    soluciones = []
    for mascara in range(1, 2 ** len(libres)):
        c = 0
        for i, b in enumerate(libres):
            if mascara >> i & 1:
                c |= 1 << b
        for fila, p in zip(filas, pivotes):       # c_p = suma de las libres de la fila
            if bin(fila & c).count("1") % 2:
                c |= 1 << p
        soluciones.append(format(c, "0%db" % n))
    return len(filas), soluciones


f_694 = {"000": "100", "001": "001", "010": "101", "011": "111",
         "100": "001", "101": "100", "110": "111", "111": "101"}
f_699 = {"000": "000", "001": "100", "010": "100", "011": "000",
         "100": "010", "101": "110", "110": "110", "111": "010"}
fig, axs = plt.subplots(1, 2, figsize=(9, 2.6))
etiquetas = [format(z, "03b") for z in range(8)]
for ax, (nombre, tabla) in zip(axs, [("(6.94)", f_694), ("(6.99)", f_699)]):
    qc = simon(de_tabla(tabla), 3)
    recuento = contar(qc)
    print("Simon", nombre, recuento)
    barras(ax, recuento, "función " + nombre, etiquetas=etiquetas)
    # Ejecuciones de una en una (get_bitstrings da el resultado de cada disparo, en orden)
    disparos = sampler.run([qc], shots=20).result()[0].data.c.get_bitstrings()
    resultados = []
    for z in disparos:
        resultados.append(z)
        rango, sols = resolver_F2(resultados, 3)
        if rango == 2:
            break
    print("   ejecuciones:", resultados, "-> rango", rango, "-> c =", sols)
guardar(fig, "simon_histogramas.png")
simon(de_tabla(f_694), 3).draw("mpl", filename=os.path.join(FIG, "simon_circuito.png"),
                                style={"backgroundcolor": "#FFFFFF"})

ej_632 = ["1010110", "0010001", "1100101", "0011011", "0101001", "0011010", "0110111"]
print("Ejemplo 6.3.2:", resolver_F2(ej_632, 7))
ej_633 = ["11110000", "01101001", "10010110", "00111100", "11111111", "11000011", "10001110", "01110001"]
print("Ejercicio 6.3.3:", resolver_F2(ej_633, 8))

# %% [markdown]
# ## Experimento 4: el algoritmo de Grover

# %%
def oraculo_grover(marcado, n):
    """Oráculo U_f para f(x) = 1 solo si x = marcado, con puertas: X donde el bit es 0, MCX, X."""
    qc = QuantumCircuit(n + 1, name="U_f")
    ceros = [q for q in range(n) if not (marcado >> q) & 1]
    if ceros:
        qc.x(ceros)
    qc.mcx(list(range(n)), n)
    if ceros:
        qc.x(ceros)
    return qc


def difusion(n):
    """Inversión respecto de la media: -(H X MCZ X H) = -I + 2A (salvo el signo global)."""
    qc = QuantumCircuit(n, name="-I+2A")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


for n in (2, 3, 4):
    N = 2 ** n
    D = -np.eye(N) + 2 * np.full((N, N), 1 / N)
    print(f"n={n}: Operator(difusion) == -(-I+2A):", np.allclose(Operator(difusion(n)).data, -D))
    m = 5 % N
    print(f"      oráculo con puertas == U_f:",
          Operator(oraculo_grover(m, n)).equiv(Operator(oraculo(lambda x: int(x == m), n, 1))))


def grover(marcado, n, iteraciones, medir=True):
    qc = QuantumCircuit(n + 1, n)
    qc.x(n)
    qc.h(n)                 # qubit auxiliar en |->, una sola vez
    qc.h(range(n))
    for _ in range(iteraciones):
        qc.barrier()
        qc.compose(oraculo_grover(marcado, n), range(n + 1), inplace=True)
        qc.compose(difusion(n), range(n), inplace=True)
    if medir:
        qc.measure(range(n), range(n))
    return qc


def prob_exacta(marcado, n, k):
    sv = Statevector.from_label("0" * (n + 1)).evolve(grover(marcado, n, k, medir=False))
    return sv.probabilities_dict(qargs=list(range(n))).get(format(marcado, "0%db" % n), 0)


for n, marcado in [(3, 0b101), (4, 0b1101)]:
    th = asin(1 / sqrt(2 ** n))
    print(f"n={n}, marcado {format(marcado, '0%db' % n)}:")
    for k in range(0, 7):
        print(f"   k={k}: Pr exacta (Qiskit) = {prob_exacta(marcado, n, k):.5f}"
              f"   sin^2((2k+1)theta) = {sin((2 * k + 1) * th) ** 2:.5f}")

# Amplitudes exactas del ejemplo 6.4.2 (n = 3, marcado 101) tras 1 y 2 iteraciones
for k in (1, 2):
    sv = Statevector.from_label("0000").evolve(grover(0b101, 3, k, medir=False))
    amp = sv.data.reshape(2, 8)  # fila 0: auxiliar en |0>, fila 1: en |1>
    x_amp = (amp[0] - amp[1]) / sqrt(2)  # componente del registro x (auxiliar en |->)
    print(f"k={k}: amplitudes del registro x * sqrt(8):", np.round(x_amp.real * sqrt(8), 4))

fig, axs = plt.subplots(1, 2, figsize=(9, 2.6))
r2 = contar(grover(0b101, 3, 2))
r3 = contar(grover(0b101, 3, 3))
print("Grover n=3, 2 iteraciones:", r2)
print("Grover n=3, 3 iteraciones:", r3)
barras(axs[0], r2, "2 iteraciones (las óptimas)", etiquetas=etiquetas)
barras(axs[1], r3, "3 iteraciones (pasado de punto)", color=SALMON, etiquetas=etiquetas)
guardar(fig, "grover_histogramas.png")
grover(0b101, 3, 1).draw("mpl", filename=os.path.join(FIG, "grover_circuito.png"),
                          style={"backgroundcolor": "#FFFFFF"}, fold=-1)

# Probabilidad de éxito frente al número de iteraciones
fig, ax = plt.subplots(figsize=(6.4, 2.8))
for n, marcado, color, marca in [(3, 0b101, ROSA, "o"), (4, 0b1101, SALMON, "s")]:
    ks = list(range(0, 9))
    ps = [prob_exacta(marcado, n, k) for k in ks]
    ax.plot(ks, ps, color=color, linewidth=2, marker=marca, markersize=6, label=f"n = {n}")
ax.set_xlabel("número de iteraciones k")
ax.set_ylabel("Pr(acertar)")
ax.set_ylim(0, 1.08)
ax.grid(color="#DDDDDD", linewidth=0.6)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
guardar(fig, "grover_probabilidad.png")

# Inversión respecto de la media (figuras 6.1 y 6.2 de Yanofsky)
V = np.array([53, 38, 17, 23, 79])
V2 = 2 * V.mean() - V
fig, axs = plt.subplots(1, 2, figsize=(9, 2.6))
for ax, datos, titulo, color in [(axs[0], V, "antes", ROSA), (axs[1], V2, "después de invertir respecto de la media", SALMON)]:
    ax.bar(range(1, 6), datos, color=color, width=0.5)
    ax.axhline(42, color=TINTA, linewidth=1, linestyle="--")
    ax.text(5.4, 43.5, "media 42", fontsize=9, color=TINTA, ha="left")
    ax.set_xlim(0.4, 6.4)
    for i, v in enumerate(datos, start=1):
        ax.text(i, v + 1.5, f"{v:g}", ha="center", fontsize=9, color=TINTA)
    ax.set_title(titulo, fontsize=10, color=TINTA)
    ax.set_xticks(range(1, 6))
    ax.set_ylim(0, 90)
guardar(fig, "grover_media.png")

# Amplitudes del ejemplo 6.4.2 paso a paso
pasos = []
v = np.full(8, 1 / sqrt(8))
pasos.append(("$|\\varphi_2\\rangle$", v.copy()))
for k in (1, 2):
    v[5] = -v[5]
    pasos.append((f"inversión de fase {k}", v.copy()))
    v = 2 * v.mean() - v
    pasos.append((f"inversión media {k}", v.copy()))
fig, axs = plt.subplots(1, 5, figsize=(10, 2.3))
for ax, (titulo, datos) in zip(axs, pasos):
    colores = [SALMON if i == 5 else ROSA for i in range(8)]
    ax.bar(range(8), datos, color=colores, width=0.6)
    ax.axhline(0, color=GRIS, linewidth=0.8)
    ax.axhline(datos.mean(), color=TINTA, linewidth=0.8, linestyle="--")
    ax.set_title(titulo, fontsize=9, color=TINTA)
    ax.set_xticks(range(8))
    ax.set_xticklabels(etiquetas, rotation=90, fontsize=7)
    ax.set_ylim(-0.5, 1.05)
guardar(fig, "grover_amplitudes.png")
print("Figuras guardadas en", FIG)
