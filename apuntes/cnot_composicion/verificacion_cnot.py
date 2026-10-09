# Verificación de los cálculos de "CNOT con control Y y composición" (Cap. 3, Sec. 3.1)
# Convención: |ab> = |a>_X ⊗ |b>_Y, X a la izquierda; Y es q0 (cable de arriba).
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

np.set_printoptions(precision=4, suppress=True)
s = 1/np.sqrt(2)
I = np.eye(2); H = s*np.array([[1, 1], [1, -1]]); X = np.array([[0, 1], [1, 0]])
k0 = np.array([[1], [0]]); k1 = np.array([[0], [1]])
P0, P1 = k0 @ k0.T, k1 @ k1.T
def ket(cad):   # |ab> -> vector de C^4
    return np.eye(4)[int(cad, 2)]

# 1. I⊗H frente a H⊗I
IH, HI = np.kron(I, H), np.kron(H, I)
print("sqrt2*(I⊗H) =\n", np.sqrt(2)*IH)
print("sqrt2*(H⊗I) =\n", np.sqrt(2)*HI)
print("(I⊗H)|01> * sqrt2 =", np.sqrt(2)*IH @ ket("01"), " (= |0>|->)")
print("(H⊗I)|01> * sqrt2 =", np.sqrt(2)*HI @ ket("01"), " (= |+>|1>)")
print("¿(I⊗H)|01> == |0>⊗H|1>?", np.allclose(IH @ ket("01"), np.kron(k0, H @ k1).ravel()))

# 2. CNOT_{Y->X}: |ab> -> |(a xor b) b>
C_YX = np.zeros((4, 4))
for a in (0, 1):
    for b in (0, 1):
        C_YX[2*(a ^ b) + b, 2*a + b] = 1
print("CNOT_{Y->X} =\n", C_YX)
print("¿= I⊗|0><0| + X⊗|1><1|?", np.allclose(C_YX, np.kron(I, P0) + np.kron(X, P1)))
CX = np.kron(P0, I) + np.kron(P1, X)           # CX del capítulo 2
print("CX (cap. 2) =\n", CX)
print("¿CNOT_{Y->X} == CX?", np.allclose(C_YX, CX))
SWAP = np.zeros((4, 4))
for a in (0, 1):
    for b in (0, 1):
        SWAP[2*b + a, 2*a + b] = 1
print("¿CNOT_{Y->X} == SWAP·CX·SWAP?", np.allclose(C_YX, SWAP @ CX @ SWAP))
print("¿CNOT_{Y->X} == (H⊗H)·CX·(H⊗H)?", np.allclose(C_YX, np.kron(H, H) @ CX @ np.kron(H, H)))
print("¿unitaria? (C^T C = I)", np.allclose(C_YX.T @ C_YX, np.eye(4)))
print("CX|01> =", CX @ ket("01"), "  CNOT_{Y->X}|01> =", C_YX @ ket("01"))

# 3. Composición
U = C_YX @ IH
print("sqrt2*U = sqrt2*CNOT_{Y->X}(I⊗H) =\n", np.sqrt(2)*U)
print("sqrt2*(I⊗H)CNOT_{Y->X} (orden equivocado) =\n", np.sqrt(2)*IH @ C_YX)
for c in ("00", "01", "10", "11"):
    medio = IH @ ket(c)
    print(f"|{c}> --(I⊗H)--> sqrt2*", np.sqrt(2)*medio, " --CNOT--> sqrt2*", np.sqrt(2)*(C_YX @ medio))
phi_p = s*np.array([1, 0, 0, 1]); phi_m = s*np.array([1, 0, 0, -1])
psi_p = s*np.array([0, 1, 1, 0]); psi_m = s*np.array([0, 1, -1, 0])
print("¿columnas = φ+, φ-, ψ+, -ψ-?", np.allclose(U, np.column_stack([phi_p, phi_m, psi_p, -psi_m])))
print("Circuito del cap. 2, CX(H⊗I): sqrt2*", "\n", np.sqrt(2)*CX @ HI)

# 4. Qiskit
qc = QuantumCircuit(2)      # q0 = Y (arriba), q1 = X (abajo)
qc.h(0)
qc.cx(0, 1)
print(qc.draw())
print("¿Operator(h(0); cx(0,1)) == U?", np.allclose(Operator(qc).data, U))
q01 = QuantumCircuit(2); q01.cx(0, 1)
q10 = QuantumCircuit(2); q10.cx(1, 0)
print("¿cx(0,1) == CNOT_{Y->X}?", np.allclose(Operator(q01).data, C_YX))
print("¿cx(1,0) == CX del cap. 2?", np.allclose(Operator(q10).data, CX))
print("Statevector desde |00>:", Statevector.from_label("00").evolve(qc).probabilities_dict())

# 5. Ejemplo con amplitudes distintas: |psi> = (|00> + 2|01> + 4|10> + 3|11>)/sqrt(30)
v = np.array([1, 2, 4, 3])/np.sqrt(30)
print("norma de v:", np.linalg.norm(v))
print("sqrt30 * CNOT_{Y->X} v =", np.sqrt(30)*C_YX @ v, "  sqrt30 * CX v =", np.sqrt(30)*CX @ v)
print("sqrt60 * (I⊗H) v       =", np.sqrt(60)*IH @ v)
print("sqrt60 * U v           =", np.sqrt(60)*U @ v, " probabilidades:", (U @ v)**2*60, "/60")
print("sqrt60 * (I⊗H)CNOT v   =", np.sqrt(60)*IH @ C_YX @ v)
qv = QuantumCircuit(2); qv.initialize(v, [0, 1]); qv.h(0); qv.cx(0, 1)
print("Qiskit, U|psi> * sqrt60 =", np.real(Statevector(qv).data)*np.sqrt(60))
