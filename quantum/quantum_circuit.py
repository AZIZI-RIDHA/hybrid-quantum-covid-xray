import pennylane as qml
import numpy as np

# Number of qubits after PCA reduction
n_qubits = 8

# Circuit depth as reported in the paper
n_layers = 12

# Quantum device (simulation)
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def circuit(x, weights):
    """
    Variational Quantum Circuit (VQC)
    - RX encoding layer
    - 12 variational layers (CNOT + RY)
    - Pauli-Z measurements
    """

    # -------- Embedding layer (RX encoding) --------
    for i in range(n_qubits):
        qml.RX(x[i], wires=i)

    # -------- Variational layers --------
    for layer in range(n_layers):
        # Entanglement (nearest-neighbor)
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])

        # Trainable rotations
        for i in range(n_qubits):
            qml.RY(weights[layer * n_qubits + i], wires=i)

    # -------- Measurement layer --------
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
``