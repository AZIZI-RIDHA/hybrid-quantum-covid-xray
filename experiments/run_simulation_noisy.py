import pennylane as qml
import numpy as np
import json
from datetime import datetime

# Configuration
N_QUBITS = 8

dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev)
def vqc(inputs, weights):
    # RX encoding
    for i in range(N_QUBITS):
        qml.RX(inputs[i], wires=i)

    # Entanglement
    for i in range(N_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])

    # Variational layer
    for i in range(N_QUBITS):
        qml.RY(weights[i], wires=i)

    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

# Dummy PCA features (simulate PCA output)
inputs = np.random.rand(N_QUBITS)
weights = np.random.rand(N_QUBITS)

outputs = vqc(inputs, weights)

print("Inputs:", inputs)
print("Weights:", weights)
print("Outputs:", outputs)

# Save logs
log = {
    "timestamp": str(datetime.now()),
    "backend": "default.qubit",
    "inputs": inputs.tolist(),
    "weights": weights.tolist(),
    "outputs": [float(o) for o in outputs]
}

with open("simulation_log.json", "w") as f:
    json.dump(log, f, indent=4)

print("✅ Simulation log saved to simulation_log.json")
