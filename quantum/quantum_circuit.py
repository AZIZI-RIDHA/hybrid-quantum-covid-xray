import pennylane as qml
n_qubits = 8
dev = qml.device('default.qubit', wires=n_qubits)
@qml.qnode(dev)
def circuit(x, w):
    for i in range(n_qubits): qml.RX(x[i], wires=i)
    for i in range(n_qubits-1): qml.CNOT(wires=[i,i+1])
    for i in range(n_qubits): qml.RY(w[i], wires=i)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
