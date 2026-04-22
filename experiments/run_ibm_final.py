import json
import numpy as np
from datetime import datetime
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_provider import IBMProvider

# -----------------------
# 1. Load IBM provider
# -----------------------
provider = IBMProvider()

BACKEND_NAME = "ibmq_quito"  # change if needed
backend = provider.get_backend(BACKEND_NAME)
print("✅ Backend:", backend.name)

# -----------------------
# 2. Load PCA features
# -----------------------
with open("resnet_pca_quantum_log.json") as f:
    log = json.load(f)

features = np.array(log["pca_features"][:5])  # 8 → 5 qubits
print("✅ PCA features used:", features)

# -----------------------
# 3. Build quantum circuit
# -----------------------
qc = QuantumCircuit(5, 5)

for i in range(5):
    qc.rx(features[i], i)

for i in range(4):
    qc.cx(i, i + 1)

qc.measure(range(5), range(5))

# -----------------------
# 4. Transpile
# -----------------------
qc_t = transpile(qc, backend, optimization_level=3)
print("✅ Circuit depth:", qc_t.depth())

# -----------------------
# 5. Execute on hardware
# -----------------------
job = backend.run(qc_t, shots=1024)
print("✅ Job ID:", job.job_id())

result = job.result()
counts = result.get_counts()
print("✅ Counts:", counts)

# -----------------------
# 6. Save logs
# -----------------------
hw_log = {
    "timestamp": str(datetime.now()),
    "backend": backend.name,
    "job_id": job.job_id(),
    "shots": 1024,
    "circuit_depth": qc_t.depth(),
    "counts": counts
}

with open("ibm_hardware_log.json", "w") as f:
    json.dump(hw_log, f, indent=4)

print("✅ Hardware log saved to ibm_hardware_log.json")