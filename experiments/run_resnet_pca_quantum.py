import torch
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import numpy as np
import pennylane as qml
import json
from sklearn.decomposition import PCA
from datetime import datetime

# -----------------------
# 1. Image preprocessing
# -----------------------
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

img = Image.open("data/sample_xray.png").convert("RGB")
x = transform(img).unsqueeze(0)

# -----------------------
# 2. ResNet50 feature extraction
# -----------------------
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()

with torch.no_grad():
    features = resnet(x)

features = features.view(1, -1).numpy()  # (1, 2048)

print("✅ ResNet features shape:", features.shape)

# -----------------------
# 3. PCA (2048 → 8)
# -----------------------
# -----------------------
# 3. PCA (fit on multiple samples)
# -----------------------
import os

feature_list = []

for file in os.listdir("data"):
    if file.endswith(".png"):
        img = Image.open(f"data/{file}").convert("RGB")
        x = transform(img).unsqueeze(0)

        with torch.no_grad():
            f = resnet(x)

        f = f.view(1, -1).numpy()
        feature_list.append(f)

feature_matrix = np.vstack(feature_list)  # shape (N_images, 2048)

pca = PCA(n_components=8)
pca.fit(feature_matrix)

# Transform ONE image (the first one)
features_pca = pca.transform(feature_matrix[:1])

print("✅ PCA features shape:", features_pca.shape)
print("✅ Explained variance:", pca.explained_variance_ratio_.sum())


# -----------------------
# 4. Quantum simulation
# -----------------------
N_QUBITS = 8
dev = qml.device("default.qubit", wires=N_QUBITS)

@qml.qnode(dev)
def vqc(inputs):
    for i in range(N_QUBITS):
        qml.RX(inputs[i], wires=i)

    for i in range(N_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])

    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

quantum_out = vqc(features_pca[0])

print("✅ Quantum outputs:", quantum_out)

# -----------------------
# 5. Save logs
# -----------------------
log = {
    "timestamp": str(datetime.now()),
    "image": "data/sample_xray.png",
    "resnet_features_shape": features.shape,
    "pca_features": features_pca[0].tolist(),
    "explained_variance": float(pca.explained_variance_ratio_.sum()),
    "quantum_outputs": [float(q) for q in quantum_out]
}

with open("resnet_pca_quantum_log.json", "w") as f:
    json.dump(log, f, indent=4)

print("✅ Log saved to resnet_pca_quantum_log.json")