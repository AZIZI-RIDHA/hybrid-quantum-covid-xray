# Hybrid Classical–Quantum Deep Learning for COVID‑19 X‑ray Classification

Reproducible source code for the paper:
Hybrid classical and quantum deep learning for COVID‑19 chest X‑ray image classification on IBM quantum hardware.

## Pipeline
ResNet50 → PCA (2048→8) → Variational Quantum Circuit → Classification

## Dataset
COVID‑19 Radiography Database (Kaggle)
https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database

Dataset is not included due to license restrictions.

## Usage
pip install -r requirements.txt
python main.py
