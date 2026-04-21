import torch
from .quantum_circuit import circuit
class HybridModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(8,1)
        self.weights = torch.nn.Parameter(torch.randn(8))
    def forward(self,x):
        q = torch.tensor([circuit(x[i], self.weights) for i in range(len(x))])
        return torch.sigmoid(self.fc(q))
