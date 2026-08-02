import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, L: int = 32, H: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(L, H)
        self.fc2 = nn.Linear(H, L)
        self.L = L
        self.H = H

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.relu(self.fc1(x)))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
