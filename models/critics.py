import torch
import torch.nn as nn


class LogTemporalCritic(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=256):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        # h: [batch, seq, in_dim]
        x = torch.log1p(h.abs())
        x = self.norm(x)
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x.squeeze(-1)  # [batch, seq]
