import torch
import torch.nn as nn


class SoftPlanningRouter(nn.Module):
    def __init__(self, layer, critic, skip_rate=0.5):
        super().__init__()
        self.layer = layer
        self.critic = critic
        self.skip_rate = skip_rate

    def forward(self, h):
        # h: [batch, seq, hidden]
        with torch.no_grad():
            probs = self.critic(h.detach())  # [batch, seq]
            thresh = torch.quantile(probs, 1 - self.skip_rate)
            mask = (probs >= thresh).float().unsqueeze(-1)  # [batch, seq, 1]
        h_out = self.layer(h) * mask + h * (1 - mask)
        return h_out
