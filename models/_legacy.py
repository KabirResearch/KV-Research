"""
Legacy model classes (VoCModel, Router, VoCSkipLayer, etc.) kept for backward compatibility.
New code should use models.critics.LogTemporalCritic and models.router.SoftPlanningRouter.
"""

import torch
import torch.nn as nn


class VoCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x)


class Router(nn.Module):
    def __init__(self, num_layers=24, per_token=False):
        super().__init__()
        self.num_layers = num_layers
        self.per_token = per_token
        self.layer_norms = nn.ModuleList([nn.LayerNorm(4) for _ in range(num_layers)])
        self.net = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        if self.per_token:
            batch, seq, _ = x.shape
            x_flat = x.view(-1, 5)
            layer_pos = x_flat[:, -1]
            layer_ids = (layer_pos * self.num_layers).long().clamp(0, self.num_layers - 1)
            features = x_flat[:, :-1]
            norm_feat = torch.zeros_like(features)
            for i in range(batch * seq):
                lid = layer_ids[i].item()
                norm_feat[i] = self.layer_norms[lid](features[i : i + 1])
            x_norm = torch.cat([norm_feat, layer_pos.unsqueeze(1)], dim=1)
            return self.net(x_norm).view(batch, seq)
        else:
            layer_id = min(int(x[0, -1] * self.num_layers), self.num_layers - 1)
            features = self.layer_norms[layer_id](x[:, :-1])
            x_norm = torch.cat([features, x[:, -1:]], dim=1)
            return self.net(x_norm)


class VoCSkipLayer(nn.Module):
    def __init__(self, layer, layer_id, config, router):
        super().__init__()
        self.layer = layer
        self.layer_id = layer_id
        self.config = config
        self.router = router

    def forward(self, hidden_states, *args, **kwargs):
        prev = self.config.get("prev_hidden", hidden_states)
        delta = (hidden_states - prev).norm(dim=-1).mean(dim=1)
        var = hidden_states.var(dim=1).mean(dim=1)
        ratio = hidden_states.norm(dim=[1, 2]) / (prev.norm(dim=[1, 2]) + 1e-6)
        abs_mean = hidden_states.abs().mean(dim=[1, 2])
        layer_pos = torch.full(
            (hidden_states.size(0),), self.layer_id / self.config["num_layers"], device=hidden_states.device
        )
        feat = torch.stack([delta, var, ratio, abs_mean, layer_pos], dim=1).float()
        self.config["prev_hidden"] = hidden_states.detach()

        if self.config.get("collect_data", False):
            with torch.no_grad():
                full_out = self.layer(hidden_states, *args, **kwargs)
                full_h = full_out[0] if isinstance(full_out, tuple) else full_out
            importance = (full_h - hidden_states).norm(dim=-1).mean().item()
            self.config["records"].append({"feat": feat.detach().cpu(), "label": importance})
            return full_out

        score = self.router(feat).mean().item()
        if score < self.config["threshold"]:
            self.config["skipped"] = self.config.get("skipped", 0) + 1
            return hidden_states
        return self.layer(hidden_states, *args, **kwargs)


class TokenLevelVoCSkipLayer(nn.Module):
    def __init__(self, layer, layer_id, config, router):
        super().__init__()
        self.layer = layer
        self.layer_id = layer_id
        self.config = config
        self.router = router

    def forward(self, hidden_states, *args, **kwargs):
        prev = self.config.get("prev_hidden", hidden_states)
        delta = (hidden_states - prev).norm(dim=-1)
        var = hidden_states.var(dim=-1)
        ratio = hidden_states.norm(dim=-1) / (prev.norm(dim=-1) + 1e-6)
        abs_mean = hidden_states.abs().mean(dim=-1)
        layer_pos = torch.full_like(delta, self.layer_id / self.config["num_layers"])
        feat = torch.stack([delta, var, ratio, abs_mean, layer_pos], dim=-1).float()
        self.config["prev_hidden"] = hidden_states.detach()

        if self.config.get("collect_data", False):
            with torch.no_grad():
                full_out = self.layer(hidden_states, *args, **kwargs)
                full_h = full_out[0] if isinstance(full_out, tuple) else full_out
            importance = (full_h - hidden_states).norm(dim=-1).mean().item()
            self.config["records"].append({"feat": feat.detach().cpu(), "label": importance})
            return full_out

        scores = self.router(feat)
        mask = (scores < self.config["threshold"]).unsqueeze(-1)
        full_out = self.layer(hidden_states, *args, **kwargs)
        full_h = full_out[0] if isinstance(full_out, tuple) else full_out
        full_h = torch.where(mask, hidden_states, full_h)
        return (full_h,) + (full_out[1:] if isinstance(full_out, tuple) else ())
