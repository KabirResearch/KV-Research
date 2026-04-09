import torch
import torch.nn as nn

class SkipLayer(torch.nn.Module):
    def __init__(self, layer, layer_id, skip_layers_ref):
        super().__init__()
        self.layer = layer
        self.layer_id = layer_id
        self.skip_layers_ref = skip_layers_ref

    def forward(self, hidden_states, *args, **kwargs):
        if self.layer_id in self.skip_layers_ref:
            return (hidden_states, None)
        return self.layer(hidden_states, *args, **kwargs)

class EntropySkipLayer(torch.nn.Module):
    def __init__(self, layer, layer_id, config):
        super().__init__()
        self.layer = layer
        self.layer_id = layer_id
        self.config = config

    def forward(self, hidden_states, *args, **kwargs):
        if not self.config["enable_skip"]:
            return self.layer(hidden_states, *args, **kwargs)
        if self.layer_id < self.config["min_layers"]:
            return self.layer(hidden_states, *args, **kwargs)
        if self.config["current_entropy"] < self.config["threshold"]:
            self.config["skipped"] += 1
            return (hidden_states, None)
        return self.layer(hidden_states, *args, **kwargs)

class VoCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

class Router(nn.Module):
    def __init__(self, num_layers=24, per_token=False):
        super().__init__()
        self.num_layers = num_layers
        self.per_token = per_token
        if per_token:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(4) for _ in range(num_layers)])  # normalize delta, var, ratio, abs_mean
            self.net = nn.Sequential(
                nn.Linear(5, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1)
            )
        else:
            self.layer_norms = nn.ModuleList([nn.LayerNorm(4) for _ in range(num_layers)])  # normalize delta, var, ratio, abs_mean
            self.net = nn.Sequential(
                nn.Linear(5, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1)
            )

    def forward(self, x):
        # x: [batch, 4] or [batch, seq, 5] for per_token
        if self.per_token:
            batch, seq, _ = x.shape
            x_flat = x.view(-1, 5)
            layer_pos = x_flat[:, -1]
            layer_ids = (layer_pos * self.num_layers).long().clamp(0, self.num_layers-1)
            features = x_flat[:, :-1]
            normalized_features = torch.zeros_like(features)
            for i in range(batch * seq):
                lid = layer_ids[i].item()
                normalized_features[i] = self.layer_norms[lid](features[i:i+1])
            x_norm = torch.cat([normalized_features, layer_pos.unsqueeze(1)], dim=1)
            scores = self.net(x_norm).view(batch, seq)
            return scores
        else:
            layer_id = int(x[0, -1] * self.num_layers)
            layer_id = min(layer_id, self.num_layers - 1)
            features = x[:, :-1]
            normalized_features = self.layer_norms[layer_id](features)
            x_norm = torch.cat([normalized_features, x[:, -1:]], dim=1)
            return self.net(x_norm)

class VoCSkipLayer(torch.nn.Module):
    def __init__(self, layer, layer_id, config, router):
        super().__init__()
        self.layer = layer
        self.layer_id = layer_id
        self.config = config
        self.router = router
        for name, module in layer.named_children():
            setattr(self, name, module)

    def forward(self, hidden_states, *args, **kwargs):
        prev = self.config.get("prev_hidden", None)
        if prev is None:
            prev = hidden_states
        # ---- features ----
        delta = (hidden_states - prev).norm(dim=-1).mean(dim=1)
        var = hidden_states.var(dim=1).mean(dim=1)
        ratio = hidden_states.norm(dim=[1,2]) / (prev.norm(dim=[1,2]) + 1e-6)
        abs_mean = hidden_states.abs().mean(dim=[1,2])
        # ---- layer context ----
        layer_pos = torch.full(
            (hidden_states.size(0),),
            self.layer_id / self.config["num_layers"],
            device=hidden_states.device
        )
        feat = torch.stack([delta, var, ratio, abs_mean, layer_pos], dim=1).float()
        self.config["prev_hidden"] = hidden_states.detach()
        # =========================
        # DATA COLLECTION
        # =========================
        if self.config.get("collect_data", False):
            with torch.no_grad():
                full_out = self.layer(hidden_states, *args, **kwargs)
                full_h = full_out[0] if isinstance(full_out, tuple) else full_out
            skip_h = hidden_states
            # better importance (per-token avg)
            importance = (full_h - skip_h).norm(dim=-1).mean().item()
            self.config["records"].append({
                "feat": feat.detach().cpu(),
                "label": importance
            })
            return full_out
        # =========================
        # INFERENCE
        # =========================
        score = self.router(feat).mean().item()
        if score < self.config["threshold"]:
            self.config["skipped"] += 1
            # Residual-safe skip: output = input (matches transformer math)
            return hidden_states
        return self.layer(hidden_states, *args, **kwargs)

class TokenLevelVoCSkipLayer(torch.nn.Module):
    def __init__(self, layer, layer_id, config, router):
        super().__init__()
        self.layer = layer
        self.layer_id = layer_id
        self.config = config
        self.router = router
        for name, module in layer.named_children():
            setattr(self, name, module)

    def forward(self, hidden_states, *args, **kwargs):
        prev = self.config.get("prev_hidden", None)
        if prev is None:
            prev = hidden_states
        # ---- per-token features ----
        delta = (hidden_states - prev).norm(dim=-1)  # [batch, seq]
        var = hidden_states.var(dim=-1)  # [batch, seq]
        ratio = hidden_states.norm(dim=-1) / (prev.norm(dim=-1) + 1e-6)  # [batch, seq]
        abs_mean = hidden_states.abs().mean(dim=-1)  # [batch, seq]
        layer_pos = torch.full_like(delta, self.layer_id / self.config["num_layers"])
        feat = torch.stack([delta, var, ratio, abs_mean, layer_pos], dim=-1).float()  # [batch, seq, 5]
        self.config["prev_hidden"] = hidden_states.detach()
        # =========================
        # DATA COLLECTION
        # =========================
        if self.config.get("collect_data", False):
            with torch.no_grad():
                full_out = self.layer(hidden_states, *args, **kwargs)
                full_h = full_out[0] if isinstance(full_out, tuple) else full_out
            skip_h = hidden_states
            # per-token importance
            importance = (full_h - skip_h).norm(dim=-1).mean(dim=1).mean().item()  # avg over batch and seq
            self.config["records"].append({
                "feat": feat.detach().cpu(),
                "label": importance
            })
            return full_out
        # =========================
        # INFERENCE
        # =========================
        scores = self.router(feat)  # [batch, seq]
        mask = scores < self.config["threshold"]  # [batch, seq], True for skip
        self.config["skipped"] += mask.sum().item()
        # Residual-safe skip: for skipped tokens, output = input
        full_out = self.layer(hidden_states, *args, **kwargs)
        full_h = full_out[0] if isinstance(full_out, tuple) else full_out
        full_h = torch.where(mask.unsqueeze(-1), hidden_states, full_h)
        if isinstance(full_out, tuple):
            return (full_h,) + full_out[1:]
        else:
            return full_h

def apply_skip(model, skip_layers):
    for i, layer in enumerate(model.gpt_neox.layers):
        model.gpt_neox.layers[i] = SkipLayer(layer, i, skip_layers)

def apply_entropy_skip(model, config):
    for i, layer in enumerate(model.gpt_neox.layers):
        model.gpt_neox.layers[i] = EntropySkipLayer(layer, i, config)

def apply_voc_skip(model, config, router):
    config["num_layers"] = len(model.gpt_neox.layers)
    for i, layer in enumerate(model.gpt_neox.layers):
        model.gpt_neox.layers[i] = VoCSkipLayer(layer, i, config, router)

def apply_token_level_voc_skip(model, config, router):
    config["num_layers"] = len(model.gpt_neox.layers)
    router.per_token = True
    for i, layer in enumerate(model.gpt_neox.layers):
        model.gpt_neox.layers[i] = TokenLevelVoCSkipLayer(layer, i, config, router)