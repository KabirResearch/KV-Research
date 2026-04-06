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
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

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
        # ---- layer context ----
        layer_pos = torch.full(
            (hidden_states.size(0),),
            self.layer_id / self.config["num_layers"],
            device=hidden_states.device
        )
        feat = torch.stack([delta, var, ratio, layer_pos], dim=1).float()
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
            if kwargs.get("use_cache", False):
                return (hidden_states, None, None)
            else:
                return (hidden_states, None)
        return self.layer(hidden_states, *args, **kwargs)

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