"""
Patchers — apply skip/routing wrappers to model layers.
"""

import torch.nn as nn


class _IdentitySkipLayer(nn.Module):
    """Stub: always passes hidden state through unchanged."""

    def __init__(self, layer):
        super().__init__()
        self._orig = layer

    def forward(self, hidden_states, *args, **kwargs):
        return (hidden_states, None)


class _EntropySkipLayer(nn.Module):
    def __init__(self, layer, layer_id, cfg):
        super().__init__()
        self.layer = layer
        self.layer_id = layer_id
        self.cfg = cfg

    def forward(self, hidden_states, *args, **kwargs):
        if not self.cfg.get("enable_skip", True):
            return self.layer(hidden_states, *args, **kwargs)
        if self.layer_id < self.cfg.get("min_layers", 0):
            return self.layer(hidden_states, *args, **kwargs)
        if self.cfg.get("current_entropy", 0) < self.cfg.get("threshold", 1e9):
            self.cfg["skipped"] = self.cfg.get("skipped", 0) + 1
            return (hidden_states, None)
        return self.layer(hidden_states, *args, **kwargs)


def apply_skip(model, skip_layers: set):
    """Replace layers in skip_layers with identity (no-op)."""
    for i in skip_layers:
        model.gpt_neox.layers[i] = _IdentitySkipLayer(model.gpt_neox.layers[i])


def apply_entropy_skip(model, cfg: dict):
    """Replace each layer with an entropy-gated skip layer."""
    for i, layer in enumerate(model.gpt_neox.layers):
        model.gpt_neox.layers[i] = _EntropySkipLayer(layer, i, cfg)


def apply_voc_skip(model, cfg: dict, router):
    """Apply VoC-Router layer-level skipping (legacy)."""
    from models._legacy import VoCSkipLayer

    cfg["num_layers"] = len(model.gpt_neox.layers)
    for i, layer in enumerate(model.gpt_neox.layers):
        model.gpt_neox.layers[i] = VoCSkipLayer(layer, i, cfg, router)


def apply_token_level_voc_skip(model, cfg: dict, router):
    """Apply token-level VoC-Router skipping (legacy)."""
    from models._legacy import TokenLevelVoCSkipLayer

    cfg["num_layers"] = len(model.gpt_neox.layers)
    router.per_token = True
    for i, layer in enumerate(model.gpt_neox.layers):
        model.gpt_neox.layers[i] = TokenLevelVoCSkipLayer(layer, i, cfg, router)
