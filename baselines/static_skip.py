"""
Static Layer Skipping Baseline
================================
Skips a fixed set of upper transformer layers regardless of input.
No learning or adaptivity — serves as a hard lower-bound reference.

Pseudocode:
    skip_set = top skip_rate% of layers (by index)
    for i in skip_set:
        layer[i] = IdentityLayer  (h_out = h_in)
"""
import torch
import torch.nn as nn


class IdentityLayer(nn.Module):
    """Wraps a transformer layer with an identity forward (residual-safe skip)."""
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states, *args, **kwargs):
        return (hidden_states,)


def apply_static_skip(model, skip_rate: float = 0.25):
    """
    Wrap the top `skip_rate` fraction of layers with IdentityLayer.

    Args:
        model: HuggingFace causal LM with model.gpt_neox.layers
        skip_rate: fraction of layers to skip (0.0 – 1.0)
    Returns:
        model with skipped layers patched in-place
    """
    layers = model.gpt_neox.layers
    num_layers = len(layers)
    num_skip = int(num_layers * skip_rate)
    # Skip the top layers (least specialized / most redundant)
    skip_set = set(range(num_layers - num_skip, num_layers))
    for i in skip_set:
        layers[i] = IdentityLayer(layers[i])
    return model, skip_set
