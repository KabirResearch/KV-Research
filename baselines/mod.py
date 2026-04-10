"""
Mixture of Depths (MoD) Baseline
===================================
Each transformer layer processes only a fixed capacity (top-c tokens by router score).
Remaining tokens follow the residual path (identity).
This is architecturally similar to SoftLayer but uses a learned capacity router
rather than an attention-supervised critic.

Pseudocode:
    capacity = int(seq * capacity_factor)
    router_logits = router(hidden_states)      # [batch, seq]
    top_indices = topk(router_logits, capacity)
    selected = gather(hidden_states, top_indices)
    processed = layer(selected)
    output = scatter(processed, top_indices, base=hidden_states)  # residual for rest

Paper: Mixture of Depths — Raposo et al. 2024
       https://arxiv.org/abs/2404.02258
"""
import torch
import torch.nn as nn


class MoDRouter(nn.Module):
    """Lightweight token router: projects hidden dim to scalar score."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states):
        return self.proj(hidden_states).squeeze(-1)  # [batch, seq]


class MoDLayer(nn.Module):
    """Wraps a transformer layer with Mixture of Depths routing."""

    def __init__(self, layer, hidden_size: int, capacity_factor: float = 0.5):
        super().__init__()
        self.layer = layer
        self.router = MoDRouter(hidden_size)
        self.capacity_factor = capacity_factor

    def forward(self, hidden_states, *args, **kwargs):
        batch, seq, dim = hidden_states.shape
        capacity = max(1, int(seq * self.capacity_factor))

        router_logits = self.router(hidden_states)  # [batch, seq]
        top_indices = router_logits.topk(capacity, dim=-1).indices  # [batch, capacity]
        top_indices_sorted, _ = top_indices.sort(dim=-1)

        idx_exp = top_indices_sorted.unsqueeze(-1).expand(-1, -1, dim)
        selected = hidden_states.gather(1, idx_exp)  # [batch, capacity, dim]

        # Process only selected tokens
        out = self.layer(selected, *args, **kwargs)
        out_h = out[0] if isinstance(out, tuple) else out

        # Residual: unselected tokens pass through unchanged
        output = hidden_states.clone()
        output.scatter_(1, idx_exp, out_h)

        if isinstance(out, tuple):
            return (output,) + out[1:]
        return output


def apply_mod(model, capacity_factor: float = 0.5):
    """
    Wrap all transformer layers with MoDLayer.

    Args:
        model: HuggingFace causal LM with model.gpt_neox.layers
        capacity_factor: fraction of tokens processed per layer
    Returns:
        model with MoD applied in-place
    """
    hidden_size = model.config.hidden_size
    for i, layer in enumerate(model.gpt_neox.layers):
        model.gpt_neox.layers[i] = MoDLayer(layer, hidden_size, capacity_factor)
    return model
