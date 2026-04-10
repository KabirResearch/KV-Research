"""
Dynamic Token Pruning Baseline
================================
Prunes low-importance tokens from the sequence before each layer.
Unpruned tokens use identity (pass-through). Inspired by ToMe / SpAtten.

Pseudocode:
    scores = norm(hidden_states, dim=-1)      # token importance proxy
    top_k_indices = topk(scores, k=keep_k)
    pruned = gather(hidden_states, top_k_indices)
    out = layer(pruned)
    full_out = scatter(out, top_k_indices) + identity for rest

Paper: Token Merging (ToMe) — Bolya et al. 2022
       https://arxiv.org/abs/2210.09461
"""

import torch.nn as nn


class TokenPruningLayer(nn.Module):
    """Wraps a transformer layer with dynamic token pruning."""

    def __init__(self, layer, keep_rate: float = 0.8):
        super().__init__()
        self.layer = layer
        self.keep_rate = keep_rate

    def forward(self, hidden_states, *args, **kwargs):
        batch, seq, dim = hidden_states.shape
        keep_k = max(1, int(seq * self.keep_rate))

        # Score each token by its L2 norm (importance proxy)
        scores = hidden_states.norm(dim=-1)  # [batch, seq]
        top_indices = scores.topk(keep_k, dim=-1).indices  # [batch, keep_k]
        top_indices_sorted, _ = top_indices.sort(dim=-1)

        # Gather selected tokens
        idx_exp = top_indices_sorted.unsqueeze(-1).expand(-1, -1, dim)
        pruned = hidden_states.gather(1, idx_exp)  # [batch, keep_k, dim]

        # Process pruned tokens
        # NOTE: positional args/kwargs may need adjustment for specific models
        out = self.layer(pruned, *args, **kwargs)
        out_h = out[0] if isinstance(out, tuple) else out

        # Scatter back; unselected tokens keep identity
        full_out = hidden_states.clone()
        full_out.scatter_(1, idx_exp, out_h)

        if isinstance(out, tuple):
            return (full_out,) + out[1:]
        return full_out


def apply_token_pruning(model, keep_rate: float = 0.8):
    """
    Wrap all transformer layers with TokenPruningLayer.

    Args:
        model: HuggingFace causal LM with model.gpt_neox.layers
        keep_rate: fraction of tokens to keep per layer
    Returns:
        model with token pruning applied in-place
    """
    for i, layer in enumerate(model.gpt_neox.layers):
        model.gpt_neox.layers[i] = TokenPruningLayer(layer, keep_rate=keep_rate)
    return model
