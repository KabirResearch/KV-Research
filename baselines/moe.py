"""
Mixture of Experts (MoE) Baseline
====================================
Replaces each dense FFN block with a set of expert FFNs.
A gating network routes each token to the top-k experts.
Compute cost scales with top_k/num_experts ratio.

Pseudocode:
    gate_logits = gate(hidden_states)           # [batch, seq, num_experts]
    top_k_weights, top_k_idx = topk(gate_logits, k=top_k)
    weights = softmax(top_k_weights)
    output = sum_k(weights_k * expert_k(hidden_states))

Paper: Switch Transformers — Fedus et al. 2022
       https://arxiv.org/abs/2101.03961
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEFFN(nn.Module):
    """
    Sparse MoE FFN replacement.
    Replaces the dense FFN inside a transformer layer's MLP block.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # Each expert is a small FFN
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, intermediate_size),
                    nn.GELU(),
                    nn.Linear(intermediate_size, hidden_size),
                )
                for _ in range(num_experts)
            ]
        )
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, hidden_states):
        batch, seq, dim = hidden_states.shape
        x_flat = hidden_states.view(-1, dim)  # [batch*seq, dim]

        gate_logits = self.gate(x_flat)  # [batch*seq, num_experts]
        top_k_logits, top_k_idx = gate_logits.topk(self.top_k, dim=-1)
        weights = F.softmax(top_k_logits, dim=-1)  # [batch*seq, top_k]

        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_idx = top_k_idx[:, k]  # [batch*seq]
            w = weights[:, k : k + 1]  # [batch*seq, 1]
            # Route each token to its assigned expert (simple loop; vectorize for prod)
            for e in range(self.num_experts):
                mask = expert_idx == e
                if mask.any():
                    output[mask] += w[mask] * self.experts[e](x_flat[mask])

        return output.view(batch, seq, dim)


# NOTE: Plugging MoEFFN into a real HuggingFace model requires replacing the
# model's MLP sub-module. This varies by architecture.
# Example for GPT-NeoX-style models (placeholder):
def apply_moe(model, num_experts: int = 8, top_k: int = 2):
    """
    Replace MLP blocks in transformer layers with MoEFFN.
    NOTE: Architecture-specific — adjust `mlp` attribute name as needed.

    Args:
        model: HuggingFace causal LM with model.gpt_neox.layers
        num_experts: total number of experts
        top_k: experts activated per token
    Returns:
        model with MoE FFNs applied in-place
    """
    hidden_size = model.config.hidden_size
    # GPT-NeoX intermediate size is typically 4x hidden
    intermediate_size = getattr(model.config, "intermediate_size", hidden_size * 4)
    for layer in model.gpt_neox.layers:
        if hasattr(layer, "mlp"):
            layer.mlp = MoEFFN(hidden_size, intermediate_size, num_experts, top_k)
    return model
