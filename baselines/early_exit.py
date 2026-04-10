"""
Early Exit Transformer Baseline
=================================
Attaches a lightweight exit head at each layer (from min_exit_layer onward).
If the head's predicted confidence exceeds a threshold, inference exits early.

Pseudocode:
    for each layer i >= min_exit_layer:
        h = layer(h)
        logits = exit_head[i](h)
        confidence = softmax(logits).max().mean_over_batch()
        if confidence > threshold:
            return logits, exit_layer=i
    return lm_head(h), exit_layer=L

Paper: BERxiT — Xin et al. 2021
       https://arxiv.org/abs/2109.15148
"""
import torch
import torch.nn as nn
from utils.config import config


class EarlyExitHead(nn.Module):
    """Lightweight exit classifier: LayerNorm → Linear → LM head projection."""
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden_states):
        return self.head(self.norm(hidden_states))


class EarlyExitModel(nn.Module):
    """
    Wraps a HuggingFace model and adds early exit heads at each layer
    from min_exit_layer onward.
    """
    def __init__(self, model, confidence_threshold: float = 0.9, min_exit_layer: int = 8):
        super().__init__()
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.min_exit_layer = min_exit_layer

        hidden_size = model.config.hidden_size
        vocab_size = model.config.vocab_size
        num_layers = len(model.gpt_neox.layers)

        self.exit_heads = nn.ModuleList([
            EarlyExitHead(hidden_size, vocab_size) if i >= min_exit_layer else None
            for i in range(num_layers)
        ])
        self.exit_layer_used = None

    def forward(self, input_ids, **kwargs):
        # Manual forward through layers
        h = self.model.gpt_neox.embed_in(input_ids)
        for i, layer in enumerate(self.model.gpt_neox.layers):
            h = layer(h)[0]
            if i >= self.min_exit_layer:
                logits = self.exit_heads[i](h)
                probs = torch.softmax(logits.float(), dim=-1)
                confidence = probs.max(dim=-1).values.mean()
                if confidence > self.confidence_threshold:
                    self.exit_layer_used = i
                    return type('Output', (), {'logits': logits})()
        self.exit_layer_used = len(self.model.gpt_neox.layers) - 1
        logits = self.model.embed_out(self.model.gpt_neox.final_layer_norm(h))
        return type('Output', (), {'logits': logits})()


def apply_early_exit(model, confidence_threshold: float = 0.9, min_exit_layer: int = 8):
    """
    Wrap model with EarlyExitModel.

    Args:
        model: HuggingFace causal LM
        confidence_threshold: confidence >= this exits early
        min_exit_layer: earliest layer allowed to exit
    Returns:
        EarlyExitModel instance
    """
    return EarlyExitModel(model, confidence_threshold, min_exit_layer)
