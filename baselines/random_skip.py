"""
Random Layer Skipping Baseline (Control)
==========================================
Skips a randomly selected subset of layers.
Used as a noise-controlled comparison — if SoftLayer doesn't
beat random skipping, the critic is not doing useful work.

Pseudocode:
    skip_set = random.sample(all_layer_indices, num_skip)
    for i in skip_set:
        layer[i] = IdentityLayer
"""
import random
import torch.nn as nn
from baselines.static_skip import IdentityLayer


def apply_random_skip(model, skip_rate: float = 0.25, seed: int = 42):
    """
    Wrap a randomly selected skip_rate fraction of layers with IdentityLayer.

    Args:
        model: HuggingFace causal LM with model.gpt_neox.layers
        skip_rate: fraction of layers to skip
        seed: random seed for reproducibility
    Returns:
        (model, skip_set)
    """
    random.seed(seed)
    layers = model.gpt_neox.layers
    num_layers = len(layers)
    num_skip = int(num_layers * skip_rate)
    skip_set = set(random.sample(range(num_layers), num_skip))
    for i in skip_set:
        layers[i] = IdentityLayer(layers[i])
    return model, skip_set
