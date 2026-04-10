"""
FLOPs Measurement
==================
Counts theoretical floating-point operations for a transformer forward pass.
Skipped layers contribute 0 FLOPs.

Formulae:
    Attention:  4 * B * S^2 * D + 4 * B * S * D^2
    FFN:        2 * B * S * D * D_ff   (where D_ff ≈ 4D)

Tool: fvcore (preferred) or manual counting.

Pseudocode:
    flops = FlopCountAnalysis(model, input_ids)
    gflops = flops.total() / 1e9
"""
import torch
import logging

logger = logging.getLogger(__name__)


def measure_flops_manual(model, seq_len: int, batch_size: int = 1) -> float:
    """
    Analytically estimate GFLOPs for a standard transformer forward pass.
    Accounts for layer skipping by checking for IdentityLayer wrappers.

    Args:
        model: HuggingFace causal LM (possibly with skip wrappers)
        seq_len: sequence length S
        batch_size: batch size B
    Returns:
        estimated GFLOPs (float)
    """
    cfg = model.config
    D = cfg.hidden_size
    D_ff = getattr(cfg, 'intermediate_size', D * 4)
    num_heads = cfg.num_attention_heads
    head_dim = D // num_heads
    B, S = batch_size, seq_len

    total_flops = 0
    for layer in model.gpt_neox.layers:
        # Check if layer is an identity skip (IdentityLayer or similar)
        if hasattr(layer, 'layer') and not hasattr(layer, 'router'):
            # Likely a static/random skip identity — 0 FLOPs
            continue
        # Attention: QKV projections + attention scores + output proj
        attn_flops = (
            3 * 2 * B * S * D * D +   # QKV projections
            2 * B * num_heads * S * S * head_dim +  # attn scores
            2 * B * S * D * D          # output projection
        )
        # FFN: two linear layers
        ffn_flops = 2 * 2 * B * S * D * D_ff
        total_flops += attn_flops + ffn_flops

    return total_flops / 1e9  # GFLOPs


def measure_flops_fvcore(model, input_ids) -> float:
    """
    Measure GFLOPs using fvcore's FlopCountAnalysis.
    Requires: pip install fvcore

    Args:
        model: HuggingFace causal LM
        input_ids: [batch, seq] tensor
    Returns:
        GFLOPs (float)
    """
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        logger.warning("fvcore not installed. Run: pip install fvcore")
        return measure_flops_manual(model, input_ids.shape[1], input_ids.shape[0])

    flops = FlopCountAnalysis(model, input_ids)
    flops.unsupported_ops_warnings(False)
    return flops.total() / 1e9
