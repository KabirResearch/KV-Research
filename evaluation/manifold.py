"""
Manifold Analysis — CKA and Cosine Similarity
===============================================
Compares internal representations of the base model vs the patched model
at each layer. This detects representational collapse caused by skipping.

CKA = 1.0 → identical representations (good)
CKA ≈ 0.0 → collapsed / uncorrelated representations (bad)

Formulae:
    HSIC(K, L) = trace(K_c @ L_c) / (n-1)^2
    CKA(X, Y) = HSIC(XX^T, YY^T) / sqrt(HSIC(XX^T, XX^T) * HSIC(YY^T, YY^T))

    CosSim(h_base, h_skip) = dot(h_base, h_skip) / (|h_base| * |h_skip|)
"""

import torch
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


def _center_kernel(K: torch.Tensor) -> torch.Tensor:
    """Center a kernel matrix: K_c = H @ K @ H where H = I - 1/n * 1*1^T"""
    n = K.shape[0]
    H = torch.eye(n, device=K.device) - (1.0 / n) * torch.ones(n, n, device=K.device)
    return H @ K @ H


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """
    Compute linear CKA between two representation matrices.

    Args:
        X: [n_samples, dim_x] float tensor
        Y: [n_samples, dim_y] float tensor
    Returns:
        CKA score in [0, 1]
    """
    X = X.float()
    Y = Y.float()
    K = X @ X.T
    L = Y @ Y.T
    K_c = _center_kernel(K)
    L_c = _center_kernel(L)
    hsic_kl = (K_c * L_c).sum()
    hsic_kk = (K_c * K_c).sum()
    hsic_ll = (L_c * L_c).sum()
    return (hsic_kl / (hsic_kk * hsic_ll).sqrt()).item()


def _extract_hidden_states(model, dataloader, device: str = "cuda", max_batches: int = 20):
    """
    Run model forward passes and collect hidden states per layer.

    Returns:
        list of [n_tokens, hidden_dim] tensors, one per layer
    """
    model.eval()
    all_hs = None

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            ids = batch["input_ids"].to(device)
            out = model(ids, output_hidden_states=True)
            hs = out.hidden_states  # tuple of [batch, seq, dim]
            # Flatten batch*seq for CKA
            if all_hs is None:
                all_hs = [h.view(-1, h.shape[-1]).cpu() for h in hs]
            else:
                all_hs = [torch.cat([all_hs[j], hs[j].view(-1, hs[j].shape[-1]).cpu()]) for j in range(len(hs))]

    return all_hs  # [num_layers+1] list of [n_tokens, dim]


def layer_cka_table(base_model, patched_model, dataloader, device: str = "cuda"):
    """
    Compute per-layer CKA between base and patched model hidden states.

    Returns:
        list of (layer_idx, cka_score) tuples
    """
    logger.info("Extracting hidden states from base model...")
    base_hs = _extract_hidden_states(base_model, dataloader, device)
    logger.info("Extracting hidden states from patched model...")
    patch_hs = _extract_hidden_states(patched_model, dataloader, device)

    results = []
    for i, (bh, ph) in enumerate(zip(base_hs, patch_hs)):
        score = linear_cka(bh, ph)
        results.append((i, score))
        logger.info(f"Layer {i}: CKA = {score:.4f}")
    return results


def layer_cosine_sim_table(base_model, patched_model, dataloader, device: str = "cuda"):
    """
    Compute mean per-layer cosine similarity between base and patched hidden states.

    Returns:
        list of (layer_idx, mean_cosine_sim) tuples
    """
    base_hs = _extract_hidden_states(base_model, dataloader, device)
    patch_hs = _extract_hidden_states(patched_model, dataloader, device)

    results = []
    for i, (bh, ph) in enumerate(zip(base_hs, patch_hs)):
        cos_sim = F.cosine_similarity(bh.float(), ph.float(), dim=-1).mean().item()
        results.append((i, cos_sim))
        logger.info(f"Layer {i}: Cosine Sim = {cos_sim:.4f}")
    return results
