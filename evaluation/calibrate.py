"""
Threshold Calibration
======================
Calibrates the router/critic threshold on a held-out validation split
by sweeping candidate thresholds and selecting the one that minimises
PPL degradation subject to a target skip rate.

Pseudocode:
    for threshold in candidates:
        ppl = evaluate(model with threshold applied, val_loader)
        if ppl <= base_ppl * tolerance:
            best_threshold = threshold
"""

import logging
from evaluation.evaluate import evaluate_goldilocks

logger = logging.getLogger(__name__)


def calibrate_voc_thresholds(
    model,
    router,
    val_loader,
    apply_fn,
    candidates=None,
    base_ppl: float = None,
    tolerance: float = 1.05,
    device: str = "cuda",
):
    """
    Sweep threshold candidates and return the most aggressive threshold
    that keeps PPL within `tolerance * base_ppl`.

    Args:
        model: base HuggingFace causal LM
        router: trained router/critic
        val_loader: validation DataLoader
        apply_fn: function(model, voc_config, router) that patches the model
        candidates: list of threshold values to try
        base_ppl: reference full-model PPL; if None, computed fresh
        tolerance: max allowed PPL inflation (default 1.05 = 5% degradation)
        device: torch device string
    Returns:
        best_threshold: float
    """
    if candidates is None:
        candidates = [-1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0]

    if base_ppl is None:
        base_ppl, _ = evaluate_goldilocks(model, val_loader, device)
        logger.info(f"Base PPL (full model): {base_ppl:.4f}")

    best_threshold = candidates[0]
    for threshold in candidates:
        voc_config = {
            "threshold": threshold,
            "skipped": 0,
            "prev_hidden": None,
            "collect_data": False,
            "records": [],
            "num_layers": len(model.gpt_neox.layers),
        }
        apply_fn(model, voc_config, router)
        ppl, _ = evaluate_goldilocks(model, val_loader, device)
        logger.info(f"Threshold {threshold:.2f}: PPL={ppl:.4f}")
        if ppl <= base_ppl * tolerance:
            best_threshold = threshold

    logger.info(f"Best threshold: {best_threshold}")
    return best_threshold
