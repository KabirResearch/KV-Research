"""
Core Evaluation
================
PPL, throughput, and full/skip evaluation functions.
All evaluations use -100 masked labels for correct PPL calculation.

Pseudocode:
    for batch in dataloader:
        out = model(input_ids, labels=labels)    # labels have -100 at padding
        active_tokens = (labels != -100).sum()
        nll += out.loss * active_tokens
        tokens += active_tokens
    ppl = exp(nll / tokens)
"""
import torch
import time
import logging
from utils.metrics import compute_loss

logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate_goldilocks(model, test_loader, device: str = "cuda"):
    """
    Evaluate PPL and throughput with correct -100 masked labels.

    Args:
        model: patched or base HuggingFace causal LM
        test_loader: DataLoader with 'input_ids' and 'labels' (-100 masked)
        device: torch device string
    Returns:
        (ppl: float, throughput_tokens_per_sec: float)
    """
    model.eval()
    nlls, total_tokens = [], 0
    start = time.time()

    for batch in test_loader:
        ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        out = model(ids, labels=labels)
        active = (labels != -100).sum().item()
        nlls.append(out.loss * active)
        total_tokens += active

    ppl = torch.exp(torch.stack(nlls).sum() / total_tokens).item()
    elapsed = time.time() - start
    throughput = total_tokens / elapsed
    return ppl, throughput


@torch.no_grad()
def run_full(model, test_loader, device: str = "cuda"):
    """Evaluate full model (no skipping) — baseline reference."""
    logger.info("Evaluating full model...")
    ppl, tput = evaluate_goldilocks(model, test_loader, device)
    logger.info(f"Full model: PPL={ppl:.4f}, throughput={tput:.1f} tok/s")
    return {"method": "full", "ppl": ppl, "throughput": tput}


@torch.no_grad()
def run_skip(model, test_loader, skip_fn, skip_kwargs: dict, device: str = "cuda"):
    """
    Evaluate a model with a given skip function applied.

    Args:
        model: base model
        test_loader: evaluation DataLoader
        skip_fn: function to apply skipping (e.g. apply_static_skip)
        skip_kwargs: keyword args for skip_fn
        device: torch device string
    Returns:
        dict with method, ppl, throughput
    """
    skip_fn(model, **skip_kwargs)
    ppl, tput = evaluate_goldilocks(model, test_loader, device)
    label = skip_fn.__name__
    logger.info(f"{label}: PPL={ppl:.4f}, throughput={tput:.1f} tok/s")
    return {"method": label, "ppl": ppl, "throughput": tput}
