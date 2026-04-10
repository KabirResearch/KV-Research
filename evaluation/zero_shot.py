"""
Zero-Shot Downstream Evaluations
===================================
PPL alone is insufficient for evaluating architectural modifications.
We evaluate on: HellaSwag, PIQA, ARC-Easy, ARC-Challenge, WinoGrande.

Tool: EleutherAI lm-evaluation-harness
      pip install lm-eval

Pseudocode:
    lm = HFLMWrapper(patched_model, tokenizer)
    results = evaluator.simple_evaluate(model=lm, tasks=[...])
    return results['results']
"""

import logging

logger = logging.getLogger(__name__)

DEFAULT_TASKS = ["hellaswag", "piqa", "arc_easy", "arc_challenge", "winogrande"]


def run_zero_shot(model, tokenizer, tasks=None, batch_size: int = 8, device: str = "cuda"):
    """
    Run zero-shot evaluation using lm-evaluation-harness.
    Requires: pip install lm-eval

    Args:
        model: HuggingFace causal LM (patched or base)
        tokenizer: associated tokenizer
        tasks: list of task names (default: HellaSwag, PIQA, ARC, WinoGrande)
        batch_size: eval batch size
        device: torch device string
    Returns:
        dict: {task_name: {'acc': float, 'acc_norm': float, ...}}
    """
    try:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
    except ImportError:
        logger.error("lm-eval not installed. Run: pip install lm-eval")
        return {}

    if tasks is None:
        tasks = DEFAULT_TASKS

    logger.info(f"Running zero-shot evaluation on: {tasks}")

    lm = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
    )
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        log_samples=False,
    )
    for task, res in results["results"].items():
        acc = res.get("acc,none", res.get("acc_norm,none", "N/A"))
        logger.info(f"  {task}: {acc}")

    return results["results"]


def print_zero_shot_table(results: dict):
    """Pretty-print zero-shot results as a Markdown table."""
    print(f"\n{'Task':<20} | {'Acc':<8} | {'Acc Norm':<10}")
    print("-" * 45)
    for task, res in results.items():
        acc = res.get("acc,none", "-")
        acc_norm = res.get("acc_norm,none", "-")
        print(f"{task:<20} | {acc!s:<8} | {acc_norm!s:<10}")
