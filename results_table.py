"""
results_table.py — Auto-fills the comparison table from W&B run summaries.

Run after all experiments are complete:
    python results_table.py

Prints a Markdown table and saves results_table.csv.
Requires: pip install wandb
"""

import json
import os
import csv

try:
    import wandb
except ImportError:
    print("wandb not installed. Run: pip install wandb")
    raise

# ── Config ──────────────────────────────────────────────────────────────
with open(os.environ.get("CONFIG_FILE", "config.json")) as f:
    _cfg = json.load(f)

PROJECT = _cfg.get("wandb_project", "softlayer")
ENTITY = _cfg.get("wandb_entity") or None  # None → uses default entity

# Maps W&B run name → row label in the table
RUN_NAME_TO_ROW = {
    "full_model": "Full Model",
    "static_skip_25": "Static Skip 25%",
    "random_skip": "Random Skip 25%",
    "baseline_token_prune": "Token Pruning",
    "baseline_mod": "Mixture of Depths (MoD)",
    "critic_eval_skip50": "SoftLayer 50% (ours)",
    "critic_eval_skip25": "SoftLayer 25% (ours)",
}

ZERO_SHOT_TASKS = ["hellaswag", "piqa", "arc_easy", "arc_challenge", "winogrande"]


def _fmt(val, decimals=4):
    if val is None:
        return "—"
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return str(val)


def fetch_runs():
    """Pull all finished runs from W&B project and return {run_name: summary}."""
    api = wandb.Api()
    path = f"{ENTITY}/{PROJECT}" if ENTITY else PROJECT
    runs = api.runs(path)
    data = {}
    for run in runs:
        data[run.name] = dict(run.summary)
        # Also attach history for CKA (logged row-by-row)
        if run.name == "cka_analysis":
            cka_rows = list(run.scan_history(keys=["layer", "cka", "cos_sim"]))
            data[run.name]["_cka_rows"] = cka_rows
    return data


def extract_zero_shot(summary, prefix):
    """Return {task: acc} from a run summary given key prefix (zero_shot/ or zero_shot_skip/)."""
    result = {}
    for task in ZERO_SHOT_TASKS:
        key = f"{prefix}/{task}"
        result[task] = summary.get(key)
    return result


def build_table(runs):
    rows = []

    for run_name, row_label in RUN_NAME_TO_ROW.items():
        s = runs.get(run_name, {})
        # zero-shot: base model rows use zero_shot/ prefix; skip models use same summary keys
        zs_base = extract_zero_shot(runs.get("zero_shot", {}), "zero_shot")
        zs_skip50 = extract_zero_shot(runs.get("zero_shot_skip50", {}), "zero_shot_skip")
        zs_skip25 = extract_zero_shot(runs.get("zero_shot_skip25", {}), "zero_shot_skip")

        # PPL and throughput
        ppl = s.get("ppl")
        tput = s.get("throughput")

        # CKA mean (for skip models)
        cka_mean = None
        cka_data = runs.get("cka_analysis", {}).get("_cka_rows", [])
        if cka_data:
            scores = [r["cka"] for r in cka_data if "cka" in r]
            if scores:
                cka_mean = sum(scores) / len(scores)

        # Pick the right zero-shot block for this row
        if row_label == "Full Model":
            zs = zs_base
        elif "50%" in row_label and "ours" in row_label:
            zs = zs_skip50
        elif "25%" in row_label and "ours" in row_label:
            zs = zs_skip25
        else:
            # baselines: we only have PPL, not zero-shot (not run through lm-eval)
            zs = {t: None for t in ZERO_SHOT_TASKS}

        rows.append(
            {
                "Method": row_label,
                "PPL ↓": _fmt(ppl),
                "Tput tok/s ↑": _fmt(tput, 1),
                "HellaSwag ↑": _fmt(zs.get("hellaswag"), 4),
                "PIQA ↑": _fmt(zs.get("piqa"), 4),
                "ARC-Easy ↑": _fmt(zs.get("arc_easy"), 4),
                "ARC-Chal ↑": _fmt(zs.get("arc_challenge"), 4),
                "WinoGrande ↑": _fmt(zs.get("winogrande"), 4),
                "CKA Sim": _fmt(cka_mean) if ("ours" in row_label) else ("1.0" if row_label == "Full Model" else "—"),
            }
        )

    return rows


def print_markdown(rows):
    cols = list(rows[0].keys())
    header = " | ".join(cols)
    sep = " | ".join(["---"] * len(cols))
    print(f"\n| {header} |")
    print(f"| {sep} |")
    for row in rows:
        print("| " + " | ".join(row[c] for c in cols) + " |")


def save_csv(rows, path="results_table.csv"):
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    print(f"Fetching runs from W&B project: {PROJECT}")
    runs = fetch_runs()
    print(f"Found {len(runs)} runs: {list(runs.keys())}")
    rows = build_table(runs)
    print_markdown(rows)
    save_csv(rows)
