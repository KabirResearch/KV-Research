"""
main.py — Orchestration entry point for SoftLayer experiments.

Usage:
  python main.py --mode full
  python main.py --mode static_25
  python main.py --mode static_50
  python main.py --mode random_skip
  python main.py --mode token_pruning
  python main.py --mode early_exit
  python main.py --mode moe
  python main.py --mode mod
  python main.py --mode baselines
  python main.py --mode critic_train
  python main.py --mode critic_eval
  python main.py --mode critic_eval --skip-rate 0.25
  python main.py --mode zero_shot
  python main.py --mode zero_shot_skip
  python main.py --mode zero_shot_skip --skip-rate 0.25
  python main.py --mode cka
  python main.py --mode flops
  python main.py --mode latency
  python main.py --model EleutherAI/pythia-2.8b --mode full
"""

import argparse
import os
import torch
import wandb

from utils.config import config
from utils.logging import setup_logging
from utils.model import load_model, device
from data.dataset import load_dataset_part as load_dataset_masked
from training.train_critic import train_block_critic
from evaluation.evaluate import run_full, run_skip, evaluate_goldilocks
from evaluation.manifold import layer_cka_table, layer_cosine_sim_table
from evaluation.zero_shot import run_zero_shot, print_zero_shot_table
from evaluation.flops import measure_flops_manual
from evaluation.latency import measure_latency
from baselines.mod import apply_mod
from baselines.static_skip import apply_static_skip
from baselines.random_skip import apply_random_skip
from baselines.token_pruning import apply_token_pruning
from baselines.early_exit import apply_early_exit
from baselines.moe import apply_moe
from models.critics import LogTemporalCritic
from models.router import SoftPlanningRouter
from logs.research_logger import log_event

logger = setup_logging()


def _make_loader(split=None, batch_size=None, drop_last: bool = False):
    from torch.utils.data import DataLoader

    ds = load_dataset_masked(split=split)
    return DataLoader(ds, batch_size=batch_size or config.get("batch_size", 1), drop_last=drop_last)


def _init_wandb(run_name: str):
    wandb.init(
        project=config.get("wandb_project", "softlayer"),
        name=run_name,
        config=config,
        reinit=True,
    )


def _load_critic(model_override=None):
    """Load critic.pth; raises if not found. Returns (model, critic)."""
    if not os.path.exists("critic.pth"):
        logger.error("critic.pth not found. Run: python main.py --mode critic_train")
        return None, None
    model, tokenizer = load_model(model_name=model_override)
    hidden_size = model.config.hidden_size
    critic = LogTemporalCritic(in_dim=hidden_size).to(device)
    critic.load_state_dict(torch.load("critic.pth", map_location=device))
    return model, critic


def _patch_router(model, critic, skip_rate):
    """Wrap target layers with SoftPlanningRouter in-place."""
    target_layers = config.get("target_layers", [10, 12, 14])
    for i in target_layers:
        orig = model.gpt_neox.layers[i]
        model.gpt_neox.layers[i] = SoftPlanningRouter(orig, critic, skip_rate=skip_rate)


def _log_ppl_tput(result, run_name):
    if not wandb.run:
        return
    wandb.log({"ppl": result["ppl"], "throughput": result["throughput"]})
    wandb.run.summary.update({"ppl": result["ppl"], "throughput": result["throughput"]})


def main():
    parser = argparse.ArgumentParser(description="SoftLayer layer-skipping experiments")
    parser.add_argument(
        "--mode",
        choices=[
            "full",
            "static_25",
            "static_50",
            "random_skip",
            "token_pruning",
            "early_exit",
            "moe",
            "mod",
            "baselines",
            "critic_train",
            "critic_eval",
            "zero_shot",
            "zero_shot_skip",
            "cka",
            "flops",
            "latency",
        ],
        default="full",
    )
    parser.add_argument("--skip-rate", type=float, default=0.5, help="Fraction of tokens/layers to skip")
    parser.add_argument("--epochs", type=int, default=config.get("epochs", 3))
    parser.add_argument("--model", type=str, default=None, help="Override model_name from config")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    # Allow --model to override config at runtime
    if args.model:
        config["model_name"] = args.model

    log_event("run_start", {"mode": args.mode, "skip_rate": args.skip_rate, "model": config.get("model_name")})

    # ── Full model reference ──────────────────────────────────────────────────
    if args.mode == "full":
        _init_wandb("full_model") if not args.no_wandb else None
        model, tokenizer = load_model()
        test_loader = _make_loader(config.get("dataset_split", "test"))
        result = run_full(model, test_loader, device=str(device))
        if not args.no_wandb:
            wandb.log(result)
            wandb.finish()
        log_event("eval_result", result)

    # ── Static skip baselines ─────────────────────────────────────────────────
    elif args.mode == "static_25":
        _init_wandb("static_skip_25") if not args.no_wandb else None
        model, _ = load_model()
        test_loader = _make_loader(config.get("dataset_split", "test"))
        r = run_skip(model, test_loader, apply_static_skip, {"skip_rate": 0.25}, device=str(device))
        r["method"] = "static_25"
        if not args.no_wandb:
            _log_ppl_tput(r, "static_skip_25")
            wandb.finish()
        log_event("eval_result", r)

    elif args.mode == "static_50":
        _init_wandb("static_skip_50") if not args.no_wandb else None
        model, _ = load_model()
        test_loader = _make_loader(config.get("dataset_split", "test"))
        r = run_skip(model, test_loader, apply_static_skip, {"skip_rate": 0.50}, device=str(device))
        r["method"] = "static_50"
        if not args.no_wandb:
            _log_ppl_tput(r, "static_skip_50")
            wandb.finish()
        log_event("eval_result", r)

    # ── Random skip control ───────────────────────────────────────────────────
    elif args.mode == "random_skip":
        _init_wandb("random_skip") if not args.no_wandb else None
        model, _ = load_model()
        test_loader = _make_loader(config.get("dataset_split", "test"))
        r = run_skip(model, test_loader, apply_random_skip, {"skip_rate": 0.25}, device=str(device))
        r["method"] = "random_skip"
        if not args.no_wandb:
            _log_ppl_tput(r, "random_skip")
            wandb.finish()
        log_event("eval_result", r)

    # ── Token pruning baseline ────────────────────────────────────────────────
    elif args.mode == "token_pruning":
        _init_wandb("baseline_token_prune") if not args.no_wandb else None
        model, _ = load_model()
        test_loader = _make_loader(config.get("dataset_split", "test"))
        r = run_skip(model, test_loader, apply_token_pruning, {"keep_rate": 0.75}, device=str(device))
        r["method"] = "token_pruning"
        if not args.no_wandb:
            _log_ppl_tput(r, "baseline_token_prune")
            wandb.finish()
        log_event("eval_result", r)

    # ── Early exit baseline ───────────────────────────────────────────────────
    elif args.mode == "early_exit":
        _init_wandb("baseline_early_exit") if not args.no_wandb else None
        model, _ = load_model()
        test_loader = _make_loader(config.get("dataset_split", "test"))
        r = run_skip(model, test_loader, apply_early_exit, {"confidence_threshold": 0.9}, device=str(device))
        r["method"] = "early_exit"
        if not args.no_wandb:
            _log_ppl_tput(r, "baseline_early_exit")
            wandb.finish()
        log_event("eval_result", r)

    # ── MoE baseline ──────────────────────────────────────────────────────────
    elif args.mode == "moe":
        _init_wandb("baseline_moe") if not args.no_wandb else None
        model, _ = load_model()
        test_loader = _make_loader(config.get("dataset_split", "test"))
        r = run_skip(model, test_loader, apply_moe, {"num_experts": 8, "top_k": 2}, device=str(device))
        r["method"] = "moe"
        if not args.no_wandb:
            _log_ppl_tput(r, "baseline_moe")
            wandb.finish()
        log_event("eval_result", r)

    # ── Mixture of Depths baseline ────────────────────────────────────────────
    elif args.mode == "mod":
        _init_wandb("baseline_mod") if not args.no_wandb else None
        model, _ = load_model()
        test_loader = _make_loader(config.get("dataset_split", "test"))
        r = run_skip(model, test_loader, apply_mod, {"capacity_factor": 0.5}, device=str(device))
        r["method"] = "mod"
        if not args.no_wandb:
            _log_ppl_tput(r, "baseline_mod")
            wandb.finish()
        log_event("eval_result", r)

    # ── All baselines aggregate ───────────────────────────────────────────────
    elif args.mode == "baselines":
        import subprocess
        import sys

        for sub_mode in ["static_25", "static_50", "random_skip", "token_pruning", "moe", "mod"]:
            cmd = [sys.executable, "main.py", "--mode", sub_mode]
            if args.no_wandb:
                cmd.append("--no-wandb")
            if args.model:
                cmd += ["--model", args.model]
            logger.info(f"Running baseline: {sub_mode}")
            ret = subprocess.run(cmd)
            if ret.returncode != 0:
                logger.warning(f"Baseline {sub_mode} exited with code {ret.returncode}")

    # ── Critic training ───────────────────────────────────────────────────────
    elif args.mode == "critic_train":
        _init_wandb("critic_train") if not args.no_wandb else None
        model, _ = load_model()
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        critic_batch_size = max(config.get("batch_size", 1), num_gpus)
        train_loader = _make_loader(
            config.get("val_dataset_split", "validation"),
            batch_size=critic_batch_size,
            drop_last=torch.cuda.is_available() and num_gpus > 1,
        )
        critic = train_block_critic(model, train_loader, epochs=args.epochs, device=str(device))
        torch.save(critic.state_dict(), "critic.pth")
        logger.info("Saved critic.pth")
        if not args.no_wandb:
            wandb.save("critic.pth")
            wandb.finish()
        log_event("model_saved", {"path": "critic.pth"})

    # ── Critic eval (PPL + throughput) ────────────────────────────────────────
    elif args.mode == "critic_eval":
        model, critic = _load_critic(model_override=args.model)
        if model is None:
            return
        _init_wandb(f"critic_eval_skip{int(args.skip_rate*100)}") if not args.no_wandb else None
        _patch_router(model, critic, args.skip_rate)
        test_loader = _make_loader(config.get("dataset_split", "test"))
        ppl, tput = evaluate_goldilocks(model, test_loader, device=str(device))
        result = {"method": f"softlayer_skip{int(args.skip_rate*100)}", "ppl": ppl, "throughput": tput}
        logger.info(f"SoftLayer skip{int(args.skip_rate*100)}: PPL={ppl:.4f}, throughput={tput:.1f} tok/s")
        if not args.no_wandb:
            wandb.log({"ppl": ppl, "throughput": tput})
            wandb.run.summary.update({"ppl": ppl, "throughput": tput})
            wandb.finish()
        log_event("eval_result", result)

    # ── Zero-shot: base model ─────────────────────────────────────────────────
    elif args.mode == "zero_shot":
        _init_wandb("zero_shot") if not args.no_wandb else None
        model, tokenizer = load_model()
        result = run_zero_shot(model, tokenizer, device=str(device))
        print_zero_shot_table(result)
        if not args.no_wandb:
            flat = {
                f"zero_shot/{task}": res.get("acc,none", res.get("acc_norm,none", 0)) for task, res in result.items()
            }
            wandb.log(flat)
            wandb.run.summary.update(flat)
            wandb.finish()
        log_event("zero_shot_result", result)

    # ── Zero-shot: SoftLayer-patched model ────────────────────────────────────
    elif args.mode == "zero_shot_skip":
        model, critic = _load_critic(model_override=args.model)
        if model is None:
            return
        _, tokenizer = load_model()
        _init_wandb(f"zero_shot_skip{int(args.skip_rate * 100)}") if not args.no_wandb else None
        _patch_router(model, critic, args.skip_rate)
        logger.info(f"Zero-shot eval: SoftLayer skip_rate={args.skip_rate}")
        result = run_zero_shot(model, tokenizer, device=str(device))
        print_zero_shot_table(result)
        if not args.no_wandb:
            flat = {
                f"zero_shot_skip/{task}": res.get("acc,none", res.get("acc_norm,none", 0))
                for task, res in result.items()
            }
            wandb.log(flat)
            wandb.run.summary.update(flat)
            wandb.finish()
        log_event("zero_shot_skip_result", result)

    # ── CKA / representation analysis ────────────────────────────────────────
    elif args.mode == "cka":
        model, critic = _load_critic(model_override=args.model)
        if model is None:
            return
        model_full, _ = load_model()
        _patch_router(model, critic, args.skip_rate)
        _init_wandb("cka_analysis") if not args.no_wandb else None
        val_loader = _make_loader(config.get("val_dataset_split", "validation"))
        cka_scores = layer_cka_table(model_full, model, val_loader, device=str(device))
        cos_scores = layer_cosine_sim_table(model_full, model, val_loader, device=str(device))
        combined = []
        for (li, cka_val), (_, cos_val) in zip(cka_scores, cos_scores):
            print(f"Layer {li:2d}  CKA={cka_val:.4f}  CosSim={cos_val:.4f}")
            combined.append({"layer": li, "cka": cka_val, "cos_sim": cos_val})
        if not args.no_wandb:
            for row in combined:
                wandb.log({"layer": row["layer"], "cka": row["cka"], "cos_sim": row["cos_sim"]})
            wandb.finish()
        log_event("cka_result", {"table": combined})

    # ── FLOPs measurement ─────────────────────────────────────────────────────
    elif args.mode == "flops":
        _init_wandb("flops") if not args.no_wandb else None
        model_full, _ = load_model()
        seq_len = config.get("max_length", 128)
        gflops_full = measure_flops_manual(model_full, seq_len=seq_len, batch_size=1)
        logger.info(f"Full model GFLOPs: {gflops_full:.3f}")
        results = {"flops/full": gflops_full}

        for skip_rate, label in [(0.25, "skip25"), (0.50, "skip50")]:
            model_s, _ = load_model()
            apply_static_skip(model_s, skip_rate=skip_rate)
            gflops = measure_flops_manual(model_s, seq_len=seq_len, batch_size=1)
            pct = (1 - gflops / gflops_full) * 100
            logger.info(f"Static skip {int(skip_rate*100)}% GFLOPs: {gflops:.3f}  ({pct:.1f}% reduction)")
            results[f"flops/{label}"] = gflops
            results[f"flops/{label}_reduction_pct"] = pct

        if not args.no_wandb:
            wandb.log(results)
            wandb.run.summary.update(results)
            wandb.finish()
        log_event("flops_result", results)

    # ── Latency measurement ───────────────────────────────────────────────────
    elif args.mode == "latency":
        if not torch.cuda.is_available():
            logger.error("Latency measurement requires CUDA. Skipping.")
            return
        _init_wandb("latency") if not args.no_wandb else None
        seq_len = config.get("max_length", 128)
        input_ids = torch.randint(0, 1000, (1, seq_len), device=device)
        results = {}

        model_full, _ = load_model()
        ms_full = measure_latency(model_full, input_ids)
        logger.info(f"Full model latency: {ms_full:.2f} ms")
        results["latency_ms/full"] = ms_full

        for skip_rate, label in [(0.25, "skip25"), (0.50, "skip50")]:
            model_s, _ = load_model()
            apply_static_skip(model_s, skip_rate=skip_rate)
            ms = measure_latency(model_s, input_ids)
            speedup = ms_full / ms
            logger.info(f"Static skip {int(skip_rate*100)}% latency: {ms:.2f} ms  ({speedup:.2f}x speedup)")
            results[f"latency_ms/{label}"] = ms
            results[f"latency_ms/{label}_speedup"] = speedup

        if not args.no_wandb:
            wandb.log(results)
            wandb.run.summary.update(results)
            wandb.finish()
        log_event("latency_result", results)


if __name__ == "__main__":
    main()
