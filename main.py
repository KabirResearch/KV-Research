"""
main.py — Orchestration entry point for SoftLayer experiments.

Usage:
  python main.py --mode full
  python main.py --mode mod
  python main.py --mode critic_train
  python main.py --mode critic_eval
  python main.py --mode critic_eval --skip-rate 0.4
  python main.py --mode zero_shot
  python main.py --mode zero_shot_skip
  python main.py --mode cka
"""

import argparse
import torch
import wandb

from utils.config import config
from utils.logging import setup_logging
from utils.model import load_model, device
from data.dataset import load_dataset_part as load_dataset_masked
from training.train_critic import train_block_critic
from evaluation.evaluate import run_full, run_skip, evaluate_goldilocks
from evaluation.manifold import layer_cka_table
from evaluation.zero_shot import run_zero_shot, print_zero_shot_table
from evaluation.manifold import layer_cosine_sim_table
from baselines.mod import apply_mod
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


def main():
    parser = argparse.ArgumentParser(description="SoftLayer layer-skipping experiments")
    parser.add_argument(
        "--mode",
        choices=[
            "full",
            "mod",
            "critic_train",
            "critic_eval",
            "zero_shot",
            "zero_shot_skip",
            "cka",
        ],
        default="full",
    )
    parser.add_argument("--skip-rate", type=float, default=0.5, help="Fraction of tokens/layers to skip")
    parser.add_argument("--epochs", type=int, default=config.get("epochs", 3))
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    args = parser.parse_args()

    log_event("run_start", {"mode": args.mode, "skip_rate": args.skip_rate})

    if args.mode == "full":
        _init_wandb("full_model") if not args.no_wandb else None
        model, tokenizer = load_model()
        test_loader = _make_loader(config.get("dataset_split", "test[:1%]"))
        result = run_full(model, test_loader, device=str(device))
        if not args.no_wandb:
            wandb.log(result)
            wandb.finish()
        log_event("eval_result", result)

    elif args.mode == "mod":
        _init_wandb("baseline_mod") if not args.no_wandb else None
        model, tokenizer = load_model()
        test_loader = _make_loader(config.get("dataset_split", "test[:1%]"))
        r = run_skip(model, test_loader, apply_mod, {"capacity_factor": 0.5}, device=str(device))
        r["method"] = "mod"
        if not args.no_wandb:
            wandb.log({"ppl": r["ppl"], "throughput": r["throughput"]})
            wandb.run.summary.update({"ppl": r["ppl"], "throughput": r["throughput"]})
            wandb.finish()
        log_event("eval_result", r)

    elif args.mode == "critic_train":
        _init_wandb("critic_train") if not args.no_wandb else None
        model, tokenizer = load_model()
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        critic_batch_size = max(config.get("batch_size", 1), num_gpus)
        train_loader = _make_loader(
            config.get("dataset_split", "test[:1%]"),
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

    elif args.mode == "critic_eval":
        import os

        if not os.path.exists("critic.pth"):
            logger.error("critic.pth not found. Run: python main.py --mode critic_train")
            return
        _init_wandb(f"critic_eval_skip{int(args.skip_rate*100)}") if not args.no_wandb else None
        model, tokenizer = load_model()
        hidden_size = model.config.hidden_size
        critic = LogTemporalCritic(in_dim=hidden_size).to(device)
        critic.load_state_dict(torch.load("critic.pth", map_location=device))

        # Wrap each target layer with the SoftPlanningRouter
        target_layers = config.get("target_layers", [10, 12, 14])
        for i in target_layers:
            orig = model.gpt_neox.layers[i]
            model.gpt_neox.layers[i] = SoftPlanningRouter(orig, critic, skip_rate=args.skip_rate)

        test_loader = _make_loader(config.get("dataset_split", "test[:1%]"))
        ppl, tput = evaluate_goldilocks(model, test_loader, device=str(device))
        result = {"method": f"softlayer_skip{int(args.skip_rate*100)}", "ppl": ppl, "throughput": tput}
        logger.info(f"SoftLayer: PPL={ppl:.4f}, throughput={tput:.1f} tok/s")
        if not args.no_wandb:
            wandb.log({"ppl": ppl, "throughput": tput})
            wandb.run.summary.update({"ppl": ppl, "throughput": tput})
            wandb.finish()
        log_event("eval_result", result)

    elif args.mode == "cka":
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

    elif args.mode == "zero_shot_skip":
        import os

        if not os.path.exists("critic.pth"):
            logger.error("critic.pth not found. Run: python main.py --mode critic_train")
            return
        _init_wandb(f"zero_shot_skip{int(args.skip_rate * 100)}") if not args.no_wandb else None
        model, tokenizer = load_model()
        hidden_size = model.config.hidden_size
        critic = LogTemporalCritic(in_dim=hidden_size).to(device)
        critic.load_state_dict(torch.load("critic.pth", map_location=device))
        target_layers = config.get("target_layers", [10, 12, 14])
        for i in target_layers:
            orig = model.gpt_neox.layers[i]
            model.gpt_neox.layers[i] = SoftPlanningRouter(orig, critic, skip_rate=args.skip_rate)
        logger.info(f"Running zero-shot eval on SoftLayer-patched model (skip_rate={args.skip_rate})")
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

    elif args.mode == "cka":
        import os

        if not os.path.exists("critic.pth"):
            logger.error("critic.pth not found. Run critic_train first.")
            return
        model_full, _ = load_model()
        model_skip, _ = load_model()
        hidden_size = model_skip.config.hidden_size
        critic = LogTemporalCritic(in_dim=hidden_size).to(device)
        critic.load_state_dict(torch.load("critic.pth", map_location=device))
        for i in config.get("target_layers", [10, 12, 14]):
            orig = model_skip.gpt_neox.layers[i]
            model_skip.gpt_neox.layers[i] = SoftPlanningRouter(orig, critic, skip_rate=args.skip_rate)
        _init_wandb("cka_analysis") if not args.no_wandb else None
        val_loader = _make_loader(config.get("val_dataset_split", "validation[:1%]"))
        cka_table = layer_cka_table(model_full, model_skip, val_loader, device=str(device))
        cos_table = layer_cosine_sim_table(model_full, model_skip, val_loader, device=str(device))
        combined = []
        for (li, cka), (_, cos) in zip(cka_table, cos_table):
            print(f"Layer {li:2d}  CKA={cka:.4f}  CosSim={cos:.4f}")
            combined.append({"layer": li, "cka": cka, "cos_sim": cos})
        if not args.no_wandb:
            for row in combined:
                wandb.log({"layer": row["layer"], "cka": row["cka"], "cos_sim": row["cos_sim"]})
            wandb.finish()
        log_event("cka_result", {"table": combined})


if __name__ == "__main__":
    main()
