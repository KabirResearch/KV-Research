"""
main.py — Orchestration entry point for SoftLayer experiments.

Usage:
  python main.py --mode full
  python main.py --mode static_25
  python main.py --mode static_50
  python main.py --mode random_skip
  python main.py --mode critic_train
  python main.py --mode critic_eval
  python main.py --mode critic_eval --skip-rate 0.4
  python main.py --mode baselines
  python main.py --mode zero_shot
"""
import argparse
import torch
import wandb

from utils.config import config
from utils.logging import setup_logging
from utils.model import load_model, device
from data import load_dataset_part
from data.dataset import load_dataset_part as load_dataset_masked
from training.train_critic import train_block_critic
from training.train_router import train_router
from evaluation.evaluate import run_full, run_skip, evaluate_goldilocks
from evaluation.flops import measure_flops_manual
from evaluation.latency import measure_latency
from evaluation.manifold import layer_cka_table
from evaluation.zero_shot import run_zero_shot, print_zero_shot_table
from evaluation.calibrate import calibrate_voc_thresholds
from baselines.static_skip import apply_static_skip
from baselines.random_skip import apply_random_skip
from models.critics import LogTemporalCritic
from models.router import SoftPlanningRouter
from logs.research_logger import log_event

logger = setup_logging()


def _make_loader(split=None):
    from torch.utils.data import DataLoader
    ds = load_dataset_masked(split=split)
    return DataLoader(ds, batch_size=config.get("batch_size", 1))


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
            "static_25", "static_50",
            "random_skip",
            "critic_train", "critic_eval",
            "baselines",
            "zero_shot",
            "flops",
            "latency",
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

    elif args.mode in ("static_25", "static_50"):
        rate = 0.25 if args.mode == "static_25" else 0.50
        _init_wandb(f"static_skip_{int(rate*100)}") if not args.no_wandb else None
        model, tokenizer = load_model()
        test_loader = _make_loader(config.get("dataset_split", "test[:1%]"))
        result = run_skip(model, test_loader, apply_static_skip, {"skip_rate": rate}, device=str(device))
        if not args.no_wandb:
            wandb.log(result)
            wandb.finish()
        log_event("eval_result", result)

    elif args.mode == "random_skip":
        _init_wandb("random_skip") if not args.no_wandb else None
        model, tokenizer = load_model()
        test_loader = _make_loader(config.get("dataset_split", "test[:1%]"))
        result = run_skip(model, test_loader, apply_random_skip, {"skip_rate": args.skip_rate}, device=str(device))
        if not args.no_wandb:
            wandb.log(result)
            wandb.finish()
        log_event("eval_result", result)

    elif args.mode == "critic_train":
        _init_wandb("critic_train") if not args.no_wandb else None
        model, tokenizer = load_model()
        train_loader = _make_loader(config.get("dataset_split", "test[:1%]"))
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
            wandb.log(result)
            wandb.finish()
        log_event("eval_result", result)

    elif args.mode == "baselines":
        from baselines.token_pruning import apply_token_pruning
        from baselines.early_exit import apply_early_exit
        results = []
        for name, fn, kwargs in [
            ("static_25",  apply_static_skip,  {"skip_rate": 0.25}),
            ("static_50",  apply_static_skip,  {"skip_rate": 0.50}),
            ("random_25",  apply_random_skip,  {"skip_rate": 0.25}),
            ("token_prune",apply_token_pruning, {"keep_rate": 0.75}),
            ("early_exit", apply_early_exit,   {"confidence_threshold": 0.9, "min_exit_layer": 8}),
        ]:
            _init_wandb(f"baseline_{name}") if not args.no_wandb else None
            model, tokenizer = load_model()
            test_loader = _make_loader(config.get("dataset_split", "test[:1%]"))
            r = run_skip(model, test_loader, fn, kwargs, device=str(device))
            r["method"] = name
            results.append(r)
            if not args.no_wandb:
                wandb.log(r)
                wandb.finish()
        for r in results:
            print(f"{r['method']:20s}  PPL={r['ppl']:.4f}  tput={r['throughput']:.1f}")
        log_event("baselines_results", {"results": results})

    elif args.mode == "zero_shot":
        result = run_zero_shot(model_name=config.get("model_name", "EleutherAI/pythia-1b"))
        print_zero_shot_table(result)
        log_event("zero_shot_result", result)

    elif args.mode == "flops":
        model, tokenizer = load_model()
        flops = measure_flops_manual(model, seq_len=config.get("max_length", 128))
        logger.info(f"Estimated GFLOPs: {flops:.2f}")
        log_event("flops", {"gflops": flops})

    elif args.mode == "latency":
        model, tokenizer = load_model()
        stats = measure_latency(model, seq_len=config.get("max_length", 128), device=str(device))
        logger.info(f"Latency p50={stats['p50_ms']:.2f}ms  p95={stats['p95_ms']:.2f}ms")
        log_event("latency", stats)

    elif args.mode == "cka":
        import os
        if not os.path.exists("critic.pth"):
            logger.error("critic.pth not found. Run critic_train first.")
            return
        model_full,  _ = load_model()
        model_skip,  _ = load_model()
        hidden_size = model_skip.config.hidden_size
        critic = LogTemporalCritic(in_dim=hidden_size).to(device)
        critic.load_state_dict(torch.load("critic.pth", map_location=device))
        for i in config.get("target_layers", [10, 12, 14]):
            orig = model_skip.gpt_neox.layers[i]
            model_skip.gpt_neox.layers[i] = SoftPlanningRouter(orig, critic, skip_rate=args.skip_rate)
        val_loader = _make_loader(config.get("val_dataset_split", "validation[:1%]"))
        table = layer_cka_table(model_full, model_skip, val_loader, device=str(device))
        for row in table:
            print(f"Layer {row['layer']:2d}  CKA={row['cka']:.4f}  CosSim={row['cos_sim']:.4f}")
        log_event("cka_result", {"table": table})


if __name__ == "__main__":
    main()
