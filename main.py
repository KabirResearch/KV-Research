
import argparse
import os
import time
from utils.model import load_model
from utils.metrics import compute_loss
from utils.config import config
from utils.logging import setup_logging
from evaluate import run_full, run_skip, run_entropy, run_voc, calibrate_voc_thresholds
from train import train_voc, train_router
from models import Router, apply_voc_skip, apply_token_level_voc_skip
from data import load_dataset_part

# Configure logging
logger = setup_logging()


def _run_voc_train_common(per_token: bool, save_name: str):
    """Shared implementation for both regular and token-level VoC training."""
    model_path = f'{save_name}.pth'
    if os.path.exists('voc_model.pth') and os.path.exists(model_path):
        logger.info("Models already exist. Skipping training. Use --force to retrain.")
        return

    logger.info("Training VoC model...")
    try:
        voc_model = train_voc()
    except Exception as e:
        logger.error(f"Error training VoC model: {e}")
        return

    logger.info(f"Collecting data for {save_name} training...")
    try:
        model, tokenizer = load_model()
        dataset = load_dataset_part()
    except Exception as e:
        logger.error(f"Error loading model or dataset: {e}")
        return

    num_layers = len(model.gpt_neox.layers)
    router = Router(num_layers, per_token=per_token).cuda().float()

    voc_config = {
        "threshold": 0.0,
        "skipped": 0,
        "prev_hidden": None,
        "collect_data": True,
        "records": [],
        "num_layers": len(model.gpt_neox.layers)
    }

    if per_token:
        apply_token_level_voc_skip(model, voc_config, router)
    else:
        apply_voc_skip(model, voc_config, router)

    for batch in dataset:
        input_ids = batch["input_ids"].unsqueeze(0).cuda()
        with torch.no_grad():
            voc_config["prev_hidden"] = None
            _ = model(input_ids)

    records = voc_config["records"]
    logger.info(f"Collected {len(records)} records for training.")

    if not records:
        logger.warning("No records collected. Skipping Router training.")
        return

    logger.info(f"Training {save_name}...")
    try:
        router = train_router(router, records)
    except Exception as e:
        logger.error(f"Error training Router: {e}")
        return

    # Save trained models
    try:
        torch.save(voc_model.state_dict(), 'voc_model.pth')
        torch.save(router.state_dict(), model_path)
        logger.info(f"Models saved: voc_model.pth, {model_path}")
    except Exception as e:
        logger.error(f"Error saving models: {e}")


def run_voc_train():
    """Train VoC model and Router for layer skipping."""
    _run_voc_train_common(per_token=False, save_name='router')


def run_voc_token_train():
    """Train VoC model and Router for token-level skipping."""
    _run_voc_train_common(per_token=True, save_name='router_token')


def run_voc_eval():
    """Evaluate VoC skipping with different thresholds."""
    if not os.path.exists('router.pth'):
        logger.error("router.pth not found. Run voc_train first.")
        return

    try:
        num_layers = config.get('num_layers', 16)
        router = Router(num_layers).cuda().float()
        router.load_state_dict(torch.load('router.pth'))
    except Exception as e:
        logger.error(f"Error loading router: {e}")
        return

    for threshold in config['voc_thresholds']:
        try:
            run_voc(threshold, router)
        except Exception as e:
            logger.error(f"Error evaluating with threshold {threshold}: {e}")

def run_voc_token_eval():
    """Evaluate token-level VoC skipping."""
    if not os.path.exists('router_token.pth'):
        logger.error("router_token.pth not found. Run voc_token_train first.")
        return

    try:
        num_layers = config.get('num_layers', 16)
        router = Router(num_layers, per_token=True).cuda().float()
        router.load_state_dict(torch.load('router_token.pth'))
    except Exception as e:
        logger.error(f"Error loading router: {e}")
        return

    for threshold in config['voc_thresholds']:
        try:
            model, tokenizer = load_model()
            dataset = load_dataset_part()
            voc_config = {
                "threshold": threshold,
                "skipped": 0,
                "prev_hidden": None,
                "collect_data": False,
                "records": [],
                "num_layers": len(model.gpt_neox.layers)
            }
            apply_token_level_voc_skip(model, voc_config, router)
            total_loss, total_skips, count = 0, 0, 0
            start = time.time()
            for batch in dataset:
                input_ids = batch["input_ids"].unsqueeze(0).cuda()
                with torch.no_grad():
                    voc_config["skipped"] = 0
                    voc_config["prev_hidden"] = None
                    outputs = model(input_ids)
                    logits = outputs.logits
                loss = compute_loss(logits, input_ids, tokenizer)
                total_loss += loss.item()
                total_skips += voc_config["skipped"]
                count += 1
            elapsed = time.time() - start
            loss = total_loss / count
            avg_skips = total_skips / count / (batch["input_ids"].shape[1] * count)  # percent tokens skipped
            logger.info(f"Token VoC t={threshold}: loss={loss:.4f}, time={elapsed:.2f}s, %tokens skipped={avg_skips:.2f}")
            # Cleanup GPU memory
            del model, tokenizer, dataset, voc_config
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error evaluating with threshold {threshold}: {e}")

def run_voc_eval_calibrated():
    """Evaluate VoC skipping with calibrated threshold."""
    if not os.path.exists('router.pth'):
        logger.error("router.pth not found. Run voc_train first.")
        return

    try:
        # Assume num_layers from config or default
        num_layers = config.get('num_layers', 16)  # pythia-1b has 16 layers
        router = Router(num_layers).cuda().float()
        router.load_state_dict(torch.load('router.pth'))
    except Exception as e:
        logger.error(f"Error loading router: {e}")
        return

    try:
        val_dataset = load_dataset_part(split=config.get('val_dataset_split', 'validation[:1%]'))
        best_threshold = calibrate_voc_thresholds(router, val_dataset)
        run_voc(best_threshold, router)
    except Exception as e:
        logger.error(f"Error in calibrated evaluation: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run layer skipping experiments for GPT-NeoX models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
 Examples:
   python main.py --mode full          # Evaluate full model
   python main.py --mode skip_25       # Evaluate 25% static skip
   python main.py --mode entropy       # Evaluate entropy-based skipping
   python main.py --mode voc_train     # Train VoC and Router models
   python main.py --mode voc_eval      # Evaluate VoC skipping
   python main.py --mode voc_eval_calibrated  # Evaluate VoC with calibrated threshold
   python main.py --mode voc_token_train     # Train token-level VoC
   python main.py --mode voc_token_eval      # Evaluate token-level VoC
         """
    )
    parser.add_argument(
        '--mode',
        choices=['full', 'skip_25', 'skip_50', 'entropy', 'voc_train', 'voc_eval', 'voc_eval_calibrated', 'voc_token_train', 'voc_token_eval'],
        default='full',
        help="Execution mode (see examples above)"
    )
    args = parser.parse_args()

    mode_functions = {
        'full': run_full,
        'skip_25': lambda: run_skip(0.25),
        'skip_50': lambda: run_skip(0.5),
        'entropy': lambda: [run_entropy(t) for t in config['entropy_thresholds']],
        'voc_train': run_voc_train,
        'voc_eval': run_voc_eval,
        'voc_eval_calibrated': run_voc_eval_calibrated,
        'voc_token_train': run_voc_token_train,
        'voc_token_eval': run_voc_token_eval
    }

    try:
        result = mode_functions[args.mode]()
        if args.mode == 'entropy':
            pass  # Already handled
    except KeyError:
        logger.error(f"Unknown mode: {args.mode}")
    except Exception as e:
        logger.error(f"Error running mode {args.mode}: {e}")


if __name__ == "__main__":
    main()