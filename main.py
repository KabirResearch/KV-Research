import argparse
import json
import logging
import os
import sys
import torch

from utils import load_model
from evaluate import run_full, run_skip, run_entropy, run_voc
from train import train_voc, train_router
from models import Router, apply_voc_skip
from data import load_dataset_part

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load global config
try:
    with open('config.json') as f:
        config = json.load(f)
except FileNotFoundError:
    print("Error: config.json not found.")
    sys.exit(1)
except json.JSONDecodeError:
    print("Error: Invalid config.json.")
    sys.exit(1)


def run_voc_train():
    """Train VoC model and Router for layer skipping."""
    if os.path.exists('voc_model.pth') and os.path.exists('router.pth'):
        logger.info("Models already exist. Skipping training. Use --force to retrain.")
        return

    logger.info("Training VoC model...")
    try:
        voc_model = train_voc()
    except Exception as e:
        logger.error(f"Error training VoC model: {e}")
        return

    logger.info("Collecting data for Router training...")
    try:
        model, tokenizer = load_model()
        dataset = load_dataset_part()
    except Exception as e:
        logger.error(f"Error loading model or dataset: {e}")
        return

    router = Router().cuda().float()

    voc_config = {
        "threshold": 0.0,
        "skipped": 0,
        "prev_hidden": None,
        "collect_data": True,
        "records": [],
        "num_layers": len(model.gpt_neox.layers)
    }

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

    logger.info("Training Router...")
    try:
        router = train_router(router, records)
    except Exception as e:
        logger.error(f"Error training Router: {e}")
        return

    # Save trained models
    try:
        torch.save(voc_model.state_dict(), 'voc_model.pth')
        torch.save(router.state_dict(), 'router.pth')
        logger.info("Models saved: voc_model.pth, router.pth")
    except Exception as e:
        logger.error(f"Error saving models: {e}")


def run_voc_eval():
    """Evaluate VoC skipping with different thresholds."""
    if not os.path.exists('router.pth'):
        logger.error("router.pth not found. Run voc_train first.")
        return

    try:
        router = Router().cuda().float()
        router.load_state_dict(torch.load('router.pth'))
    except Exception as e:
        logger.error(f"Error loading router: {e}")
        return

    for threshold in config['voc_thresholds']:
        try:
            run_voc(threshold, router)
        except Exception as e:
            logger.error(f"Error evaluating with threshold {threshold}: {e}")


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
        """
    )
    parser.add_argument(
        '--mode',
        choices=['full', 'skip_25', 'skip_50', 'entropy', 'voc_train', 'voc_eval'],
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
        'voc_eval': run_voc_eval
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