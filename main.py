import argparse
import json
import torch

from utils import load_model
from evaluate import run_full, run_skip, run_entropy, run_voc
from train import train_voc, train_router
from models import Router, apply_voc_skip
from data import load_dataset_part

# Load global config
with open('config.json') as f:
    config = json.load(f)


def run_voc_train():
    """Train VoC model and Router for layer skipping."""
    print("Training VoC model...")
    voc_model = train_voc()

    print("Collecting data for Router training...")
    model, tokenizer = load_model()
    dataset = load_dataset_part()
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
    print(f"Collected {len(records)} records for training.")

    print("Training Router...")
    router = train_router(router, records)

    # Save trained router
    torch.save(router.state_dict(), 'router.pth')
    print("Router saved to router.pth")


def run_voc_eval():
    """Evaluate VoC skipping with different thresholds."""
    router = Router().cuda().float()
    router.load_state_dict(torch.load('router.pth'))

    for threshold in config['voc_thresholds']:
        run_voc(threshold, router)


def main():
    parser = argparse.ArgumentParser(description="Run layer skipping experiments.")
    parser.add_argument(
        '--mode',
        choices=['full', 'skip_25', 'skip_50', 'entropy', 'voc_train', 'voc_eval'],
        default='full',
        help="Execution mode"
    )
    args = parser.parse_args()

    if args.mode == 'full':
        run_full()
    elif args.mode == 'skip_25':
        run_skip(0.25)
    elif args.mode == 'skip_50':
        run_skip(0.5)
    elif args.mode == 'entropy':
        for threshold in config['entropy_thresholds']:
            run_entropy(threshold)
    elif args.mode == 'voc_train':
        run_voc_train()
    elif args.mode == 'voc_eval':
        run_voc_eval()


if __name__ == "__main__":
    main()