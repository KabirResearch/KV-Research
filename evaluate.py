import time
from utils import load_model, compute_loss, compute_entropy
from data import load_dataset_part
from models import apply_skip, apply_entropy_skip, apply_voc_skip
import json

with open('config.json') as f:
    config = json.load(f)

def evaluate_model(model, tokenizer, dataset):
    total_loss = 0
    count = 0
    start = time.time()
    for batch in dataset:
        input_ids = batch["input_ids"].unsqueeze(0).cuda()
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
        loss = compute_loss(logits, input_ids, tokenizer)
        total_loss += loss.item()
        count += 1
    elapsed = time.time() - start
    return total_loss / count, elapsed

def run_full():
    model, tokenizer = load_model()
    dataset = load_dataset_part()
    loss, time_sec = evaluate_model(model, tokenizer, dataset)
    print(f"Full: loss={loss:.4f}, time={time_sec:.2f}s")

def run_skip(skip_percent):
    model, tokenizer = load_model()
    dataset = load_dataset_part()
    total_layers = len(model.gpt_neox.layers)
    skip_layers = set(range(int(skip_percent * total_layers), total_layers))
    apply_skip(model, skip_layers)
    loss, time_sec = evaluate_model(model, tokenizer, dataset)
    print(f"Skip {skip_percent}: loss={loss:.4f}, time={time_sec:.2f}s")

def run_entropy(threshold):
    model, tokenizer = load_model()
    dataset = load_dataset_part()
    entropy_config = {
        "threshold": threshold,
        "min_layers": config['entropy_min_layers'],
        "current_entropy": 0.0,
        "skipped": 0,
        "enable_skip": True
    }
    apply_entropy_skip(model, entropy_config)
    total_loss = 0
    total_skips = 0
    count = 0
    start = time.time()
    for batch in dataset:
        input_ids = batch["input_ids"].unsqueeze(0).cuda()
        with torch.no_grad():
            # PASS 1: no skipping
            entropy_config["enable_skip"] = False
            outputs = model(input_ids)
            logits = outputs.logits
            entropy = compute_entropy(logits)
            entropy_config["current_entropy"] = entropy.item()
            # PASS 2: with skipping
            entropy_config["enable_skip"] = True
            entropy_config["skipped"] = 0
            outputs = model(input_ids)
            logits = outputs.logits
            loss = compute_loss(logits, input_ids, tokenizer)
            total_loss += loss.item()
            total_skips += entropy_config["skipped"]
            count += 1
    elapsed = time.time() - start
    loss = total_loss / count
    avg_skips = total_skips / count
    print(f"Entropy t={threshold}: loss={loss:.4f}, time={elapsed:.2f}s, skips={avg_skips:.2f}")

def run_voc(threshold, router):
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
    apply_voc_skip(model, voc_config, router)
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
    avg_skips = total_skips / count
    print(f"VoC t={threshold}: loss={loss:.4f}, time={elapsed:.2f}s, skips={avg_skips:.2f}")