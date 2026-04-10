import torch
import pytest
from utils.model import load_model
from utils.metrics import compute_loss
from models import apply_skip, apply_entropy_skip, apply_voc_skip, Router

def make_dummy_input(tokenizer, seq_len=16):
    vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 1000
    return torch.randint(0, vocab_size, (1, seq_len)).to(next(tokenizer.model_input_names, 'input_ids'))

def test_v2_skip():
    model, tokenizer = load_model()
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, 16)).to(model.device)
    skip_layers = set(range(0, len(model.gpt_neox.layers), 2))
    apply_skip(model, skip_layers)
    with torch.no_grad():
        output = model(input_ids)
    if isinstance(output, tuple):
        output = output[0]
    assert output.shape[0] == 1
    assert output.shape[1] == 16


def test_v3_entropy_skip():
    model, tokenizer = load_model()
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, 16)).to(model.device)
    config = {"enable_skip": True, "min_layers": 0, "current_entropy": 0.0, "threshold": 1e9, "skipped": 0}
    apply_entropy_skip(model, config)
    with torch.no_grad():
        output = model(input_ids)
    if isinstance(output, tuple):
        output = output[0]
    assert output.shape[0] == 1
    assert output.shape[1] == 16
    assert config["skipped"] > 0


def test_v4_voc_skip():
    model, tokenizer = load_model()
    input_ids = torch.randint(0, tokenizer.vocab_size, (1, 16)).to(model.device)
    config = {"threshold": 1e9, "skipped": 0, "prev_hidden": None, "collect_data": False, "records": [], "num_layers": len(model.gpt_neox.layers)}
    router = Router(len(model.gpt_neox.layers)).to(model.device)
    apply_voc_skip(model, config, router)
    with torch.no_grad():
        output = model(input_ids)
    if isinstance(output, tuple):
        output = output[0]
    assert output.shape[0] == 1
    assert output.shape[1] == 16
    assert config["skipped"] >= 0
