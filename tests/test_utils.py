import torch
import pytest
from utils.metrics import compute_loss, compute_entropy

def test_compute_loss():
    logits = torch.randn(1, 10, 100)
    labels = torch.randint(0, 100, (1, 10))
    loss = compute_loss(logits, labels, type('MockTokenizer', (), {'pad_token_id': 0})())
    assert loss.item() > 0

def test_compute_entropy():
    logits = torch.randn(1, 10, 100)
    entropy = compute_entropy(logits)
    assert entropy.item() > 0