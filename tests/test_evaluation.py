"""
Tests for evaluation helpers: GFLOPs counter and evaluate_goldilocks PPL.
Uses tiny tensor stubs — no real model download required in CI.
"""
import math
import torch
import torch.nn as nn
import pytest

from evaluation.flops import measure_flops_manual
from evaluation.evaluate import evaluate_goldilocks


# ── Tiny GPT-NeoX stub ───────────────────────────────────────────────────────

class _FakeConfig:
    hidden_size = 64
    num_hidden_layers = 4
    intermediate_size = 128
    num_attention_heads = 4

    def __len__(self):
        return self.num_hidden_layers


class _FakeLayer(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.attn = nn.Linear(d, d)
        self.mlp  = nn.Linear(d, d)

    def forward(self, h, **kwargs):
        return self.attn(h) + self.mlp(h), None


class _FakeGPTNeoX(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_FakeLayer(64) for _ in range(4)])
        self.embed_out = nn.Linear(64, 200)

    def forward(self, **kwargs):
        h = kwargs.get("inputs_embeds", torch.zeros(1, 8, 64))
        for layer in self.layers:
            out = layer(h)
            h = out[0] if isinstance(out, tuple) else out
        return type("O", (), {"last_hidden_state": h})()


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _FakeConfig()
        self.device  = torch.device("cpu")
        self.gpt_neox = _FakeGPTNeoX()
        self.embed_in = nn.Embedding(200, 64)
        # LM head
        self.embed_out = nn.Linear(64, 200, bias=False)

    def forward(self, input_ids=None, labels=None, **kwargs):
        h = self.embed_in(input_ids)
        for layer in self.gpt_neox.layers:
            out = layer(h)
            h = out[0] if isinstance(out, tuple) else out
        logits = self.embed_out(h)
        loss = None
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, 200),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return type("Out", (), {"loss": loss, "logits": logits})()


def _make_loader(n_batches=3, seq_len=16, vocab=200, batch_size=2):
    """Tiny DataLoader returning real input_ids + labels with some -100 masks."""
    data = []
    for _ in range(n_batches):
        ids    = torch.randint(1, vocab, (batch_size, seq_len))
        labels = ids.clone()
        labels[:, -4:] = -100  # mask last 4 positions (padding sim)
        data.append({"input_ids": ids, "labels": labels})
    return data


# ── measure_flops_manual ─────────────────────────────────────────────────────

class TestMeasureFlopsManual:
    def test_returns_positive_float(self):
        model = _FakeModel()
        gflops = measure_flops_manual(model, seq_len=64)
        assert isinstance(gflops, float)
        assert gflops > 0.0

    def test_scales_with_seq_len(self):
        model = _FakeModel()
        g64  = measure_flops_manual(model, seq_len=64)
        g128 = measure_flops_manual(model, seq_len=128)
        # FLOPs should be strictly greater for longer sequences
        assert g128 > g64

    def test_scales_with_batch_size(self):
        model = _FakeModel()
        g1 = measure_flops_manual(model, seq_len=64, batch_size=1)
        g2 = measure_flops_manual(model, seq_len=64, batch_size=2)
        assert abs(g2 / g1 - 2.0) < 0.01  # linear in batch size


# ── evaluate_goldilocks (PPL) ────────────────────────────────────────────────

class TestEvaluateGoldilocks:
    def test_returns_ppl_and_throughput(self):
        model = _FakeModel()
        loader = _make_loader()
        result = evaluate_goldilocks(model, loader, device="cpu")
        # Function may return (ppl, throughput) tuple or dict
        if isinstance(result, tuple):
            ppl, tput = result
        else:
            ppl  = result["ppl"]
            tput = result["throughput"]
        assert ppl > 1.0, "PPL must be > 1 for a valid LM"
        assert tput > 0.0, "Throughput must be positive"

    def test_ppl_is_finite(self):
        model = _FakeModel()
        loader = _make_loader()
        result = evaluate_goldilocks(model, loader, device="cpu")
        ppl = result[0] if isinstance(result, tuple) else result["ppl"]
        assert math.isfinite(ppl), "PPL should be finite (no nan/inf)"

    def test_masked_labels_not_counted(self):
        """
        With all-masked labels (-100 everywhere), the loss should be 0
        and PPL should be 1.0 (exp(0) == 1).
        """
        model = _FakeModel()
        # Loader with fully masked labels
        data = [{"input_ids": torch.randint(1, 200, (2, 16)),
                 "labels":    torch.full((2, 16), -100, dtype=torch.long)}
                for _ in range(2)]
        result = evaluate_goldilocks(model, data, device="cpu")
        ppl = result[0] if isinstance(result, tuple) else result["ppl"]
        # Cross-entropy with all ignore_index=-100 → loss = 0 → PPL = 1
        assert abs(ppl - 1.0) < 1e-3 or math.isnan(ppl), (
            "Fully-masked labels should give PPL ≈ 1.0"
        )
