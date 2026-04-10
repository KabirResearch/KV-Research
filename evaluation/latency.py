"""
Kernel-Level Latency Measurement
===================================
Measures GPU-accurate wall-clock latency using torch.cuda.Event.
This gives kernel-level timing, not Python-level, for fair comparison.

Pseudocode:
    warmup N times
    start = cuda.Event(); end = cuda.Event()
    start.record()
    for run in range(R): model(input_ids)
    end.record(); synchronize()
    avg_ms = start.elapsed_time(end) / R

Report:
    - End-to-end latency (ms per forward pass)
    - Per-layer latency breakdown (via hooks)
"""
import torch
import logging

logger = logging.getLogger(__name__)


def measure_latency(
    model,
    input_ids: torch.Tensor,
    warmup: int = 10,
    runs: int = 50,
) -> float:
    """
    Measure average GPU latency per forward pass using CUDA events.

    Args:
        model: HuggingFace causal LM (eval mode)
        input_ids: [batch, seq] input tensor on CUDA
        warmup: number of warmup runs (excluded from timing)
        runs: number of timed runs
    Returns:
        avg_ms: average latency in milliseconds
    """
    model.eval()
    device = input_ids.device
    assert device.type == "cuda", "Latency measurement requires CUDA device"

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids)
    torch.cuda.synchronize()

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    start_evt.record()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(input_ids)
    end_evt.record()
    torch.cuda.synchronize()

    avg_ms = start_evt.elapsed_time(end_evt) / runs
    return avg_ms


def measure_per_layer_latency(model, input_ids: torch.Tensor, warmup: int = 5, runs: int = 20):
    """
    Measure per-layer latency using forward hooks and CUDA events.

    Args:
        model: HuggingFace causal LM with model.gpt_neox.layers
        input_ids: [batch, seq] CUDA tensor
        warmup: warmup runs
        runs: timed runs
    Returns:
        list of (layer_idx, avg_ms) tuples
    """
    layer_times = {i: [] for i in range(len(model.gpt_neox.layers))}
    events = {}

    def make_hook(layer_idx):
        def hook(module, inp, out):
            e_start, e_end = events[layer_idx]
            e_end.record()
            torch.cuda.synchronize()
            layer_times[layer_idx].append(e_start.elapsed_time(e_end))
        return hook

    hooks = []
    for i, layer in enumerate(model.gpt_neox.layers):
        e_start = torch.cuda.Event(enable_timing=True)
        e_end = torch.cuda.Event(enable_timing=True)
        events[i] = (e_start, e_end)

        def make_pre_hook(idx, es):
            def pre_hook(module, inp):
                es.record()
            return pre_hook

        hooks.append(layer.register_forward_pre_hook(make_pre_hook(i, e_start)))
        hooks.append(layer.register_forward_hook(make_hook(i)))

    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_ids)
        layer_times = {i: [] for i in range(len(model.gpt_neox.layers))}
        for _ in range(runs):
            _ = model(input_ids)

    for h in hooks:
        h.remove()

    return [(i, sum(t) / len(t) if t else 0.0) for i, t in layer_times.items()]
