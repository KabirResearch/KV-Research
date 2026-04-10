# Baseline Methods

All baselines are evaluated under the same conditions: same model (Pythia-1B), same dataset (WikiText-103), same batch size and sequence length. Results are recorded for PPL, GFLOPs, latency, and zero-shot accuracy.

---

## Comparison Table (target)

| Method | PPL ↓ | GFLOPs ↓ | Latency ↓ | HellaSwag ↑ | PIQA ↑ | ARC-e ↑ | CKA Sim |
|---|---|---|---|---|---|---|---|
| Full Model (baseline) | — | — | — | — | — | — | 1.0 |
| Static Skip 25% | | | | | | | |
| Static Skip 50% | | | | | | | |
| Random Skip (matched rate) | | | | | | | |
| Dynamic Token Pruning | | | | | | | |
| Early Exit | | | | | | | |
| MoE (top-2) | | | | | | | |
| MoD (Mixture of Depths) | | | | | | | |
| Speculative Decoding | | | | | | | |
| **SoftLayer (ours)** | | | | | | | |

---

## 1. Static Layer Skipping

**Idea:** Always skip a fixed set of upper layers regardless of input.

```python
# Pseudocode
def apply_static_skip(model, skip_rate=0.25):
    num_layers = len(model.gpt_neox.layers)
    num_skip = int(num_layers * skip_rate)
    # Skip top layers (least specialized)
    skip_set = set(range(num_layers - num_skip, num_layers))
    for i, layer in enumerate(model.gpt_neox.layers):
        if i in skip_set:
            model.gpt_neox.layers[i] = IdentityLayer(layer)
    return model
```

**Implementation:** `baselines/static_skip.py`

---

## 2. Random Layer Skipping (Control)

**Idea:** Skip a random subset of layers to serve as a noise-controlled comparison.

```python
# Pseudocode
def apply_random_skip(model, skip_rate=0.25, seed=42):
    random.seed(seed)
    num_layers = len(model.gpt_neox.layers)
    num_skip = int(num_layers * skip_rate)
    skip_set = set(random.sample(range(num_layers), num_skip))
    for i in skip_set:
        model.gpt_neox.layers[i] = IdentityLayer(model.gpt_neox.layers[i])
    return model
```

**Implementation:** `baselines/random_skip.py`

---

## 3. Dynamic Token Pruning

**Idea:** Prune low-importance tokens from the sequence before feeding to each layer; restore at output.

```python
# Pseudocode
def token_pruning_forward(hidden_states, keep_rate=0.8):
    batch, seq, dim = hidden_states.shape
    # Score tokens by norm (proxy for importance)
    scores = hidden_states.norm(dim=-1)          # [batch, seq]
    k = int(seq * keep_rate)
    top_indices = scores.topk(k, dim=-1).indices  # [batch, k]
    pruned = gather(hidden_states, top_indices)   # [batch, k, dim]
    out = layer(pruned)                           # process pruned tokens
    # Scatter back; unpruned tokens keep identity
    full_out = scatter(out, top_indices, original=hidden_states)
    return full_out
```

**Paper reference:** [Token Merging (ToMe) — Bolya et al. 2022](https://arxiv.org/abs/2210.09461)  
**Implementation:** `baselines/token_pruning.py`

---

## 4. Early Exit Transformers

**Idea:** Add a lightweight classifier at each layer; if confidence exceeds a threshold, exit early without running remaining layers.

```python
# Pseudocode
def early_exit_forward(model, hidden_states, confidence_threshold=0.9):
    for i, layer in enumerate(model.layers):
        hidden_states = layer(hidden_states)
        if i >= min_exit_layer:
            logits = exit_head[i](hidden_states)
            confidence = softmax(logits).max(dim=-1).values.mean()
            if confidence > confidence_threshold:
                return logits, i  # exit at layer i
    return model.lm_head(hidden_states), len(model.layers)
```

**Paper reference:** [BERxiT — Xin et al. 2021](https://arxiv.org/abs/2109.15148)  
**Implementation:** `baselines/early_exit.py`

---

## 5. Mixture of Experts (MoE)

**Idea:** Replace dense FFN with multiple expert FFNs; a gating network routes each token to top-k experts.

```python
# Pseudocode
def moe_forward(hidden_states, experts, gate, top_k=2):
    gate_logits = gate(hidden_states)          # [batch, seq, num_experts]
    top_k_weights, top_k_idx = topk(gate_logits, k=top_k)
    weights = softmax(top_k_weights)           # normalize routing weights
    output = zeros_like(hidden_states)
    for k in range(top_k):
        expert_out = experts[top_k_idx[:, :, k]](hidden_states)
        output += weights[:, :, k:k+1] * expert_out
    return output
```

**Paper reference:** [Switch Transformers — Fedus et al. 2022](https://arxiv.org/abs/2101.03961)  
**Implementation:** `baselines/moe.py`

---

## 6. Mixture of Depths (MoD)

**Idea:** Each layer processes only a fixed capacity (top-c tokens by router score); remaining tokens use identity/residual.

```python
# Pseudocode
def mod_forward(hidden_states, layer, router, capacity_factor=0.5):
    batch, seq, dim = hidden_states.shape
    capacity = int(seq * capacity_factor)
    router_logits = router(hidden_states)        # [batch, seq]
    top_indices = router_logits.topk(capacity, dim=-1).indices
    selected = gather(hidden_states, top_indices)
    processed = layer(selected)                  # process only top-c tokens
    # Residual-safe: unselected tokens pass through unchanged
    output = hidden_states.clone()
    output = scatter(processed, top_indices, output)
    return output
```

**Paper reference:** [Mixture of Depths — Raposo et al. 2024](https://arxiv.org/abs/2404.02258)  
**Implementation:** `baselines/mod.py`

---

## 7. Speculative Decoding

**Idea:** A small draft model generates K candidate tokens quickly; a larger target model verifies them in parallel, accepting valid tokens and resampling on rejection.

```python
# Pseudocode
def speculative_decode(draft_model, target_model, prompt, K=4, max_tokens=100):
    tokens = prompt.clone()
    while len(tokens) < max_tokens:
        # Step 1: Draft K tokens
        draft_tokens, draft_probs = draft_model.generate_k(tokens, K)
        # Step 2: Target scores all K tokens in one forward pass
        target_probs = target_model.score(tokens, draft_tokens)
        # Step 3: Token-level acceptance sampling
        accepted = []
        for k in range(K):
            ratio = target_probs[k] / (draft_probs[k] + 1e-9)
            if random() < min(1.0, ratio):
                accepted.append(draft_tokens[k])
            else:
                # Resample from corrected distribution
                accepted.append(sample(target_probs[k] - draft_probs[k]))
                break
        tokens = cat([tokens, accepted])
    return tokens
```

**Paper reference:** [Speculative Decoding — Leviathan et al. 2023](https://arxiv.org/abs/2211.17192)  
**Implementation:** `baselines/speculative.py`
