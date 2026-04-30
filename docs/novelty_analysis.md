# SoftLayer — Novelty Analysis & Prior Art

## TL;DR

**SoftLayer is moderately novel.** The exact combination of (1) a *separately trained*, (2) *attention-supervised* saliency critic used for (3) *per-token, post-hoc layer skipping* at inference does not appear to exist in the literature as of April 2026. The closest threat is Mixture of Depths (MoD), but the training paradigm and supervision signal are meaningfully different. That said, the novelty claim needs careful framing — several individual components exist in prior work.

---

## Method Summary (what we're actually doing)

```
Offline phase:
  frozen model → attention weights at TARGET_LAYERS
  critic trained: MSE(critic(block_mean_hidden), normalized_attn_received_per_token)

Inference phase:
  per token, per target layer: critic(h) → saliency score
  if score < quantile_threshold(seq): skip layer (h_out = h_in)
```

Key design choices:
- **Decoupled training**: critic is a separate MLP, frozen base model is never modified
- **Attention supervision**: training signal = how much each token is attended to (∑ₐ attn[a→token] / max), not task loss
- **Log-temporal features**: `log1p(|h|)` input to critic compresses dynamic range
- **Block-level aggregation**: average over multiple middle layers (TARGET_LAYERS) as critic input
- **Dynamic quantile threshold**: per-sequence, not a fixed global scalar

---

## Prior Art Survey

### 1. Mixture of Depths — Raposo et al. (2024) [`arXiv:2404.02258`]
**Closest competitor.**

| | MoD | SoftLayer |
|---|---|---|
| Routing granularity | Per token, per layer | Per token, per target layer |
| Router trained | Jointly with model (task loss) | Separately offline (attention supervision) |
| Base model modified? | Yes (router added, retrained) | No (frozen pretrained model) |
| Supervision signal | Next-token prediction loss | Attention weight sum |
| Threshold | Fixed capacity k (top-k) | Dynamic quantile per sequence |

**Key distinction**: MoD is a *training-time* architectural choice — you cannot apply it to an existing pretrained model without retraining. SoftLayer is post-hoc and applies to any frozen transformer.

---

### 2. LayerSkip — Elhoushi et al. (2024) [`arXiv:2404.16710`], ACL 2024
Early exit with layer dropout training + self-speculative decoding.

- Requires **retraining** with layer dropout (graduated dropout rates) and early exit loss
- Exits at a prefix of layers (layers 0..k), not arbitrary middle layers
- No per-token routing — entire sequence exits at the same layer
- Achieves 1.82–2.16× speedup on Llama variants

**Verdict**: Orthogonal mechanism (whole-sequence early exit vs. per-token per-layer skip), but requires model retraining — SoftLayer does not.

---

### 3. ShortGPT — Men et al. (2024) [`arXiv:2403.03853`]
Block Influence (BI) metric = `1 - cos_sim(layer_input, layer_output)`. Statically removes low-BI layers post-hoc.

- **Static** pruning: same layers removed for all inputs
- No routing, no criterion, no inference-time adaptivity
- Works post-hoc on frozen models (similar deployment story to ours)
- Shows that upper layers of LLMs are highly redundant

**Verdict**: Validates SoftLayer's premise (redundant layers exist), but ShortGPT is static while SoftLayer is *dynamic and input-adaptive*.

---

### 4. Token Pruning / H2O / Scissorhands
Attention-pattern-based methods for KV cache eviction or token dropping inside a single layer's attention.

- These use attention scores to decide *which tokens to keep in the KV cache*, not *which layers to skip*
- Operating at the attention-head level, not the layer level
- SoftLayer uses attention patterns for a *different purpose*: as a training signal for a cross-layer saliency predictor

**Verdict**: Conceptually related (both leverage attention as a proxy for token importance), but the mechanism and objective are different. SoftLayer should cite these to justify the attention supervision signal.

---

### 5. SkipDecode — Del Corro et al. (2023)
Token-level early exit for autoregressive generation. Tokens exit at different layers but only in a "prefix-first" manner (later tokens in the sequence exit earlier).

- Static schedule: earlier tokens in autoregressive generation always process more layers
- No learned routing, no critic
- Only validated for causal LM generation (not classification or comprehension)

**Verdict**: Addresses a similar goal (skipping layers per token), but uses a static schedule rather than a learned critic.

---

## Novelty Assessment

### What is genuinely novel

1. **Attention-supervised decoupled critic training**
   — Using multi-head attention weight sums as a *supervision signal* for a learned saliency predictor, trained independently of the base model's task loss, is not present in any paper found. Prior token routing methods either use task gradients (MoD) or static importance metrics (ShortGPT).

2. **Post-hoc applicability to frozen pretrained models**
   — SoftLayer requires no base model retraining. LayerSkip, MoD, and MoE baselines all require training or architectural changes from scratch. This is a practically significant property.

3. **Cross-layer block-level representation for routing**
   — Averaging hidden states over multiple middle layers as the critic's input is a multi-scale feature that captures intermediate-block behavior rather than single-layer state. This does not appear in existing routing literature.

4. **Dynamic per-sequence quantile threshold**
   — Most routing methods use a fixed global capacity (MoD uses fixed top-k per layer). SoftLayer adapts the threshold to each sequence's distribution, which is better suited to variable-difficulty inputs.

### Where the novelty is fragile

1. **The "attention as token importance" idea is not new** — it is widely discussed in interpretability literature (attention rollout, gradient×attention, etc.) and in KV eviction (H2O, Scissorhands). SoftLayer's novelty is the *use* of it as a *training signal for a layer-routing critic*, not the idea itself.

2. **MoD is very close in spirit** — the per-token per-layer routing design is nearly identical. The paper must foreground the *training decoupling + attention supervision* distinction clearly and early, or reviewers will treat SoftLayer as a variant of MoD.

3. **Scale**: All experiments use Pythia-1B on WikiText-103. Reviewers at top venues will ask for results on ≥7B models. The findings may not transfer — smaller models have fewer redundant layers.

---

## Recommended Framing for a Paper

**Claim to make**: 
> "We show that multi-layer attention patterns in a frozen pretrained LLM can be distilled into a lightweight saliency critic, enabling post-hoc per-token layer skipping without any modification to the base model's weights or training procedure."

**Claims to avoid**:
- "First adaptive layer skipping method" — MoD does this
- "First use of attention for importance scoring" — well-established

**Differentiators to emphasize**:
- Post-hoc (no retraining)
- Self-supervised attention supervision (no human labels, no task loss)
- Dynamic quantile threshold (input-adaptive)
- Representation preservation validated via CKA

---

## Recommended Additional Baselines (if targeting top venue)

Current baseline set:

| Baseline | Purpose |
|---|---|
| Full model | Upper bound |
| Static Skip 25% | Lower bound for any skipping |
| Random Skip 25% | Ablates: is routing better than random? |
| Token Pruning | Ablates: layer skip vs. token pruning |
| Mixture of Depths (MoD) | Closest learned routing comparator |
| SoftLayer 25% / 50% (ours) | Main results |

**Note**: MoD is the closest learned-routing baseline, so it should stay in the main comparison set. It is the strongest external comparator for SoftLayer.

---

## Conclusion

SoftLayer is **novel enough to publish** at a workshop or mid-tier venue (EMNLP findings, EACL, CoLM). For top venues (NeurIPS, ICML, ACL main), the primary risks are:
1. Perceived similarity to MoD (must show clear empirical + conceptual differentiation)
2. Scale of experiments (Pythia-1B only)
3. The attention-supervision choice needs an ablation (vs. task-loss-trained critic) to prove it actually helps

The core insight — that attention patterns encode layer-skippability per token, and that this signal can be distilled into an offline critic — is a clean and practical contribution.
