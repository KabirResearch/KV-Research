# SoftLayer: Adaptive Layer Skipping via Attention-Supervised Critics

## Goal
Build a publishable, rigorously evaluated adaptive layer-skipping framework for transformer LLMs.
Compare against strong baselines. Report multiple metrics beyond PPL.

---

## Repository Structure

```
Layer_skpping/
|
+-- main.py                        <- arg parsing + orchestration only
+-- run.py                         <- Kaggle runner
+-- config.json                    <- experiment config
+-- requirements.txt
|
+-- models/                        <- ALL model architecture
|   +-- __init__.py
|   +-- critics.py                 <- LogTemporalCritic
|   +-- router.py                  <- SoftPlanningRouter
|   +-- patchers.py                <- apply_skip / apply_entropy_skip / apply_router_skip
|
+-- baselines/                     <- ALL baseline methods
|   +-- __init__.py
|   +-- static_skip.py             <- Static layer skipping (25%, 50%)
|   +-- random_skip.py             <- Random skip (control)
|   +-- token_pruning.py           <- Dynamic token pruning (ToMe-style)
|   +-- early_exit.py              <- Early exit transformers
|   +-- moe.py                     <- Mixture of Experts
|   +-- mod.py                     <- Mixture of Depths (Raposo et al. 2024)
|   +-- speculative.py             <- Speculative decoding
|
+-- training/                      <- ALL training logic
|   +-- __init__.py
|   +-- train_critic.py            <- train_block_critic (attention supervised)
|   +-- train_router.py            <- train_router
|
+-- evaluation/                    <- ALL evaluation logic
|   +-- __init__.py
|   +-- evaluate.py                <- evaluate_goldilocks, run_full, run_skip
|   +-- calibrate.py               <- calibrate_voc_thresholds
|   +-- flops.py                   <- FLOPs measurement (fvcore + manual)
|   +-- latency.py                 <- Kernel-level latency (CUDA events)
|   +-- manifold.py                <- CKA + Cosine Sim analysis
|   +-- zero_shot.py               <- HellaSwag, PIQA, ARC (lm-eval-harness)
|
+-- data/                          <- ALL data handling
|   +-- __init__.py
|   +-- dataset.py                 <- load_dataset_part with -100 label masking
|
+-- utils/                         <- General utilities
|   +-- __init__.py
|   +-- config.py                  <- config loading
|   +-- model.py                   <- load_model + gpt_neox patching
|   +-- metrics.py                 <- compute_loss, compute_entropy
|   +-- logging.py                 <- setup_logging
|
+-- docs/                          <- Documentation + diagrams
|   +-- architecture.md            <- System architecture + Mermaid diagram
|   +-- baselines.md               <- Baseline pseudocodes + paper refs
|   +-- metrics.md                 <- Metrics plan + formulas
|   +-- diagrams/
|       +-- system.mmd             <- Raw Mermaid diagram
|
+-- tests/
    +-- test_models.py
    +-- test_baselines.py
    +-- test_evaluation.py
    +-- test_utils.py
```

---

## Experiment Versions

### v0 -- Static Layer Skipping (baseline)
- Rule: fixed top 25% / 50% of layers replaced with identity
- No learning, no adaptivity
- Code: `baselines/static_skip.py`
- Metric: PPL upper bound, speed reference

### v1 -- Random Skip (control)
- Randomly selected layers skipped at matched rate
- Verifies that SoftLayer beats random; if not, the critic adds no value
- Code: `baselines/random_skip.py`

### v2 -- Dynamic Token Pruning
- Per-layer token importance scoring; low-importance tokens take identity path
- Reference: ToMe (Bolya et al. 2023), DynamicViT
- Code: `baselines/token_pruning.py`

### v3 -- Early Exit Transformers
- Exit after layer L when confidence (max softmax) exceeds threshold
- Reference: DeeBERT, PABEE
- Code: `baselines/early_exit.py`

### v4 -- Residual-safe Skipping (critical correctness fix)
- Skip = identity residual: `h_out = layer(h)*mask + h*(1-mask)`
- Prevents norm explosion from raw bypass
- Used as the skip primitive in SoftLayer and all skip-based baselines

### v5 -- SoftLayer (our method)
- LogTemporalCritic: `log1p|h| -> LN -> Linear(2048->256) -> GELU -> Linear(256->1) -> Sigmoid`
- Trained with attention supervision: target = normalised per-token attended mass from frozen model
- SoftPlanningRouter: `mask = (critic(h) >= quantile(probs, skip_rate))`
- TARGET_LAYERS = [10, 12, 14]  (deeper layers first; empirically most skippable)
- Code: `models/critics.py`, `models/router.py`, `training/train_critic.py`

### v6 -- Mixture of Experts (MoE)
- Replace MLP block with sparse top-k MoE
- Maintains same parameter count; redistributes compute
- Reference: Switch Transformer (Fedus et al. 2022)
- Code: `baselines/moe.py`

### v7 -- Mixture of Depths (MoD)
- Token-level routing per layer; unrouted tokens pass through via residual only
- Reference: Raposo et al., "Mixture of Depths", arXiv:2404.02258, 2024
- Code: `baselines/mod.py`

### v8 -- Speculative Decoding
- Draft model generates K tokens; target model verifies with acceptance sampling
- Orthogonal to skipping; included as a compute-efficiency upper bound
- Reference: Leviathan et al. 2023
- Code: `baselines/speculative.py`

---

## Baselines Comparison Table (to be filled after experiments)

| Method              | PPL down | GFLOPs down | Latency ms down | HellaSwag up | PIQA up | ARC-E up | CKA-delta down |
|---------------------|----------|-------------|-----------------|--------------|---------|----------|----------------|
| Full Model          |   --     |    --       |    --           |    --        |  --     |  --      |  0.0           |
| Static 25%          |          |             |                 |              |         |          |                |
| Static 50%          |          |             |                 |              |         |          |                |
| Random Skip         |          |             |                 |              |         |          |                |
| Token Pruning       |          |             |                 |              |         |          |                |
| Early Exit          |          |             |                 |              |         |          |                |
| MoE                 |          |             |                 |              |         |          |                |
| MoD                 |          |             |                 |              |         |          |                |
| SoftLayer (ours)    |          |             |                 |              |         |          |                |

---

## Metrics Plan

### 1. Perplexity
- PPL = exp(-1/N * sum log p(w_i | w_{<i}))
- Labels must be -100 at padding positions (fixed in data/dataset.py)

### 2. FLOPs
- Per transformer layer: FLOPs_attn = 4*L*d^2 + 2*L^2*d, FLOPs_MLP = 8*L*d^2
  (L=128, d=2048 for pythia-1b)
- Measured with fvcore.nn.FlopCountAnalysis or manual formula
- Report: GFLOPs per forward pass, % reduction vs full model

### 3. Kernel-Level Latency
- torch.cuda.Event(enable_timing=True) per forward call
- 5 warmup iters, 50 measurement iters, report p50/p95 in ms
- Per-layer hooks via register_forward_hook on gpt_neox.layers[i]

### 4. CKA / Cosine Similarity (Manifold Analysis)
- CKA(K, L) = HSIC(K,L) / sqrt(HSIC(K,K) * HSIC(L,L))
  where K = X X^T, L = Y Y^T (centred Gram matrices)
- High CKA delta between full model and baseline => representation divergence => quality risk
- Implemented in evaluation/manifold.py

### 5. Zero-Shot Evaluation
- Harness: lm_eval (EleutherAI/lm-evaluation-harness)
- Tasks: hellaswag, piqa, arc_easy, arc_challenge, winogrande
- Metric: normalised accuracy, 0-shot
- Implemented in evaluation/zero_shot.py

---

## Implementation Checklist

- [x] utils/ package           -- config, model, metrics, logging
- [x] baselines/ package       -- 7 baselines, all stubbed with correct API
- [x] training/ package        -- train_critic (attention supervised), train_router
- [x] evaluation/ package      -- evaluate, calibrate, flops, latency, manifold, zero_shot
- [x] data/ package            -- -100 label masking fix
- [x] docs/                    -- architecture.md, baselines.md, metrics.md, diagrams/system.mmd
- [ ] models/ package          -- critics.py (LogTemporalCritic), router.py (SoftPlanningRouter), patchers.py
- [ ] Fix stale imports        -- train.py, data.py, tests/ still use old `from utils import ...`
- [ ] Wire main.py             -- replace inline logic with imports from new packages
- [ ] Run experiments          -- fill in baselines comparison table above

---

## Key Bugs Fixed
- PPL inflation: old code included padding tokens in cross-entropy. Fixed in data/dataset.py with -100 masking.
- Import chain: `from utils import ...` must become `from utils.{module} import ...`
- Residual math: skip primitive must be `h_out = layer(h)*mask + h*(1-mask)`, not raw bypass.
