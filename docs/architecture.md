# SoftLayer — System Architecture

## Overview

SoftLayer is an adaptive layer-skipping framework for transformer LLMs. An attention-supervised `LogTemporalCritic` learns to predict token saliency, and a `SoftPlanningRouter` uses this per-token signal to skip unimportant layers at inference time — with a residual-safe identity mapping ensuring mathematical correctness.

---

## Full System Diagram

```mermaid
flowchart TD
    subgraph DATA["Data Pipeline"]
        A["Raw WikiText-103"] --> B["Tokenizer\nEleutherAI/pythia-1b"]
        B --> C["input_ids + labels\n(-100 padding mask)"]
        C --> DL1["train_dataloader"]
        C --> DL2["test_dataloader"]
    end

    subgraph CRITIC["Critic Training (train_block_critic)"]
        DL1 --> FWD["Frozen Base Model Forward\noutput_attentions=True\noutput_hidden_states=True"]
        FWD --> HS["Hidden States h_0 ... h_L"]
        FWD --> ATT["Attention Weights A_0 ... A_L"]
        HS --> CRIT_IN["Block Mean Hidden\nTARGET_LAYERS avg"]
        ATT --> ATGT["Attention Supervision Target\nsum attended / max+eps"]
        CRIT_IN --> CRIT["LogTemporalCritic\nlog1p|h| → LayerNorm → Linear → GELU → Sigmoid"]
        CRIT --> MSE["MSE Loss vs target"]
        ATGT --> MSE
        MSE --> TRAINED_C["Trained Critic"]
    end

    subgraph ROUTER["SoftPlanningRouter (Inference)"]
        DL2 --> EMBED["Embedding + Early Layers"]
        EMBED --> RL["Router Layer i"]
        TRAINED_C --> RL
        RL --> PROB["critic(h.detach()) → token probs"]
        PROB --> THRESH["Dynamic Threshold\nquantile(probs, skip_rate)"]
        THRESH -->|"prob >= thresh\nImportant"| FULL["Full Layer Forward"]
        THRESH -->|"prob < thresh\nSkip"| IDENT["Identity: h_out = h_in\n(Residual-safe)"]
        FULL --> NEXT["Next Layer ..."]
        IDENT --> NEXT
        NEXT --> LOGITS["Output Logits"]
    end

    subgraph EVAL["Evaluation Suite"]
        LOGITS --> PPL["Perplexity PPL"]
        LOGITS --> ZS["Zero-Shot: HellaSwag · PIQA · ARC"]
        HS --> MANIFOLD["CKA / Cosine Sim\nManifold Analysis"]
        FLOP_C["FLOP Counter"] --> FLOPS["GFLOPs"]
        CUDA_T["CUDA Timer"] --> LAT["Kernel Latency ms"]
    end

    subgraph BASELINES["Baseline Comparisons"]
        BS1["Static Skip"]
        BS2["Random Skip"]
        BS3["Token Pruning"]
        BS4["Early Exit"]
        BS5["MoE / MoD"]
        BS6["Speculative Decoding"]
    end

    subgraph REPORT["Results Table"]
        PPL --> RT["Method | PPL | GFLOPs | Latency | HellaSwag | CKA"]
        ZS --> RT
        MANIFOLD --> RT
        FLOPS --> RT
        LAT --> RT
    end
```

> Raw `.mmd` file: [`docs/diagrams/system.mmd`](diagrams/system.mmd)

---

## Module Map

```mermaid
graph LR
    main.py --> training/train_critic.py
    main.py --> training/train_router.py
    main.py --> evaluation/evaluate.py
    main.py --> evaluation/zero_shot.py
    main.py --> evaluation/flops.py
    main.py --> evaluation/latency.py
    main.py --> evaluation/manifold.py
    main.py --> baselines/static_skip.py
    main.py --> baselines/random_skip.py
    main.py --> baselines/token_pruning.py
    main.py --> baselines/early_exit.py
    main.py --> baselines/moe.py
    main.py --> baselines/mod.py
    main.py --> baselines/speculative.py
    training/train_critic.py --> models/critics.py
    training/train_router.py --> models/router.py
    evaluation/evaluate.py --> models/router.py
    models/router.py --> models/patchers.py
    data/dataset.py --> utils/config.py
    utils/model.py --> utils/config.py
```

---

## Critic Architecture

```
Input: hidden_state h  [batch, seq, dim]
    ↓
log1p(|h|)             [batch, seq, dim]  — log-scale stabilization
    ↓
LayerNorm(dim)
    ↓
Linear(dim → hidden)
    ↓
GELU
    ↓
Linear(hidden → 1)
    ↓
Sigmoid                → saliency score ∈ [0, 1]
```

## Router Skip Decision

```
probs = critic(h.detach())                  # [batch, seq, 1]
thresh = quantile(probs.view(-1), skip_rate)
mask = (probs >= thresh).to(h.dtype)        # 1 = process, 0 = skip

h_out = (layer(h) * mask) + (h * (1 - mask))  # Goldilocks zone
```
