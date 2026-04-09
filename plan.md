# Plan: Systematic evolution of KV/skip idea into publishable variants

## Goal
Systematic evolution of KV/skip idea into publishable variants

---

### v0 — Static Layer Skipping (baseline)
- Rule: fixed layers skipped (e.g., top 25%)
- No learning, no adaptivity
- Code: mask layers by index
- Metric: speed ↑, loss ↑ (reference point)

---

### v1 — Heuristic Dynamic Skipping
- Signals: norm, variance, entropy proxies
- Per-token / per-batch decision
- Your current feature stack ≈ this
- Weakness: noisy, unstable thresholds

---


---

### v3 — Residual-safe Skipping (critical fix)
- Respect transformer math:
  - LN → Attn → Residual
  - LN → MLP → Residual
- Skip = identity residual, not raw bypass
- Needed for stability + correctness

---

### v4 — Token-level Routing
- Instead of skipping whole layer:
  - route only *unimportant tokens*
- Keep important tokens full-compute
- Implementation:
  - mask tokens inside attention/MLP
- Gain: better quality-speed tradeoff

---

### v5 — KV-aware Importance
- Use attention/KV stats:
  - attention weights
  - key norms
- Score = “future usefulness”
- Idea: tokens that won’t be attended → skip

---

### v6 — Planning-aware (research-grade)
- Predict multi-step usefulness
- Use:
  - temporal consistency
  - RL / learned critic
- “Will this token matter 5 steps later?”
- This is novel territory (NeurIPS-level)

---

### v7 — Joint System (full architecture)
- Components:
  - planner (importance predictor)
  - executor (skip/route)
  - KV controller (evict/compress)
- Closed-loop system:
  - feedback from loss / attention

---

### How to implement (order)
1. v0 → sanity baseline
2. v3 → correct math (must)
4. v4 → token routing (big gain)
5. v5 → KV-aware scoring
6. v6 → future-aware predictor

---

### Metrics to track
- loss / perplexity
- tokens/sec
- % compute skipped
- attention drift

---

### Research hooks
- v4–v6 = publishable
- v6–v7 = strong paper (planner + KV control)

---

Let me know if you want the exact PyTorch structure for v3–v5 or further breakdowns for implementation!
