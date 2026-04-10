#!/usr/bin/env python
"""
Kaggle kernel runner.
Pulls the latest repo from GitHub, installs deps, runs all experiments,
and logs results to wandb.

Flow:
  1. Pull latest code from GitHub
  2. Install dependencies
  3. Run baselines (no training needed): full, static_25, static_50, random_skip, baselines
  4. Train critic (critic_train) — always retrain on latest code
  5. Evaluate critic (critic_eval) — only after confirmed training success
  6. Run zero_shot eval

Set Kaggle secrets (Add-ons → Secrets):
  WANDB_API_KEY  — your wandb API key
  GITHUB_TOKEN   — optional, for private repo access
  HF_TOKEN       — optional, for higher HuggingFace rate limits
"""

import os
import sys
import subprocess

# ── Secrets ────────────────────────────────────────────────────────────
try:
    from kaggle_secrets import UserSecretsClient  # type: ignore  # Kaggle-only

    secrets = UserSecretsClient()
    for key in ("WANDB_API_KEY", "HF_TOKEN", "GITHUB_TOKEN"):
        try:
            os.environ[key] = secrets.get_secret(key)
        except Exception:
            pass
except ImportError:
    pass  # running locally

# ── Repo setup ─────────────────────────────────────────────────────────
REPO_URL = "https://github.com/KabirResearch/KV-Research.git"
REPO_DIR = "/kaggle/working/repo"
CRITIC_CKPT = os.path.join(REPO_DIR, "critic.pth")

if os.path.exists(REPO_DIR):
    # Always pull latest code so eval reflects newest changes
    result = subprocess.run(["git", "-C", REPO_DIR, "pull", "origin", "master"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(f"[WARN] git pull failed: {result.stderr}")
else:
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)

# ── Install extra deps not on Kaggle by default ─────────────────────────
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "fvcore", "lm-eval", "wandb"],
    check=True,
)

# ── Helpers ─────────────────────────────────────────────────────────────
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)


def run_mode(mode, extra_args=None):
    cmd = [sys.executable, "main.py", "--mode", mode]
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n{'='*60}\nRunning mode: {mode}\n{'='*60}")
    ret = subprocess.run(cmd, cwd=REPO_DIR)
    if ret.returncode != 0:
        print(f"[WARN] mode={mode} exited with code {ret.returncode}")
    return ret.returncode == 0


# ── Stage 1: Baselines — no training required ───────────────────────────
for mode in ["full", "static_25", "static_50", "random_skip", "baselines"]:
    run_mode(mode)

# ── Stage 2: Train critic on latest code ───────────────────────────────
print(f"\n{'='*60}\nTraining critic (always retrain on latest code)\n{'='*60}")
train_ok = run_mode("critic_train")

# ── Stage 3: Eval critic — only if training succeeded ──────────────────
if train_ok and os.path.exists(CRITIC_CKPT):
    print(f"\n[INFO] critic.pth found ({os.path.getsize(CRITIC_CKPT)//1024} KB) — running eval")
    run_mode("critic_eval")
else:
    print("[ERROR] Skipping critic_eval — training failed or critic.pth not found")

# ── Stage 4: Zero-shot eval ─────────────────────────────────────────────
run_mode("zero_shot")

print("\nAll experiments complete. Check wandb for results.")
