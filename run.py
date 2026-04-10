#!/usr/bin/env python
"""
Kaggle kernel runner.
Pulls the latest repo from GitHub, installs deps, runs all experiments,
and logs results to wandb.

Set Kaggle secrets (Add-ons → Secrets):
  WANDB_API_KEY  — your wandb API key
  GITHUB_TOKEN   — optional, for private repo
"""

import os
import sys
import subprocess

# ── Secrets ────────────────────────────────────────────────────────────
from kaggle_secrets import UserSecretsClient  # type: ignore  # Kaggle-only

secrets = UserSecretsClient()
try:
    os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
except Exception:
    pass  # local or secret not set

# ── Repo setup ─────────────────────────────────────────────────────────
REPO_URL = "https://github.com/Kabir08/Jumper.git"
REPO_DIR = "/kaggle/working/repo"

if os.path.exists(REPO_DIR):
    subprocess.run(["git", "-C", REPO_DIR, "pull", "origin", "main"], check=True)
else:
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)

# ── Install extra deps not on Kaggle by default ─────────────────────────
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fvcore", "lm-eval", "wandb"], check=True)

# ── Run experiments ─────────────────────────────────────────────────────
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

modes = ["full", "static_25", "static_50", "random_skip", "critic_train", "critic_eval", "baselines"]
for mode in modes:
    print(f"\n{'='*60}\nRunning mode: {mode}\n{'='*60}")
    ret = subprocess.run(
        [sys.executable, "main.py", "--mode", mode],
        cwd=REPO_DIR,
    )
    if ret.returncode != 0:
        print(f"[WARN] mode={mode} exited with code {ret.returncode}")

print("\nAll experiments complete. Check wandb for results.")
