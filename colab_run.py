"""
Google Colab runner  —  paste this entire file into a single code cell and run.

Runtime: GPU (T4)  |  Runtime > Change runtime type > T4 GPU

Colab Secrets required  (left sidebar → key icon → "Secrets"):
  WANDB_API_KEY  — your wandb API key
  HF_TOKEN       — HuggingFace token (for model downloads)
  GITHUB_TOKEN   — personal access token with `repo` scope (private repo access)

Flow mirrors run.py (Kaggle runner) but uses Colab paths and secret APIs:
  1. Mount Drive (optional – keeps critic.pth between sessions)
  2. Pull / clone latest repo via authenticated HTTPS
  3. Install missing deps
  4. Run baselines → train critic → eval critic → zero_shot → cka
"""

import os
import sys
import subprocess
import torch
from google.colab import userdata  # type: ignore

# ── 0. Confirm GPU ──────────────────────────────────────────────────────
if not torch.cuda.is_available():
    raise RuntimeError(
        "No GPU detected. Runtime > Change runtime type > T4 GPU, then re-run."
    )

print(f"[GPU] {torch.cuda.get_device_name(0)}  |  VRAM {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")

# ── 1. Colab Secrets ────────────────────────────────────────────────────
for key in ("WANDB_API_KEY", "HF_TOKEN", "GITHUB_TOKEN"):
    try:
        os.environ[key] = userdata.get(key)
        print(f"[SECRET] {key} loaded")
    except Exception:
        print(f"[WARN]   {key} not found in Colab Secrets — skipping")

# ── 2. Optional: Mount Drive to persist checkpoints ─────────────────────
DRIVE_AVAILABLE = False
try:
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive", force_remount=False)
    DRIVE_CHECKPOINT_DIR = "/content/drive/MyDrive/KV-Research/checkpoints"
    os.makedirs(DRIVE_CHECKPOINT_DIR, exist_ok=True)
    DRIVE_AVAILABLE = True
    print(f"[DRIVE] Checkpoints will be mirrored to {DRIVE_CHECKPOINT_DIR}")
except Exception as e:
    print(f"[WARN] Drive not mounted ({e}) — checkpoints live only in /content/repo")

# ── 3. Clone / pull repo ────────────────────────────────────────────────
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
REPO_BASE    = "github.com/KabirResearch/KV-Research.git"
REPO_URL     = f"https://{GITHUB_TOKEN}:x-oauth-basic@{REPO_BASE}" if GITHUB_TOKEN else f"https://{REPO_BASE}"
REPO_DIR     = "/content/repo"
CRITIC_CKPT  = os.path.join(REPO_DIR, "critic.pth")

if os.path.exists(os.path.join(REPO_DIR, ".git")):
    result = subprocess.run(
        ["git", "-C", REPO_DIR, "pull", "origin", "master"],
        capture_output=True, text=True
    )
    print(result.stdout.strip() or "[GIT] Already up to date")
    if result.returncode != 0:
        print(f"[WARN] git pull failed: {result.stderr.strip()}")
else:
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
    print(f"[GIT] Cloned to {REPO_DIR}")

# ── 4. Install dependencies ─────────────────────────────────────────────
# Colab ships torch + transformers; install the extras that are missing
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "fvcore", "lm-eval", "wandb", "accelerate", "datasets"],
    check=True,
)
print("[DEPS] Extra dependencies installed")

# ── 5. Bootstrap Python path ────────────────────────────────────────────
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

# ── 6. Runner helper ────────────────────────────────────────────────────
def run_mode(mode, extra_args=None):
    cmd = [sys.executable, "main.py", "--mode", mode]
    if extra_args:
        cmd.extend(extra_args)
    sep = "=" * 60
    print(f"\n{sep}\nRunning mode: {mode}\n{sep}")
    ret = subprocess.run(cmd, cwd=REPO_DIR)
    if ret.returncode != 0:
        print(f"[WARN] mode={mode} exited with code {ret.returncode}")
    return ret.returncode == 0


def mirror_to_drive(src, label="critic.pth"):
    """Copy a checkpoint to Drive if available."""
    if DRIVE_AVAILABLE and os.path.exists(src):
        import shutil
        dst = os.path.join(DRIVE_CHECKPOINT_DIR, label)
        shutil.copy2(src, dst)
        print(f"[DRIVE] Saved {label} → {dst}")

# ── 7. Stage 1: Baselines ───────────────────────────────────────────────
for mode in ["full", "static_25", "static_50", "random_skip", "baselines"]:
    run_mode(mode)

# ── 8. Stage 2: Train critic ────────────────────────────────────────────
print(f"\n{'='*60}\nTraining critic (always retrain on latest code)\n{'='*60}")
train_ok = run_mode("critic_train")
mirror_to_drive(CRITIC_CKPT, "critic.pth")

# ── 9. Stage 3: Eval critic ─────────────────────────────────────────────
if train_ok and os.path.exists(CRITIC_CKPT):
    print(f"\n[INFO] critic.pth found ({os.path.getsize(CRITIC_CKPT)//1024} KB) — running eval")
    run_mode("critic_eval")
    run_mode("critic_eval", ["--skip-rate", "0.25"])
else:
    print("[ERROR] Skipping critic_eval — training failed or critic.pth not found")

# ── 10. Stage 4: Zero-shot (base model reference) ───────────────────────
run_mode("zero_shot")

# ── 11. Stage 5: Zero-shot with SoftLayer skip ──────────────────────────
if train_ok and os.path.exists(CRITIC_CKPT):
    run_mode("zero_shot_skip", ["--skip-rate", "0.5"])
    run_mode("zero_shot_skip", ["--skip-rate", "0.25"])
else:
    print("[ERROR] Skipping zero_shot_skip — critic.pth not available")

# ── 12. Stage 6: Representation analysis (CKA) ──────────────────────────
run_mode("cka")

print("\n[DONE] All stages complete.")
if DRIVE_AVAILABLE:
    print(f"[DRIVE] Checkpoints saved to {DRIVE_CHECKPOINT_DIR}")
