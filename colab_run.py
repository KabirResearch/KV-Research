"""
Google Colab runner  —  paste this entire file into a single code cell and run.

Runtime: GPU (T4 / A100)  |  Runtime > Change runtime type > GPU

Colab Secrets required  (left sidebar → key icon → "Secrets"):
  WANDB_API_KEY  — your wandb API key
  HF_TOKEN       — HuggingFace token (for model downloads)
  GITHUB_TOKEN   — personal access token with `repo` scope (private repo access)

Pipeline (runs automatically end-to-end):
  Stage 1 : full model PPL reference
  Stage 2 : all baselines (static_25/50, random_skip, token_pruning, moe, mod)
  Stage 3 : train critic (pythia-1b)
  Stage 4 : critic_eval at skip 25% and 50%
  Stage 5 : zero-shot benchmarks — base + skip 25%/50%
  Stage 6 : CKA / representation analysis
  Stage 7 : FLOPs + latency measurements
  Stage 8 : repeat stages 1-7 for pythia-2.8b (CONFIG_FILE=config_2.8b.json)
  Stage 9 : generate results_table.csv from W&B
"""

import os
import sys
import shutil
import subprocess
import torch
from google.colab import userdata  # type: ignore

# ── 0. Confirm GPU ──────────────────────────────────────────────────────
if not torch.cuda.is_available():
    raise RuntimeError("No GPU detected. Runtime > Change runtime type > GPU, then re-run.")

vram_gb = torch.cuda.get_device_properties(0).total_memory // 1024**3
print(f"[GPU] {torch.cuda.get_device_name(0)}  |  VRAM {vram_gb} GB")

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
REPO_BASE = "github.com/KabirResearch/KV-Research.git"
REPO_URL = f"https://{GITHUB_TOKEN}:x-oauth-basic@{REPO_BASE}" if GITHUB_TOKEN else f"https://{REPO_BASE}"
REPO_DIR = "/content/repo"
CRITIC_CKPT = os.path.join(REPO_DIR, "critic.pth")

if os.path.exists(os.path.join(REPO_DIR, ".git")):
    result = subprocess.run(["git", "-C", REPO_DIR, "pull", "origin", "master"], capture_output=True, text=True)
    print(result.stdout.strip() or "[GIT] Already up to date")
    if result.returncode != 0:
        print(f"[WARN] git pull failed: {result.stderr.strip()}")
else:
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
    print(f"[GIT] Cloned to {REPO_DIR}")

# ── 4. Install dependencies ─────────────────────────────────────────────
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "fvcore", "lm-eval", "wandb", "accelerate", "datasets"],
    check=True,
)
print("[DEPS] Extra dependencies installed")

# ── 5. Bootstrap Python path ────────────────────────────────────────────
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

# ── 6. Helpers ──────────────────────────────────────────────────────────

def run_mode(mode, extra_args=None, config_file=None):
    """Run main.py --mode <mode> [extra_args] with optional CONFIG_FILE override."""
    cmd = [sys.executable, "main.py", "--mode", mode]
    if extra_args:
        cmd.extend(extra_args)
    env = os.environ.copy()
    if config_file:
        env["CONFIG_FILE"] = os.path.join(REPO_DIR, config_file)
    sep = "=" * 60
    print(f"\n{sep}\nRunning mode: {mode}  config: {config_file or 'config.json'}\n{sep}")
    ret = subprocess.run(cmd, cwd=REPO_DIR, env=env)
    if ret.returncode != 0:
        print(f"[WARN] mode={mode} exited with code {ret.returncode}")
    return ret.returncode == 0


def mirror_to_drive(src, label):
    if DRIVE_AVAILABLE and os.path.exists(src):
        dst = os.path.join(DRIVE_CHECKPOINT_DIR, label)
        shutil.copy2(src, dst)
        print(f"[DRIVE] Saved {label} → {dst}")


def run_pipeline(config_file=None, critic_backup="critic.pth"):
    """
    Run the full SoftLayer experiment pipeline for one model config.
    critic_backup: filename used when mirroring critic.pth to Drive.
    """
    label = config_file or "config.json"
    print(f"\n{'#'*60}\n## PIPELINE START: {label}\n{'#'*60}")

    # Stage 1: Full model reference
    run_mode("full", config_file=config_file)

    # Stage 2: All baselines (static_25/50, random_skip, token_pruning, moe, mod)
    run_mode("baselines", config_file=config_file)

    # Stage 3: Train critic
    print(f"\n{'='*60}\nTraining critic\n{'='*60}")
    train_ok = run_mode("critic_train", config_file=config_file)
    mirror_to_drive(CRITIC_CKPT, critic_backup)

    # Stage 4: Critic eval (PPL + throughput)
    if train_ok and os.path.exists(CRITIC_CKPT):
        kb = os.path.getsize(CRITIC_CKPT) // 1024
        print(f"\n[INFO] critic.pth ready ({kb} KB)")
        run_mode("critic_eval", ["--skip-rate", "0.50"], config_file=config_file)
        run_mode("critic_eval", ["--skip-rate", "0.25"], config_file=config_file)
    else:
        print("[ERROR] Skipping critic_eval — training failed or critic.pth not found")

    # Stage 5: Zero-shot benchmarks
    run_mode("zero_shot", config_file=config_file)
    if train_ok and os.path.exists(CRITIC_CKPT):
        run_mode("zero_shot_skip", ["--skip-rate", "0.50"], config_file=config_file)
        run_mode("zero_shot_skip", ["--skip-rate", "0.25"], config_file=config_file)
    else:
        print("[ERROR] Skipping zero_shot_skip — critic.pth not available")

    # Stage 6: CKA / representation analysis
    if train_ok and os.path.exists(CRITIC_CKPT):
        run_mode("cka", ["--skip-rate", "0.50"], config_file=config_file)
    else:
        print("[ERROR] Skipping cka — critic.pth not available")

    # Stage 7: FLOPs + latency
    run_mode("flops", config_file=config_file)
    run_mode("latency", config_file=config_file)

    print(f"\n{'#'*60}\n## PIPELINE DONE: {label}\n{'#'*60}")


# ── 7. Run pythia-1b pipeline ────────────────────────────────────────────
run_pipeline(config_file=None, critic_backup="critic_1b.pth")

# Back up 1b critic before 2.8b overwrites it
if os.path.exists(CRITIC_CKPT):
    shutil.copy2(CRITIC_CKPT, os.path.join(REPO_DIR, "critic_1b.pth"))

# ── 8. Run pythia-2.8b pipeline ─────────────────────────────────────────
# 2.8b requires ~6 GB VRAM; safe on A100, may OOM on T4 (skip if needed)
if vram_gb >= 16:
    run_pipeline(config_file="config_2.8b.json", critic_backup="critic_2b.pth")
    # Restore 1b critic so results_table.py sees the right checkpoint
    if os.path.exists(os.path.join(REPO_DIR, "critic_1b.pth")):
        shutil.copy2(os.path.join(REPO_DIR, "critic_1b.pth"), CRITIC_CKPT)
else:
    print(f"[SKIP] pythia-2.8b pipeline skipped — only {vram_gb} GB VRAM (need ≥16 GB)")
    print("[TIP]  Switch to A100 runtime and re-run to include the 2.8b results")

# ── 9. Build results table ───────────────────────────────────────────────
print(f"\n{'='*60}\nBuilding results table from W&B\n{'='*60}")
ret = subprocess.run([sys.executable, "results_table.py"], cwd=REPO_DIR)
if ret.returncode != 0:
    print("[WARN] results_table.py failed — check WANDB_API_KEY secret")

print("\n[DONE] All stages complete.")
if DRIVE_AVAILABLE:
    print(f"[DRIVE] Checkpoints saved to {DRIVE_CHECKPOINT_DIR}")
