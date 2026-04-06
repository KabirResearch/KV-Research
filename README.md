# Layer Skipping for GPT-NeoX

This repository implements dynamic layer skipping techniques for GPT-NeoX models to improve inference efficiency. It includes static skipping, entropy-based skipping, and value-of-computation (VoC) based skipping.

## Features
- Modular scripts for targeted execution
- Automated Kaggle integration via GitHub Actions
- Support for CPU testing and GPU evaluation on Kaggle

## Setup

### Local Development
1. Clone the repo:
   ```bash
   git clone https://github.com/Kabir08/Jumper.git
   cd Jumper
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run locally (CPU-only for testing):
   ```bash
   python main.py --mode full
   ```

### Kaggle GPU Testing
- Pushes are automated via GitHub Actions.
- Run kernels on Kaggle for GPU access.
- Check outputs in kernel logs or download files.

## Usage

### Modes
- `full`: Evaluate full model
- `skip_25`: Static 25% skip
- `entropy`: Entropy-based skipping
- `voc_train`: Train VoC and Router models
- `voc_eval`: Evaluate VoC skipping

Example:
```bash
python main.py --mode voc_train
```

### Configuration
Edit `config.json` for hyperparameters.

## Constraints
- Local: CPU-only, no GPU
- Kaggle: Free tier (30 GPU hours/month)
- Manual kernel runs required

## Development
- Format code: `black *.py`
- Lint: `flake8 *.py`
- Test: `pytest`

## License
MIT