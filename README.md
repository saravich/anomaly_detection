# Anomaly Portfolio (Classical & Lightweight)

Image anomaly detection with a *production-ready* feel: classical SSIM/LBP pipelines, CLI tools,
Docker, tests, CI, and a short PDF report. Ships with a tiny **synthetic dataset** so you can demo offline.

## Quickstart

```bash
# 1) Create and activate a virtualenv (recommended)
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install
pip install -U pip
pip install -e .[dev]

# 3) Generate tiny synthetic dataset
make demo-data

# 4) Fit a classical model (SSIM template) on "good" training images
python -m adlib.cli.fit --config configs/classical_ssim.yaml --out .artifacts/ssim_model.pkl

# 5) Infer on test images and save heatmaps/predictions
python -m adlib.cli.infer --config configs/classical_ssim.yaml --model .artifacts/ssim_model.pkl --out .artifacts/preds

# 6) Evaluate and export a small PDF report
python -m adlib.cli.eval --preds .artifacts/preds --out .artifacts/metrics.json
python -m adlib.cli.report --config configs/classical_ssim.yaml --metrics .artifacts/metrics.json --preds .artifacts/preds --out .artifacts/report.pdf
```

## Features
- ✅ Classical methods: **SSIM heatmaps**, **LBP + IsolationForest** scoring
- ✅ Self-contained **synthetic dataset** (no external downloads)
- ✅ CLI tools: fit / infer / eval / report
- ✅ Reproducible: Dockerfile, pinned deps via `pyproject.toml`
- ✅ Tests + GitHub Actions CI + coverage
- ✅ Simple PDF report with metrics & qualitative panels

## Layout
```
anomaly-portfolio/
  src/adlib/...
  configs/
  tests/
  docker/
  .github/workflows/ci.yml
  Makefile
  README.md
```
