#!/usr/bin/env bash
# Create .venv and install dependencies. Run from repo root:
#   bash scripts/setup_venv.sh
# Or:  bash scripts/setup_venv.sh cpu
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE="${1:-gpu}"
if [[ "$MODE" != "gpu" && "$MODE" != "cpu" ]]; then
  echo "Usage: $0 [cpu|gpu]" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
.venv/bin/pip install -U pip setuptools wheel

if [[ "$MODE" == "cpu" ]]; then
  echo "Installing CPU-only PyTorch (smaller download)..."
  .venv/bin/pip install 'torch>=2.0.0' --index-url https://download.pytorch.org/whl/cpu
else
  echo "Installing PyTorch from default PyPI (may include large CUDA wheels)..."
  .venv/bin/pip install 'torch>=2.0.0'
fi

echo "Installing remaining packages..."
.venv/bin/pip install -r requirements-after-pytorch.txt

echo ""
echo "Done. Activate the environment:"
echo "  source .venv/bin/activate"
echo "Then run (always use this Python, not /bin/python3):"
echo "  python scripts/make_paper_figures.py"
echo "  streamlit run dashboard/app.py"
