#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-config/table1.yaml}

echo "Running Table 1 reproduction"
echo "Using config: $CONFIG"

echo "[1/3] prepare all cutoffs"
python -m src.prepare_cutoff --config "$CONFIG"

echo "[2/3] smoothing stage (global, cached)"
python -m src.smoothing --config "$CONFIG"

echo "[3/3] train/eval across cutoffs"
python -m src.train_eval --config "$CONFIG"

echo "Done."
