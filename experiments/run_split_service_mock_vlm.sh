#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="experiments/results/split_service_mock_vlm"
mkdir -p "$OUT_DIR"

PYTHONPATH="$ROOT_DIR" python experiments/split_service_mock.py \
  --input wildfire_24fps.mp4 \
  --output "$OUT_DIR" \
  --frame-stride 30 \
  --environment-name split_service_mock_vlm

echo "DONE: $OUT_DIR/performance_summary.json"
