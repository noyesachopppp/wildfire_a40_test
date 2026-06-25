#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="experiments/results/local_split_service_baseline"
mkdir -p "$OUT_DIR"

PYTHONPATH="$ROOT_DIR" python3 experiments/split_service_mock.py \
  --input wildfire_24fps.mp4 \
  --output "$OUT_DIR" \
  --frame-stride 30 \
  --environment-name local_split_service_baseline \
  --sam2-worker-count 1 \
  --vlm-worker-count 1 \
  --event-burst-size 1 \
  --timeout-ms 2000

echo "DONE: $OUT_DIR/performance_summary.json"
