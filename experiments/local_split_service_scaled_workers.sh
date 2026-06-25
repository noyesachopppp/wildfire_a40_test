#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="experiments/results/local_split_service_scaled_workers"
mkdir -p "$OUT_DIR"

PYTHONPATH="$ROOT_DIR" python3 experiments/split_service_mock.py \
  --input wildfire_24fps.mp4 \
  --output "$OUT_DIR" \
  --frame-stride 30 \
  --environment-name local_split_service_scaled_workers \
  --sam2-worker-count 2 \
  --vlm-worker-count 2 \
  --force-event-mode \
  --event-burst-size 10 \
  --timeout-ms 1500

echo "DONE: $OUT_DIR/performance_summary.json"
