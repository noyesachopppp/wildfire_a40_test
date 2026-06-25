#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="experiments/results/runpod_yolov8m_vlm"
mkdir -p "$OUT_DIR"

python -m src.main \
  --input wildfire_24fps.mp4 \
  --output "$OUT_DIR" \
  --profile \
  --frame-stride 30 \
  --detector yolo \
  --yolo-model yolov8m \
  --yolo-backend ultralytics \
  --enable-vlm \
  --vlm-event-only \
  --vlm-model Qwen2.5-VL-7B \
  --vlm-max-new-tokens 64 \
  --environment-name runpod_yolov8m_vlm \
  --run-type runpod_host_python

echo "DONE: $OUT_DIR/performance_summary.json"
