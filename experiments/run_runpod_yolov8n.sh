#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p experiments/results/runpod_yolov8n

python3 -m src.main \
  --input wildfire_24fps.mp4 \
  --output experiments/results/runpod_yolov8n \
  --profile \
  --frame-stride 30 \
  --detector yolo \
  --yolo-model yolov8n \
  --yolo-backend ultralytics \
  --environment-name runpod_yolov8n \
  --run-type single_process_real

echo "DONE: experiments/results/runpod_yolov8n/performance_summary.json"
