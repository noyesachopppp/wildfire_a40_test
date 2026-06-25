#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p experiments/results/runpod_mock

python3 -m src.main \
  --input wildfire_24fps.mp4 \
  --output experiments/results/runpod_mock \
  --profile \
  --frame-stride 30 \
  --detector yolo \
  --yolo-model yolov8n \
  --yolo-backend mock \
  --environment-name runpod_mock \
  --run-type single_process_mock

echo "DONE: experiments/results/runpod_mock/performance_summary.json"
