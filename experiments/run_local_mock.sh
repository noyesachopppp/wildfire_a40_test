#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p experiments/results/local_mock

python3 -m src.main \
  --input wildfire_24fps.mp4 \
  --output experiments/results/local_mock \
  --profile \
  --frame-stride 30 \
  --detector yolo \
  --yolo-model yolov8n \
  --yolo-backend mock \
  --environment-name local_mock \
  --run-type single_process_mock

echo "DONE: experiments/results/local_mock/performance_summary.json"
