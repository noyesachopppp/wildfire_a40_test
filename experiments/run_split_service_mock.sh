#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p experiments/results/split_service_mock

PYTHONPATH="$ROOT_DIR" python3 experiments/split_service_mock.py \
  --input wildfire_24fps.mp4 \
  --output experiments/results/split_service_mock \
  --frame-stride 30 \
  --environment-name split_service_mock

echo "DONE: experiments/results/split_service_mock/performance_summary.json"
