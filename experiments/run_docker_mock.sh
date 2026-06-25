#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p experiments/results/docker_mock

if docker build -t wildfire-ai-alert:latest .; then
  docker run --rm \
    -v "$ROOT_DIR:/app" \
    -w /app \
    wildfire-ai-alert:latest \
    python3 -m src.main \
      --input wildfire_24fps.mp4 \
      --output experiments/results/docker_mock \
      --profile \
      --frame-stride 30 \
      --detector yolo \
      --yolo-model yolov8n \
      --yolo-backend mock \
      --environment-name docker_mock \
      --run-type single_process_docker_mock
else
  echo "Docker build failed, creating fallback docker_mock result on host."
  python3 -m src.main \
    --input wildfire_24fps.mp4 \
    --output experiments/results/docker_mock \
    --profile \
    --frame-stride 30 \
    --detector yolo \
    --yolo-model yolov8n \
    --yolo-backend mock \
    --environment-name docker_mock \
    --run-type single_process_docker_mock_fallback
fi

echo "DONE: experiments/results/docker_mock/performance_summary.json"
