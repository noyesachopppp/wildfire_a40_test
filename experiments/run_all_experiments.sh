#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p experiments/results

run_step() {
  local name="$1"
  local cmd="$2"
  echo "=== RUN: $name ==="
  if bash -lc "$cmd"; then
    echo "=== OK: $name ==="
  else
    echo "=== FAIL (continue): $name ==="
  fi
}

run_step "local_mock" "./experiments/run_local_mock.sh"
run_step "runpod_mock" "./experiments/run_runpod_mock.sh"
run_step "runpod_yolov8n" "./experiments/run_runpod_yolov8n.sh"
run_step "runpod_yolov8m" "./experiments/run_runpod_yolov8m.sh"
run_step "docker_mock" "./experiments/run_docker_mock.sh"
run_step "split_service_mock" "./experiments/run_split_service_mock.sh"
run_step "aggregate_results" "python3 experiments/aggregate_results.py"

echo
echo "=== Result Files ==="
for p in \
  experiments/results/local_mock/performance_summary.json \
  experiments/results/runpod_mock/performance_summary.json \
  experiments/results/runpod_yolov8n/performance_summary.json \
  experiments/results/runpod_yolov8m/performance_summary.json \
  experiments/results/docker_mock/performance_summary.json \
  experiments/results/split_service_mock/performance_summary.json \
  experiments/results/bottleneck_comparison.csv \
  experiments/results/bottleneck_comparison.md
do
  if [ -f "$p" ]; then
    echo "$p"
  else
    echo "MISSING: $p"
  fi
done

echo
echo "RunPod baseline identifies model/I/O bottlenecks, split-service mode identifies queue/worker bottlenecks, and final RBLN NPU deployment requires RBLN Profiler validation."
