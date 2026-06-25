#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p experiments/results

echo "Python: $(python --version 2>&1)"
echo "CWD: $(pwd)"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not available"
fi

DOCKER_AVAILABLE=false
if command -v docker >/dev/null 2>&1 && docker --version >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  DOCKER_AVAILABLE=true
fi
echo "Docker available: ${DOCKER_AVAILABLE}"

run_step() {
  local name="$1"
  local cmd="$2"
  echo "=== RUN: ${name} ==="
  if bash -lc "$cmd"; then
    echo "=== OK: ${name} ==="
  else
    echo "=== FAIL: ${name} ==="
  fi
}

run_step "runpod_host_mock_vlm" "./experiments/run_runpod_host_mock_vlm.sh"
run_step "runpod_yolov8n_vlm" "./experiments/run_runpod_yolov8n_vlm.sh"
run_step "runpod_yolov8m_vlm" "./experiments/run_runpod_yolov8m_vlm.sh"
run_step "docker_yolov8n_vlm" "./experiments/run_docker_yolov8n_vlm.sh"
run_step "split_service_mock_vlm" "./experiments/run_split_service_mock_vlm.sh"
run_step "aggregate_results" "python experiments/aggregate_results.py"

python - <<'PY'
import json
from pathlib import Path

root = Path("experiments/results")
targets = [
    "runpod_host_mock_vlm",
    "runpod_yolov8n_vlm",
    "runpod_yolov8m_vlm",
    "docker_yolov8n_vlm",
    "split_service_mock_vlm",
]

docker_status = "UNKNOWN"
for name in targets:
    p = root / name / "performance_summary.json"
    if not p.exists():
        continue
    data = json.loads(p.read_text(encoding="utf-8"))
    status = data.get("status", "UNKNOWN")
    bottleneck = data.get("bottleneck_stage")
    if status == "SUCCESS":
        print(f"{name}: vlm_used_count={data.get('vlm_used_count')} bottleneck_stage={bottleneck}")
    if name == "docker_yolov8n_vlm":
        docker_status = status

print(f"Docker experiment status: {docker_status}")
print("Report path: experiments/results/bottleneck_comparison.md")
PY
