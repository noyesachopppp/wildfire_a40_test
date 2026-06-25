#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="experiments/results/local_docker_yolov8n_vlm"
mkdir -p "$OUT_DIR"

if ! python3 - <<'PY'
import subprocess
try:
    subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20, check=True)
except Exception:
    raise SystemExit(1)
PY
then
  python3 - <<'PY'
import json
from pathlib import Path
p = Path("experiments/results/local_docker_yolov8n_vlm/performance_summary.json")
p.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "environment_name": "local_docker_yolov8n_vlm",
    "status": "NOT_MEASURED",
    "run_location": "local",
    "actual_runtime": "local_docker_no_runtime",
    "container_validation": True,
    "container_runtime": "docker",
    "gpu_available": False,
    "detector_model": "YOLOv8n",
    "detector_backend": "ultralytics",
    "vlm_enabled": True,
    "vlm_event_only": True,
    "vlm_model": "Qwen2.5-VL-7B",
    "vlm_max_new_tokens": 64,
    "force_event_mode": True,
    "note": "Local Docker runtime unavailable or unresponsive; real YOLO GPU detector latency was not measured.",
}
p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Wrote: {p}")
PY
  echo "DONE: $OUT_DIR/performance_summary.json (NOT_MEASURED)"
  exit 0
fi

GPU_AVAILABLE=true
if ! docker run --rm --gpus all wildfire-pipeline:vlm python -c "import torch; print(torch.cuda.is_available())" >/tmp/docker_gpu_check.txt 2>/tmp/docker_gpu_check.err; then
  GPU_AVAILABLE=false
fi

if [ "$GPU_AVAILABLE" != "true" ]; then
  python - <<'PY'
import json
from pathlib import Path
p = Path("experiments/results/local_docker_yolov8n_vlm/performance_summary.json")
p.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "environment_name": "local_docker_yolov8n_vlm",
    "status": "NOT_MEASURED",
    "run_location": "local",
    "actual_runtime": "local_docker_no_gpu",
    "container_validation": True,
    "container_runtime": "docker",
    "gpu_available": False,
    "detector_model": "YOLOv8n",
    "detector_backend": "ultralytics",
    "vlm_enabled": True,
    "vlm_event_only": True,
    "vlm_model": "Qwen2.5-VL-7B",
    "vlm_max_new_tokens": 64,
    "force_event_mode": True,
    "note": "Local Docker is available but GPU passthrough is unavailable; real YOLO GPU detector latency was not measured."
}
p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Wrote: {p}")
PY
  echo "DONE: $OUT_DIR/performance_summary.json (NOT_MEASURED)"
  exit 0
fi

docker run --rm --gpus all \
  -v "$(pwd):/workspace" \
  -w /workspace \
  wildfire-pipeline:vlm \
  python -m src.main \
  --input wildfire_24fps.mp4 \
  --output "$OUT_DIR" \
  --profile \
  --frame-stride 30 \
  --detector yolo \
  --yolo-model yolov8n \
  --yolo-backend ultralytics \
  --enable-vlm \
  --vlm-event-only \
  --vlm-model Qwen2.5-VL-7B \
  --vlm-max-new-tokens 64 \
  --force-medium-high-events \
  --environment-name local_docker_yolov8n_vlm \
  --run-type local_docker_container

python - <<'PY'
import json
from pathlib import Path
p = Path("experiments/results/local_docker_yolov8n_vlm/performance_summary.json")
d = json.loads(p.read_text(encoding="utf-8"))
d["status"] = "SUCCESS"
d["actual_runtime"] = "local_docker_container"
d["run_location"] = "local"
d["container_validation"] = True
d["container_runtime"] = "docker"
d["gpu_available"] = True
p.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Updated: {p}")
PY

echo "DONE: $OUT_DIR/performance_summary.json"
