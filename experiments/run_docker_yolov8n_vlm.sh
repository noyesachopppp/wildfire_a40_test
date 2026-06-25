#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUT_DIR="experiments/results/docker_yolov8n_vlm"
mkdir -p "$OUT_DIR"
SUMMARY_PATH="$OUT_DIR/performance_summary.json"

if ! command -v docker >/dev/null 2>&1; then
  python - <<'PY'
import json
from pathlib import Path
p = Path("experiments/results/docker_yolov8n_vlm/performance_summary.json")
p.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "environment_name": "docker_yolov8n_vlm",
    "status": "NOT_MEASURED",
    "actual_runtime": "not_measured",
    "docker_available": False,
    "docker_fallback": False,
    "note": "Docker runtime unavailable; true container runtime performance was not measured.",
}
p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Wrote: {p}")
PY
  exit 0
fi

if ! docker --version >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
  python - <<'PY'
import json
from pathlib import Path
p = Path("experiments/results/docker_yolov8n_vlm/performance_summary.json")
p.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "environment_name": "docker_yolov8n_vlm",
    "status": "NOT_MEASURED",
    "actual_runtime": "not_measured",
    "docker_available": False,
    "docker_fallback": False,
    "note": "Docker runtime unavailable; true container runtime performance was not measured.",
}
p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Wrote: {p}")
PY
  exit 0
fi

docker build -t wildfire-pipeline:vlm .
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
  --environment-name docker_yolov8n_vlm \
  --run-type docker_container

python - <<'PY'
import json
from pathlib import Path
p = Path("experiments/results/docker_yolov8n_vlm/performance_summary.json")
payload = json.loads(p.read_text(encoding="utf-8"))
payload["status"] = "SUCCESS"
payload["actual_runtime"] = "docker_container"
payload["docker_available"] = True
payload["docker_fallback"] = False
payload["note"] = payload.get("note", "Measured inside Docker container runtime.")
p.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"Updated: {p}")
PY

echo "DONE: $SUMMARY_PATH"
