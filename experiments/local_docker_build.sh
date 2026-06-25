#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

docker --version
python3 - <<'PY'
import subprocess
try:
    subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20, check=True)
    print("DOCKER_INFO_OK")
except Exception:
    raise SystemExit("Docker runtime unavailable or unresponsive (docker info timeout/fail).")
PY
docker build -t wildfire-pipeline:vlm .

echo "DONE: wildfire-pipeline:vlm"
