#!/usr/bin/env bash
set -euo pipefail

if ! command -v kind >/dev/null 2>&1; then
  echo "kind is not installed. Install kind first."
  exit 1
fi

if ! command -v kubectl >/dev/null 2>&1; then
  echo "kubectl is not installed. Install kubectl first."
  exit 1
fi

kind create cluster --name wildfire-local || true
kubectl cluster-info
echo "DONE: local kind cluster is ready"
