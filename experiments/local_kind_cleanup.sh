#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

kubectl delete -f k8s/output-api-deployment.yaml --ignore-not-found
kubectl delete -f k8s/vlm-worker-deployment.yaml --ignore-not-found
kubectl delete -f k8s/sam2-worker-deployment.yaml --ignore-not-found
kubectl delete -f k8s/event-router-deployment.yaml --ignore-not-found
kubectl delete -f k8s/risk-engine-deployment.yaml --ignore-not-found
kubectl delete -f k8s/detector-deployment.yaml --ignore-not-found
kind delete cluster --name wildfire-local || true
echo "DONE: kind cleanup complete"
