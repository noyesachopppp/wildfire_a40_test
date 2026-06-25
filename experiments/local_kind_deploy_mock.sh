#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

kubectl apply -f k8s/detector-deployment.yaml
kubectl apply -f k8s/risk-engine-deployment.yaml
kubectl apply -f k8s/event-router-deployment.yaml
kubectl apply -f k8s/sam2-worker-deployment.yaml
kubectl apply -f k8s/vlm-worker-deployment.yaml
kubectl apply -f k8s/output-api-deployment.yaml
kubectl get pods
echo "DONE: mock split-service manifests deployed"
