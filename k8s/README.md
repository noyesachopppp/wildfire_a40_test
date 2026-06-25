# Mock K8s Service Split

This directory provides a minimal mock-only decomposition of the wildfire architecture:

- detector
- risk engine
- event router
- sam2 worker
- vlm worker
- output api

These manifests are for service separation and operational bottleneck modeling
(queue wait, worker saturation, service overhead), not full GPU inference deployment.

Important scope notes:

- This validates split-service design and operational bottleneck behavior.
- It does **not** validate real GPU inference runtime.
- It does **not** validate final RBLN NPU runtime.
- Real GPU/RBLN serving should be validated later with dedicated runtime profiling.

## Apply on kind/minikube

```bash
kubectl apply -f k8s/detector-deployment.yaml
kubectl apply -f k8s/risk-engine-deployment.yaml
kubectl apply -f k8s/event-router-deployment.yaml
kubectl apply -f k8s/sam2-worker-deployment.yaml
kubectl apply -f k8s/vlm-worker-deployment.yaml
kubectl apply -f k8s/output-api-deployment.yaml
```

## Verify pods

```bash
kubectl get pods
```

## Cleanup

```bash
kubectl delete -f k8s/output-api-deployment.yaml
kubectl delete -f k8s/vlm-worker-deployment.yaml
kubectl delete -f k8s/sam2-worker-deployment.yaml
kubectl delete -f k8s/event-router-deployment.yaml
kubectl delete -f k8s/risk-engine-deployment.yaml
kubectl delete -f k8s/detector-deployment.yaml
```
