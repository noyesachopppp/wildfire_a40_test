# Bottleneck Comparison Report

## Executive Summary
- This validation uses RBLN Model Zoo / RBLN-documented model candidates at the architecture level.
- Current measurements are RunPod host / GPU / mock measurements, not final RBLN NPU runtime measurements.
- Docker is counted only when `actual_runtime=local_docker_container`.
- K8s-style split-service simulation is used for queue/worker operational bottleneck analysis.
- Final deployment still requires RBLN Profiler validation.

## Model Candidates
| Model Candidate | Role | Executed In Current Experiments |
|---|---|---|
| YOLOv8n / YOLOv8m | Detection | Yes |
| SAM2 | Segmentation | Yes |
| Qwen2.5-VL-7B | VLM Explanation | Yes |
| A.X-4.0-Light | Optional LLM Summary | No (architecture candidate only) |

## Experiment Comparison
| Experiment | Status | Actual Runtime | Detector | VLM Included | Frame Load Avg | YOLO Avg | SAM2 Avg | VLM Avg | Queue Wait Avg | E2E p95 | Bottleneck Stage | Note |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| docker_mock | SUCCESS | runpod_host_python | YOLOv8n (mock) | No | 224.4986 | 0.028 | 0.6697 | None | None | 344.9143 | frame_load | Mock mode timings are useful for validating pipeline overhead and stage orchestration. Use real GroundingDINO/SAM2/VLM backends for true model latency benchmarking. |
| local_docker_mock_vlm | NOT_MEASURED | local_docker_no_runtime | YOLOv8n (mock) | Yes | None | None | None | None | None | None | None | Local Docker runtime unavailable or unresponsive; mock container run not measured. |
| local_docker_yolov8n_vlm | NOT_MEASURED | local_docker_no_runtime | YOLOv8n (ultralytics) | Yes | None | None | None | None | None | None | None | Local Docker runtime unavailable or unresponsive; real YOLO GPU detector latency was not measured. |
| local_mock | SUCCESS | runpod_host_python | YOLOv8n (mock) | No | 274.1918 | 0.0262 | 0.6497 | None | None | 432.7488 | frame_load | Mock mode timings are useful for validating pipeline overhead and stage orchestration. Use real GroundingDINO/SAM2/VLM backends for true model latency benchmarking. |
| local_split_service_baseline | SUCCESS | split_service_simulation | YOLOv8n (mock) | Yes | 228.2561 | 0.0219 | 0.612 | 0.0099 | 0.0 | 268.5647 | frame_load | Mock split-service run validates queue wait, worker saturation, backpressure, and event-gated SAM2/VLM path. |
| local_split_service_event_burst | SUCCESS | split_service_simulation | YOLOv8n (mock) | Yes | 227.0897 | 0.0235 | 0.1505 | 0.0057 | 0.2228 | 271.5557 | frame_load | Mock split-service run validates queue wait, worker saturation, backpressure, and event-gated SAM2/VLM path. |
| local_split_service_scaled_workers | SUCCESS | split_service_simulation | YOLOv8n (mock) | Yes | 243.5361 | 0.0246 | 0.2388 | 0.0052 | 0.1066 | 322.6862 | frame_load | Mock split-service run validates queue wait, worker saturation, backpressure, and event-gated SAM2/VLM path. |
| runpod_host_yolov8m_vlm | SUCCESS | runpod_host_python | YOLOv8m (ultralytics) | Yes | 399.9916 | 112.9666 | 0.008 | 0.0111 | None | 1361.1644 | visualization | Timings include active backend inference paths. Forced MEDIUM/HIGH events enabled for bottleneck-load profiling (not accuracy evaluation). |
| runpod_host_yolov8n_vlm | SUCCESS | runpod_host_python | YOLOv8n (ultralytics) | Yes | 418.8401 | 313.4468 | 0.0119 | 0.0136 | None | 2542.7849 | visualization | Timings include active backend inference paths. Forced MEDIUM/HIGH events enabled for bottleneck-load profiling (not accuracy evaluation). |
| runpod_mock | SUCCESS | runpod_host_python | YOLOv8n (mock) | No | 300.5945 | 0.0295 | 0.6381 | None | None | 462.329 | frame_load | Mock mode timings are useful for validating pipeline overhead and stage orchestration. Use real GroundingDINO/SAM2/VLM backends for true model latency benchmarking. |
| runpod_yolov8m | SUCCESS | runpod_host_python | YOLOv8m (ultralytics) | No | 334.0116 | 204.0415 | 0.0002 | None | None | 830.1056 | frame_load | Timings include active backend inference paths. |
| runpod_yolov8n | SUCCESS | runpod_host_python | YOLOv8n (ultralytics) | No | 298.759 | 52.7585 | 0.0002 | None | None | 507.582 | frame_load | Timings include active backend inference paths. |
| split_service_mock | SUCCESS | runpod_host_python | YOLOv8n (mock) | No | 329.0786 | 0.024 | 0.5859 | None | 0.0 | 438.1712 | frame_load | Mock split-service run validates queue wait, worker saturation, and stage instrumentation. |

## Validation Scope Summary
### 1) RunPod GPU Host Baseline
- Purpose: model + pipeline runtime bottleneck in RunPod host environment.
- Metrics focus: frame_load, yolo_inference, sam2_inference, vlm_explanation.
- `runpod_host_yolov8m_vlm`: bottleneck=`visualization`, frame_load_avg_ms=399.9916, yolo_avg_ms=112.9666, vlm_avg_ms=0.0111
- `runpod_host_yolov8n_vlm`: bottleneck=`visualization`, frame_load_avg_ms=418.8401, yolo_avg_ms=313.4468, vlm_avg_ms=0.0136
- `runpod_mock`: bottleneck=`frame_load`, frame_load_avg_ms=300.5945, yolo_avg_ms=0.0295, vlm_avg_ms=None
- `runpod_yolov8m`: bottleneck=`frame_load`, frame_load_avg_ms=334.0116, yolo_avg_ms=204.0415, vlm_avg_ms=None
- `runpod_yolov8n`: bottleneck=`frame_load`, frame_load_avg_ms=298.759, yolo_avg_ms=52.7585, vlm_avg_ms=None

### 2) Local Docker Container Validation
- Purpose: container reproducibility and container I/O overhead locally.
- Interpretation rule: do not compare absolute latency with RunPod because hardware/storage differ.
- `local_docker_mock_vlm`: status=`NOT_MEASURED`, actual_runtime=`local_docker_no_runtime`, container bottleneck=`None`
- `local_docker_yolov8n_vlm`: status=`NOT_MEASURED`, actual_runtime=`local_docker_no_runtime`, container bottleneck=`None`

### 3) Local K8s-style Operational Simulation
- Purpose: queue/worker/backpressure bottleneck under split-service separation.
- Metrics focus: queue_wait, queue_depth, worker utilization, timeout/fallback.
- `local_split_service_baseline`: queue_wait_avg_ms=0.0, bottleneck=`frame_load`
- `local_split_service_event_burst`: queue_wait_avg_ms=0.2228, bottleneck=`frame_load`
- `local_split_service_scaled_workers`: queue_wait_avg_ms=0.1066, bottleneck=`frame_load`
- `split_service_mock`: queue_wait_avg_ms=0.0, bottleneck=`frame_load`

## Bottleneck Interpretation
- **docker_mock**: Primary bottleneck: frame loading / decoding / resize I/O. Response: async prefetch, separate decode pipeline, frame buffer, and storage I/O optimization.
- **local_docker_mock_vlm**: No dominant bottleneck rule triggered from current metrics.
- **local_docker_yolov8n_vlm**: No dominant bottleneck rule triggered from current metrics.
- **local_mock**: Primary bottleneck: frame loading / decoding / resize I/O. Response: async prefetch, separate decode pipeline, frame buffer, and storage I/O optimization.
- **local_split_service_baseline**: Primary bottleneck: frame loading / decoding / resize I/O. Response: async prefetch, separate decode pipeline, frame buffer, and storage I/O optimization.
- **local_split_service_event_burst**: Primary bottleneck: frame loading / decoding / resize I/O. Response: async prefetch, separate decode pipeline, frame buffer, and storage I/O optimization.
- **local_split_service_scaled_workers**: Primary bottleneck: frame loading / decoding / resize I/O. Response: async prefetch, separate decode pipeline, frame buffer, and storage I/O optimization.
- **runpod_host_yolov8m_vlm**: Primary bottleneck: frame loading / decoding / resize I/O. Response: async prefetch, separate decode pipeline, frame buffer, and storage I/O optimization.
- **runpod_host_yolov8n_vlm**: Primary bottleneck: frame loading / decoding / resize I/O. Response: async prefetch, separate decode pipeline, frame buffer, and storage I/O optimization.
- **runpod_mock**: Primary bottleneck: frame loading / decoding / resize I/O. Response: async prefetch, separate decode pipeline, frame buffer, and storage I/O optimization.
- **runpod_yolov8m**: Primary bottleneck: frame loading / decoding / resize I/O. Response: async prefetch, separate decode pipeline, frame buffer, and storage I/O optimization.
- **runpod_yolov8n**: Primary bottleneck: frame loading / decoding / resize I/O. Response: async prefetch, separate decode pipeline, frame buffer, and storage I/O optimization.
- **split_service_mock**: Primary bottleneck: frame loading / decoding / resize I/O. Response: async prefetch, separate decode pipeline, frame buffer, and storage I/O optimization.

## Mitigation Strategy
- I/O bottleneck (`frame_load`) dominates: prioritize async prefetch, decode pipeline split, and storage tuning.
- Detector bottleneck dominates: keep YOLOv8n as edge default and tune resolution/stride before upscaling model.
- VLM bottleneck dominates: keep MEDIUM/HIGH event-gating, cap max tokens, enforce timeout/fallback.
- Queue bottleneck dominates: scale SAM2/VLM workers and apply priority/backpressure control.

## Profiling Notes
- Forced MEDIUM/HIGH event mode is for load/bottleneck profiling only and not for accuracy evaluation.
- RunPod and local Docker latency values should not be interpreted as absolute cross-environment comparisons.

## FDE Deployment Interpretation
- RunPod host experiments identify model/I/O bottlenecks under current host/GPU assumptions.
- Docker container experiments identify container reproducibility and container I/O overhead only when actual runtime is true container execution.
- Split-service/K8s-style simulation identifies queue wait, worker saturation, and backpressure risks.
- Final RBLN NPU deployment requires RBLN Profiler to verify p50/p95/p99 latency and bottleneck movement.
