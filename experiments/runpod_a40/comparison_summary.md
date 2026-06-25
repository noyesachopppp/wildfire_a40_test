# C-2 Comparison Summary (FDE Bottleneck Validation)

## Scope
- Model candidates follow RBLN model-zoo documented targets at architecture level: `YOLOv8n/v8m`, `SAM2`, `Qwen2.5-VL-7B`, optional `A.X-4.0-Light`.
- Current numbers are RunPod host/GPU/mock measurements, not final RBLN NPU runtime measurements.
- Docker is counted only if `actual_runtime=local_docker_container`.

## Experiment Table
| experiment | status | actual_runtime | detector | VLM event-gated | processed_frames | avg_latency_ms | avg_fps | frame_load_avg_ms | yolo_avg_ms | sam2_avg_ms | vlm_avg_ms | queue_wait_avg_ms | bottleneck_stage | note |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| runpod_host_yolov8n_vlm | SUCCESS | runpod_host_python | YOLOv8n(ultralytics) | yes | 14 | 1203.1857 | 0.8143 | 418.8401 | 313.4468 | 0.0119 | 0.0136 | null | visualization | forced MEDIUM/HIGH events for bottleneck profiling; RunPod host scope |
| runpod_host_yolov8m_vlm | SUCCESS | runpod_host_python | YOLOv8m(ultralytics) | yes | 14 | 987.2518 | 0.9878 | 399.9916 | 112.9666 | 0.0080 | 0.0111 | null | visualization | forced MEDIUM/HIGH events for bottleneck profiling; RunPod host scope |
| local_docker_mock_vlm | NOT_MEASURED | local_docker_no_runtime | n/a | n/a | null | null | null | null | null | null | null | null | null | local Docker runtime unavailable/unresponsive |
| local_docker_yolov8n_vlm | NOT_MEASURED | local_docker_no_runtime | n/a | n/a | null | null | null | null | null | null | null | null | null | local Docker runtime unavailable/unresponsive; GPU passthrough not measured |
| local_split_service_baseline | SUCCESS | split_service_simulation | YOLOv8n(mock) | yes | 14 | 240.5970 | 3.9571 | 228.2561 | 0.0219 | 0.6120 | 0.0099 | 0.0 | frame_load | baseline split-service queue/worker simulation |
| local_split_service_event_burst | SUCCESS | split_service_simulation | YOLOv8n(mock) | yes | 14 | 235.7675 | 4.0284 | 227.0897 | 0.0235 | 0.1505 | 0.0057 | 0.2228 | frame_load | forced event burst (size=10), 1x SAM2 + 1x VLM worker |
| local_split_service_scaled_workers | SUCCESS | split_service_simulation | YOLOv8n(mock) | yes | 14 | 252.7619 | 3.7614 | 243.5361 | 0.0246 | 0.2388 | 0.0052 | 0.1066 | frame_load | same burst with scaled workers (2x SAM2 + 2x VLM), queue wait reduced |

## FDE Interpretation
- RunPod host scope confirms model+pipeline bottlenecks under GPU host assumptions (current top stage: `visualization` with high frame I/O load).
- Local Docker scope is explicitly `NOT_MEASURED`; no container runtime number is used for bottleneck comparison.
- Local split-service scope shows operational behavior under burst: `queue_wait_avg_ms` drops from `0.2228` to `0.1066` when workers scale from 1/1 to 2/2.
- Forced MEDIUM/HIGH event mode is used for load/bottleneck profiling only, not for model accuracy evaluation.
- Final RBLN NPU deployment still requires RBLN Profiler (`p50/p95/p99`, static shape, graph-break, host-device transfer) before absolute conclusions.
