# Experiment Summary: real GroundingDINO + SAM2 + VLM event-driven

## Experiment
- experiment name: real GroundingDINO + SAM2 + VLM event-driven
- models used: GroundingDINO + SAM2 + VLM(huggingface, event-driven, fallback-capable)
- input video: wildfire_24fps.mp4
- frame_stride: 30

## Performance
- processed_frames: 14
- total_runtime_sec: 28.075
- average_fps: 0.499
- average_latency_per_frame_ms: 1981.9928
- bottleneck_stage: grounding_dino_inference
- bottleneck_avg_ms: 536.0281

## Stage Timing Summary
- frame_load: avg=397.4787 / p50=387.2364 / p90=433.4106 / p95=467.1895 / p99=497.1265
- grounding_dino_inference: avg=536.0281 / p50=349.4477 / p90=369.0020 / p95=1275.5821 / p99=2621.2873
- sam2_inference: avg=440.3981 / p50=427.0399 / p90=554.1603 / p95=671.0598 / p99=803.0921
- vlm_explanation: avg=347.0076 / p50=0.0270 / p90=688.4169 / p95=1274.4167 / p99=2121.2344
- visualization: avg=232.7250 / p50=214.1580 / p90=373.2157 / p95=404.7662 / p99=408.9625
- total_per_frame: avg=1981.9928 / p50=1524.5181 / p90=2222.2178 / p95=3902.2741 / p99=6344.7079

## Alerts Summary
- total records: 14
- LOW/MEDIUM/HIGH counts: LOW=0, MEDIUM=14, HIGH=0
- label distribution: smoke:14
- confidence avg/max: 0.4104 / 0.4344
- mask_area_ratio avg/max: 0.162778 / 0.404221
- vlm_used_count: 5
- vlm_model: Qwen/Qwen2.5-1.5B-Instruct
- fallback detected: True
- note: event-driven VLM invocation path and fallback behavior verified
