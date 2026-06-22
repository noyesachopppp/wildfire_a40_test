# Experiment Summary: real GroundingDINO + SAM2, no VLM

## Experiment
- experiment name: real GroundingDINO + SAM2, no VLM
- models used: GroundingDINO + SAM2
- input video: wildfire_24fps.mp4
- frame_stride: 30

## Performance
- processed_frames: 14
- total_runtime_sec: 30.317
- average_fps: 0.462
- average_latency_per_frame_ms: 2139.5296
- bottleneck_stage: grounding_dino_inference
- bottleneck_avg_ms: 729.0004

## Stage Timing Summary
- frame_load: avg=465.7957 / p50=461.8081 / p90=510.0197 / p95=525.0798 / p99=525.6999
- grounding_dino_inference: avg=729.0004 / p50=404.9465 / p90=467.6620 / p95=2008.7908 / p99=4298.2248
- sam2_inference: avg=455.9107 / p50=370.6152 / p90=555.6598 / p95=820.0216 / p99=1185.9123
- visualization: avg=444.4836 / p50=434.6125 / p90=533.8181 / p95=565.8288 / p99=575.0638
- total_per_frame: avg=2139.5296 / p50=1721.4271 / p90=1951.9767 / p95=3846.4489 / p99=6656.0468

## Alerts Summary
- total records: 14
- LOW/MEDIUM/HIGH counts: LOW=0, MEDIUM=14, HIGH=0
- label distribution: smoke:14
- confidence avg/max: 0.4104 / 0.4344
- mask_area_ratio avg/max: 0.162778 / 0.404221
