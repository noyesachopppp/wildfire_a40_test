# C-2 Comparison Summary

| experiment | models used | processed_frames | average_latency_per_frame_ms | average_fps | bottleneck_stage | bottleneck_avg_ms | VLM calls | key interpretation |
|---|---|---:|---:|---:|---|---:|---:|---|
| mock profiling (reference) | mock GroundingDINO + mock SAM2 | N/A | N/A | N/A | visualization/I/O (reference) | N/A | 0 | Mock profiling had no real model inference, so visualization and I/O dominated. |
| real GroundingDINO + SAM2 no-VLM | GroundingDINO + SAM2 | 14 | 2139.5296 | 0.462 | grounding_dino_inference | 729.0004 | 0 | Real DINO->SAM2 linkage succeeded, and grounding_dino_inference was the bottleneck. |
| real GroundingDINO + SAM2 + VLM event-driven | GroundingDINO + SAM2 + VLM | 14 | 1981.9928 | 0.499 | grounding_dino_inference | 536.0281 | 5 | Event-driven VLM path was conditionally invoked (vlm_used_count=5) and fallback behavior was verified. |

## C-2 Interpretation
- Mock profiling에서는 실제 모델 추론이 없어서 visualization/I/O가 병목이었다.
- real no-VLM에서는 GroundingDINO + SAM2 연동이 성공했고, bottleneck은 grounding_dino_inference였다.
- VLM event-driven 실험에서는 모든 프레임이 아니라 이벤트 프레임에서만 VLM 경로를 호출하도록 설계했고, vlm_used_count=5로 조건부 호출이 확인되었다.
- 현재 HF VLM 실추론은 환경 제약으로 fallback이 사용되었으므로, 과제에는 "event-driven VLM invocation path and fallback behavior verified"라고 정확히 기록한다.
- 실서비스 아키텍처에서는 detection worker를 별도 scaling 대상으로 두고, VLM은 MEDIUM/HIGH 이벤트에서만 호출하여 latency와 비용을 줄이는 구조가 적절하다.
