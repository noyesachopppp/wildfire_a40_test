# YOLO 전환 연동/병목 점검 보고서

## 1) 결론 (요약)
- YOLO 기반 1차 검출 경로 연동은 완료됨.
- 파이프라인은 `Input -> YOLO -> Risk Engine -> Event Router -> SAM2(이벤트시) -> VLM -> Output` 순서로 동작 확인됨.
- 이번 로컬 실행에서는 SAM2 실가중치 경로가 없어 `SAM2 mock`으로 이벤트 경로 검증함.
- 병목은 GroundingDINO 실험과 달리 detector가 아니라 `frame_load`로 이동함.

## 2) 스테이지별 연동 확인 (YOLO mock run 기준)
- 근거 파일:
  - `outputs_yolo_mock/alerts.json`
  - `outputs_yolo_mock/performance_summary.json`

| 스테이지 | 연동 상태 | 확인 근거 |
|---|---|---|
| YOLO detector | 완료 | `detector_type=yolo`, `detector_model=YOLOv8n`, `detector_backend=mock` 기록됨 |
| Risk Engine | 완료 | `risk_level`이 LOW/MEDIUM/HIGH로 분기되어 기록됨 |
| Event Router | 완료 | `event_triggered` 필드와 `event_count=8`로 이벤트 라우팅 확인 |
| SAM2 (event-triggered) | 완료(조건부 호출) | `sam2_used=true/false` 혼재, 요약에서 `sam2_used_count=8` |
| VLM | 완료(조건부 호출) | `vlm_used=true/false` 혼재, 요약에서 `vlm_used_count=8` |
| Output/API용 레코드 | 완료 | `alerts.json`에 detector/risk/event/sam2/vlm 메타 모두 기록 |

## 3) 병목 구간 확인

### A. YOLO mock (아키텍처 검증 실행)
- 파일: `outputs_yolo_mock/performance_summary.json`
- `processed_frames`: 14
- `bottleneck_stage`: `frame_load`
- `bottleneck_avg_ms`: 256.1082
- `yolo_inference.avg_ms`: 0.0257
- `sam2_inference.avg_ms`: 0.6168
- `vlm_explanation.avg_ms`: 0.0274
- `event_count/sam2_used_count/vlm_used_count`: 8 / 8 / 8

### B. YOLO ultralytics (yolov8n)
- 파일: `outputs_yolo_ultralytics_n/performance_summary.json`
- `processed_frames`: 14
- `bottleneck_stage`: `frame_load`
- `bottleneck_avg_ms`: 250.0541
- `yolo_inference.avg_ms`: 54.0243
- `sam2_inference.avg_ms`: 0.0002
- `vlm_explanation.avg_ms`: 0.0050
- `event_count/sam2_used_count/vlm_used_count`: 0 / 0 / 0
- 해석: COCO pretrained 기반으로 smoke/fire 매핑 결과가 없어 이벤트 경로가 거의 비활성화됨.

### C. GroundingDINO 기존 실험(비교 기준)
- 파일: `experiments/runpod_a40/01_real_dino_sam2_no_vlm/performance_summary.json`
- `processed_frames`: 14
- `bottleneck_stage`: `grounding_dino_inference`
- `bottleneck_avg_ms`: 729.0004
- `grounding_dino_inference.avg_ms`: 729.0004
- `sam2_inference.avg_ms`: 455.9107

## 4) 비교 요약표
| 실험 | detector stage avg_ms | bottleneck_stage | bottleneck_avg_ms | 평균 FPS | 비고 |
|---|---:|---|---:|---:|---|
| GroundingDINO + SAM2 (no-VLM) | 729.0004 | grounding_dino_inference | 729.0004 | 0.4618 | 기존 실험 기준 detector가 주 병목 |
| YOLOv8n mock | 0.0257 | frame_load | 256.1082 | 2.6843 | 구조 검증용 mock, 이벤트 경로 활성(8회) |
| YOLOv8n ultralytics | 54.0243 | frame_load | 250.0541 | 2.4522 | COCO class 한계로 이벤트 경로 비활성(0회) |

## 5) 현재 상태 판단
- 연동 자체는 완료:
  - YOLO detector 어댑터 추가
  - CLI 선택자(`--detector`, `--yolo-model`, `--yolo-backend`) 추가
  - 이벤트 라우팅 + SAM2/VLM 조건부 실행 + 프로파일 확장 완료
- 정확도 관점의 실전 검증은 별도:
  - 실제 산불 검출 성능 검증은 smoke/fire 파인튜닝 YOLO 가중치와 데이터셋이 필요
  - 현재 ultralytics 실행은 지연/아키텍처 경로 검증 용도
