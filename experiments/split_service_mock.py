from __future__ import annotations

import argparse
import json
import os
import time
import heapq
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.input.video_loader import sample_video_frames
from src.models.sam2_wrapper import SAM2Wrapper
from src.models.vlm_explainer import VLMExplainer
from src.models.yolo_detector import YOLODetector
from src.pipeline.postprocessing import compute_growth, count_regions, mask_area_ratio, merge_masks
from src.pipeline.preprocessing import preprocess_frame
from src.pipeline.risk_engine import RuleBasedRiskEngine


@dataclass
class EventItem:
    frame_id: int
    timestamp: float
    frame: np.ndarray
    detections: list[dict[str, Any]]
    risk_level: str
    avg_conf: float
    enqueue_time: float
    consecutive_frames: int


def _stage_stats(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return {"avg_ms": 0.0, "p95_ms": 0.0}
    return {"avg_ms": float(arr.mean()), "p95_ms": float(np.percentile(arr, 95))}


def _pick_bottleneck(metrics: dict[str, Any]) -> str:
    stage_pairs = [
        ("frame_load", metrics["frame_load_avg_ms"]),
        ("preprocess", metrics["preprocess_avg_ms"]),
        ("yolo_inference", metrics["yolo_inference_avg_ms"]),
        ("risk_engine", metrics["risk_engine_avg_ms"]),
        ("event_router", metrics["event_router_avg_ms"]),
        ("queue_wait", metrics["queue_wait_avg_ms"]),
        ("sam2_inference", metrics["sam2_inference_avg_ms"]),
        ("vlm_explanation", metrics["vlm_explanation_avg_ms"]),
        ("output_write", metrics["output_write_avg_ms"]),
    ]
    valid = [(k, v) for k, v in stage_pairs if isinstance(v, (int, float))]
    if not valid:
        return "n/a"
    return max(valid, key=lambda kv: kv[1])[0]


def _acquire_worker(worker_heap: list[float], event_time: float) -> tuple[float, int]:
    available_at, idx = heapq.heappop(worker_heap)
    start_at = max(event_time, available_at)
    return start_at, idx


def _release_worker(worker_heap: list[float], finish_time: float, idx: int) -> None:
    heapq.heappush(worker_heap, (finish_time, idx))


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.makedirs(args.output, exist_ok=True)

    detector = YOLODetector(model_name="yolov8n", backend="mock", conf_threshold=0.35, device=None)
    risk_engine = RuleBasedRiskEngine(
        detection_confidence=0.35,
        high_risk_confidence=0.65,
        min_mask_area_ratio=0.005,
        high_growth_ratio=0.2,
    )
    sam2 = SAM2Wrapper(mock=True)
    vlm = VLMExplainer(enabled=True, backend="placeholder", model_name_or_path="mock-vlm", selected_frame_interval=1)

    frame_load_ms: list[float] = []
    preprocess_ms: list[float] = []
    yolo_ms: list[float] = []
    risk_ms: list[float] = []
    event_router_ms: list[float] = []
    queue_wait_ms: list[float] = []
    sam2_ms: list[float] = []
    vlm_ms: list[float] = []
    output_write_ms: list[float] = []
    end_to_end_ms: list[float] = []

    alerts: list[dict[str, Any]] = []
    event_items: list[EventItem] = []

    previous_area_ratio: float | None = None
    consecutive_detected = 0
    event_count = 0
    sam2_used_count = 0
    vlm_used_count = 0
    sam2_worker_busy_ms_total = 0.0
    vlm_worker_busy_ms_total = 0.0
    queue_depth_max = 0
    timeout_count = 0
    fallback_count = 0

    sam2_worker_heap = [(time.perf_counter(), idx) for idx in range(max(1, args.sam2_worker_count))]
    vlm_worker_heap = [(time.perf_counter(), idx) for idx in range(max(1, args.vlm_worker_count))]
    heapq.heapify(sam2_worker_heap)
    heapq.heapify(vlm_worker_heap)

    run_start = time.perf_counter()
    frame_iter = sample_video_frames(args.input, frame_stride=args.frame_stride)
    while True:
        frame_start = time.perf_counter()

        t0 = time.perf_counter()
        try:
            packet = next(frame_iter)
        except StopIteration:
            break
        t1 = time.perf_counter()
        frame_load_ms.append((t1 - t0) * 1000.0)

        t0 = time.perf_counter()
        frame = preprocess_frame(packet.image_bgr)
        t1 = time.perf_counter()
        preprocess_ms.append((t1 - t0) * 1000.0)

        t0 = time.perf_counter()
        detections = detector.detect(frame=frame, frame_id=packet.frame_id, timestamp=packet.timestamp)
        t1 = time.perf_counter()
        yolo_ms.append((t1 - t0) * 1000.0)

        if detections:
            consecutive_detected += 1
        else:
            consecutive_detected = 0
        avg_conf = float(sum(float(d["score"]) for d in detections) / len(detections)) if detections else 0.0

        t0 = time.perf_counter()
        risk_level, alert_message = risk_engine.evaluate(
            avg_confidence=avg_conf,
            mask_area_ratio=0.0,
            mask_growth=0.0,
            consecutive_frames=consecutive_detected,
            region_count=0,
        )
        t1 = time.perf_counter()
        risk_ms.append((t1 - t0) * 1000.0)

        t0 = time.perf_counter()
        event_triggered = risk_level in {"MEDIUM", "HIGH"} and len(detections) > 0
        t1 = time.perf_counter()
        event_router_ms.append((t1 - t0) * 1000.0)

        area_ratio = 0.0
        growth = 0.0
        sam2_used = False
        vlm_used = False
        explanation = vlm.fallback_explanation(
            detections=detections,
            mask_area_ratio=0.0,
            mask_growth=0.0,
            consecutive_frames=consecutive_detected,
            risk_level=risk_level,
        )

        if args.force_event_mode:
            event_triggered = True
            if risk_level == "LOW":
                risk_level = "MEDIUM"
                alert_message = "Forced MEDIUM event for split-service bottleneck profiling."

        if event_triggered:
            burst = max(1, int(args.event_burst_size)) if args.force_event_mode else 1
            for _ in range(burst):
                event_count += 1
                event_items.append(
                    EventItem(
                        frame_id=packet.frame_id,
                        timestamp=packet.timestamp,
                        frame=frame,
                        detections=detections,
                        risk_level=risk_level,
                        avg_conf=avg_conf,
                        enqueue_time=time.perf_counter(),
                        consecutive_frames=consecutive_detected,
                    )
                )
            queue_depth_max = max(queue_depth_max, len(event_items))

        # Process only one event per frame tick to emulate queue accumulation under burst.
        if event_items:
            evt = event_items.pop(0)
            sam2_start, sam2_idx = _acquire_worker(sam2_worker_heap, evt.enqueue_time)
            sam2_wait = max(0.0, sam2_start - evt.enqueue_time)

            t0 = time.perf_counter()
            masks = sam2.segment(evt.frame, evt.detections, evt.frame_id)
            t1 = time.perf_counter()
            sam2_infer_ms = (t1 - t0) * 1000.0
            sam2_finish = sam2_start + (sam2_infer_ms / 1000.0)
            _release_worker(sam2_worker_heap, sam2_finish, sam2_idx)
            sam2_ms.append(sam2_infer_ms)
            sam2_worker_busy_ms_total += sam2_infer_ms
            sam2_used = True
            sam2_used_count += 1

            if masks:
                merged_mask = merge_masks(masks, image_shape=evt.frame.shape[:2])
                area_ratio = mask_area_ratio(merged_mask)
                growth = compute_growth(area_ratio, previous_area_ratio)
                previous_area_ratio = area_ratio
                region_count = count_regions(merged_mask)
            else:
                region_count = 0
                merged_mask = None

            t0 = time.perf_counter()
            risk_level, alert_message = risk_engine.evaluate(
                avg_confidence=evt.avg_conf,
                mask_area_ratio=area_ratio,
                mask_growth=growth,
                consecutive_frames=evt.consecutive_frames,
                region_count=region_count,
            )
            t1 = time.perf_counter()
            risk_ms.append((t1 - t0) * 1000.0)

            vlm_start, vlm_idx = _acquire_worker(vlm_worker_heap, sam2_finish)
            vlm_wait = max(0.0, vlm_start - sam2_finish)
            total_wait_ms = (sam2_wait + vlm_wait) * 1000.0
            queue_wait_ms.append(total_wait_ms)

            if total_wait_ms > args.timeout_ms:
                timeout_count += 1
                fallback_count += 1
                explanation = vlm.fallback_explanation(
                    detections=evt.detections,
                    mask_area_ratio=area_ratio,
                    mask_growth=growth,
                    consecutive_frames=evt.consecutive_frames,
                    risk_level=risk_level,
                )
                vlm_infer_ms = 0.0
                vlm_used = False
            else:
                t0 = time.perf_counter()
                explanation = vlm.explain(
                    detections=evt.detections,
                    mask_area_ratio=area_ratio,
                    mask_growth=growth,
                    consecutive_frames=evt.consecutive_frames,
                    risk_level=risk_level,
                    frame=evt.frame,
                    merged_mask=merged_mask,
                )
                t1 = time.perf_counter()
                vlm_infer_ms = (t1 - t0) * 1000.0
                vlm_worker_busy_ms_total += vlm_infer_ms
                vlm_used = True
                vlm_used_count += 1
            vlm_finish = vlm_start + (vlm_infer_ms / 1000.0)
            _release_worker(vlm_worker_heap, vlm_finish, vlm_idx)
            vlm_ms.append(vlm_infer_ms)
        else:
            sam2_ms.append(0.0)
            vlm_ms.append(0.0)
            queue_wait_ms.append(0.0)

        t0 = time.perf_counter()
        alerts.append(
            {
                "frame_id": packet.frame_id,
                "timestamp": round(packet.timestamp, 3),
                "detector_type": "yolo",
                "detector_model": "YOLOv8n",
                "detector_backend": "mock",
                "detected_labels": sorted(set(str(d["label"]) for d in detections)),
                "confidence": round(avg_conf, 4),
                "mask_area_ratio": round(area_ratio, 6),
                "mask_growth": round(growth, 6),
                "risk_level": risk_level,
                "event_triggered": event_triggered,
                "sam2_used": sam2_used,
                "vlm_used": vlm_used,
                "vlm_model": "Qwen2.5-VL-7B",
                "explanation": explanation,
                "fallback_used": bool(vlm.last_fallback_used) or not vlm_used,
                "alert_message": alert_message,
            }
        )
        t1 = time.perf_counter()
        output_write_ms.append((t1 - t0) * 1000.0)
        end_to_end_ms.append((time.perf_counter() - frame_start) * 1000.0)

    total_runtime_ms = (time.perf_counter() - run_start) * 1000.0
    processed_frames = len(alerts)
    average_fps = (processed_frames * 1000.0 / total_runtime_ms) if total_runtime_ms > 0 else 0.0

    def avg_or_null(values: list[float]) -> float | None:
        if not values:
            return None
        return float(np.mean(np.array(values, dtype=np.float64)))

    queue_wait_summary = _stage_stats(queue_wait_ms)
    sam2_worker_utilization = (
        sam2_worker_busy_ms_total / max(1.0, total_runtime_ms * float(max(1, args.sam2_worker_count)))
    )
    vlm_worker_utilization = (
        vlm_worker_busy_ms_total / max(1.0, total_runtime_ms * float(max(1, args.vlm_worker_count)))
    )

    metrics: dict[str, Any] = {
        "environment_name": args.environment_name,
        "status": "SUCCESS",
        "run_location": "local",
        "actual_runtime": "split_service_simulation",
        "container_validation": False,
        "container_runtime": None,
        "run_type": "split_service_mock_vlm",
        "detector_model": "YOLOv8n",
        "detector_backend": "mock",
        "model_zoo_candidates": {
            "detection": ["YOLOv8n", "YOLOv8m"],
            "segmentation": ["SAM2"],
            "vlm_explanation": ["Qwen2.5-VL-7B"],
            "optional_llm_summary": ["A.X-4.0-Light"],
        },
        "vlm_enabled": True,
        "vlm_event_only": True,
        "vlm_model": "Qwen2.5-VL-7B",
        "vlm_max_new_tokens": 64,
        "force_event_mode": bool(args.force_event_mode),
        "processed_frames": processed_frames,
        "total_runtime_ms": round(total_runtime_ms, 4),
        "avg_latency_per_frame_ms": round(avg_or_null(end_to_end_ms) or 0.0, 4),
        "average_fps": round(average_fps, 4),
        "frame_load_avg_ms": round(avg_or_null(frame_load_ms) or 0.0, 4),
        "preprocess_avg_ms": round(avg_or_null(preprocess_ms) or 0.0, 4),
        "yolo_inference_avg_ms": round(avg_or_null(yolo_ms) or 0.0, 4),
        "risk_engine_avg_ms": round(avg_or_null(risk_ms) or 0.0, 4),
        "event_router_avg_ms": round(avg_or_null(event_router_ms) or 0.0, 4),
        "queue_wait_avg_ms": round(avg_or_null(queue_wait_ms) or 0.0, 4),
        "queue_wait_p95_ms": round(queue_wait_summary["p95_ms"], 4),
        "sam2_inference_avg_ms": round(avg_or_null(sam2_ms) or 0.0, 4),
        "vlm_explanation_avg_ms": round(avg_or_null(vlm_ms) or 0.0, 4),
        "vlm_inference_avg_ms": round(avg_or_null(vlm_ms) or 0.0, 4),
        "output_write_avg_ms": round(avg_or_null(output_write_ms) or 0.0, 4),
        "end_to_end_avg_ms": round(avg_or_null(end_to_end_ms) or 0.0, 4),
        "end_to_end_p95_ms": round(_stage_stats(end_to_end_ms)["p95_ms"], 4),
        "event_count": event_count,
        "sam2_used_count": sam2_used_count,
        "vlm_used_count": vlm_used_count,
        "bottleneck_stage": None,
        "worker_busy_time_ms": {
            "sam2_worker_busy_ms": round(sam2_worker_busy_ms_total, 4),
            "vlm_worker_busy_ms": round(vlm_worker_busy_ms_total, 4),
        },
        "sam2_worker_utilization": round(float(sam2_worker_utilization), 6),
        "vlm_worker_utilization": round(float(vlm_worker_utilization), 6),
        "queue_depth_max": queue_depth_max,
        "timeout_count": timeout_count,
        "fallback_count": fallback_count,
        "note": "Mock split-service run validates queue wait, worker saturation, backpressure, and event-gated SAM2/VLM path.",
    }
    metrics["notes"] = metrics["note"]
    metrics["bottleneck_stage"] = _pick_bottleneck(metrics)

    with open(os.path.join(args.output, "alerts.json"), "w", encoding="utf-8") as f:
        json.dump(alerts, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output, "performance_summary.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Split-service mock bottleneck simulation")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output result directory")
    parser.add_argument("--frame-stride", type=int, default=30)
    parser.add_argument("--environment-name", default="split_service_mock_vlm")
    parser.add_argument("--sam2-worker-count", type=int, default=1)
    parser.add_argument("--vlm-worker-count", type=int, default=1)
    parser.add_argument("--force-event-mode", action="store_true")
    parser.add_argument("--event-burst-size", type=int, default=1)
    parser.add_argument("--timeout-ms", type=float, default=2000.0)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

