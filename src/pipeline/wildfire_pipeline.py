from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from tqdm import tqdm

from src.input.video_loader import sample_video_frames
from src.models.grounding_dino_wrapper import GroundingDINOWrapper
from src.models.sam2_wrapper import SAM2Wrapper
from src.models.vlm_explainer import VLMExplainer
from src.models.yolo_detector import YOLODetector
from src.pipeline.postprocessing import (
    compute_growth,
    count_regions,
    filter_detections,
    mask_area_ratio,
    merge_masks,
)
from src.pipeline.preprocessing import preprocess_frame
from src.pipeline.risk_engine import RuleBasedRiskEngine
from src.utils.logging_utils import get_logger
from src.utils.visualization import draw_overlay, save_image


@dataclass
class PipelineConfig:
    prompts: list[str]
    frame_stride: int
    detection_confidence: float
    high_risk_confidence: float
    min_mask_area_ratio: float
    high_growth_ratio: float
    box_thickness: int
    mask_alpha: float
    grounding_checkpoint: str | None = None
    grounding_config: str | None = None
    grounding_box_threshold: float = 0.35
    grounding_text_threshold: float = 0.25
    sam2_checkpoint: str | None = None
    sam2_model_config: str | None = None
    model_device: str | None = None
    vlm_enabled: bool = True
    vlm_backend: str = "placeholder"
    vlm_model_name_or_path: str | None = None
    vlm_device: str | None = None
    vlm_torch_dtype: str = "float16"
    vlm_load_in_4bit: bool = False
    vlm_max_new_tokens: int = 64
    vlm_only_on_risk_levels: list[str] | None = None
    vlm_selected_frame_interval: int = 1
    detector_type: str = "yolo"
    yolo_model: str = "yolov8n"
    yolo_backend: str = "mock"
    yolo_conf_threshold: float = 0.35
    environment_name: str = "local_mock"
    run_type: str = "single_process"
    force_event_mode: bool = False


class WildfirePipeline:
    @staticmethod
    def _infer_run_location(environment_name: str) -> str:
        name = str(environment_name).lower()
        if name.startswith("runpod"):
            return "runpod"
        if name.startswith("local"):
            return "local"
        return "unknown"

    @staticmethod
    def _infer_actual_runtime(run_type: str) -> str:
        lowered = str(run_type).lower()
        if "docker" in lowered and "fallback" not in lowered:
            return "local_docker_container"
        if "split_service" in lowered or "split-service" in lowered:
            return "split_service_simulation"
        return "runpod_host_python"

    @staticmethod
    def _is_unset_path(path: str | None) -> bool:
        if path is None:
            return True
        normalized = str(path).strip()
        return not normalized or normalized.upper().startswith("TODO:")

    def __init__(self, cfg: PipelineConfig, mock: bool = False):
        self.cfg = cfg
        self.logger = get_logger(self.__class__.__name__)
        self.detector_type = str(cfg.detector_type).lower()
        self.yolo = None
        self.grounding = None
        if self.detector_type == "yolo":
            self.yolo = YOLODetector(
                model_name=cfg.yolo_model,
                backend=cfg.yolo_backend,
                conf_threshold=cfg.yolo_conf_threshold,
                device=cfg.model_device,
            )
        elif self.detector_type == "groundingdino":
            self.grounding = GroundingDINOWrapper(
                mock=mock,
                checkpoint=cfg.grounding_checkpoint,
                config_path=cfg.grounding_config,
                box_threshold=cfg.grounding_box_threshold,
                text_threshold=cfg.grounding_text_threshold,
                device=cfg.model_device,
            )
        else:
            raise ValueError(f"Unsupported detector_type: {cfg.detector_type}")
        sam2_mock = mock
        if not sam2_mock and (
            self._is_unset_path(cfg.sam2_checkpoint)
            or self._is_unset_path(cfg.sam2_model_config)
            or not os.path.exists(str(cfg.sam2_checkpoint))
            or not os.path.exists(str(cfg.sam2_model_config))
        ):
            sam2_mock = True
            self.logger.warning(
                "SAM2 weights/config are unavailable. Using SAM2 mock backend for event path. "
                "Set models.sam2.checkpoint/model_config for real segmentation."
            )

        self.sam2 = SAM2Wrapper(
            mock=sam2_mock,
            checkpoint=cfg.sam2_checkpoint,
            model_config=cfg.sam2_model_config,
            device=cfg.model_device,
        )
        self.vlm = VLMExplainer(
            mode=cfg.vlm_backend,
            enabled=cfg.vlm_enabled,
            backend=cfg.vlm_backend,
            model_name_or_path=cfg.vlm_model_name_or_path,
            device=cfg.vlm_device or cfg.model_device,
            torch_dtype=cfg.vlm_torch_dtype,
            load_in_4bit=cfg.vlm_load_in_4bit,
            max_new_tokens=cfg.vlm_max_new_tokens,
            only_on_risk_levels=cfg.vlm_only_on_risk_levels,
            selected_frame_interval=cfg.vlm_selected_frame_interval,
        )
        self.risk_engine = RuleBasedRiskEngine(
            detection_confidence=cfg.detection_confidence,
            high_risk_confidence=cfg.high_risk_confidence,
            min_mask_area_ratio=cfg.min_mask_area_ratio,
            high_growth_ratio=cfg.high_growth_ratio,
        )

    def _detect(self, frame: np.ndarray, frame_id: int, timestamp: float) -> list[dict]:
        if self.detector_type == "yolo":
            assert self.yolo is not None
            return self.yolo.detect(frame=frame, frame_id=frame_id, timestamp=timestamp)
        assert self.grounding is not None
        detections = self.grounding.predict(frame, self.cfg.prompts, frame_id=frame_id)
        normalized: list[dict] = []
        for d in detections:
            bbox = list(d.get("box", d.get("bbox", [0, 0, 1, 1])))
            normalized.append(
                {
                    **d,
                    "bbox": bbox,
                    "box": bbox,
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "source_model": "GroundingDINO",
                }
            )
        return normalized

    @staticmethod
    def _ms(start: float, end: float) -> float:
        return (end - start) * 1000.0

    @staticmethod
    def _stage_stats(values: list[float]) -> dict[str, float]:
        arr = np.array(values, dtype=np.float64)
        return {
            "avg_ms": float(arr.mean()) if arr.size else 0.0,
            "min_ms": float(arr.min()) if arr.size else 0.0,
            "max_ms": float(arr.max()) if arr.size else 0.0,
            "p50_ms": float(np.percentile(arr, 50)) if arr.size else 0.0,
            "p90_ms": float(np.percentile(arr, 90)) if arr.size else 0.0,
            "p95_ms": float(np.percentile(arr, 95)) if arr.size else 0.0,
            "p99_ms": float(np.percentile(arr, 99)) if arr.size else 0.0,
        }

    def _write_profile_outputs(
        self,
        output_dir: str,
        per_frame_timings: list[dict[str, Any]],
        processed_frames: int,
        total_runtime_sec: float,
        mock_mode: bool,
        detector_type: str,
        detector_model: str,
        detector_backend: str,
        event_count: int,
        sam2_used_count: int,
        vlm_used_count: int,
    ) -> dict[str, Any]:
        performance_path = os.path.join(output_dir, "performance.json")
        with open(performance_path, "w", encoding="utf-8") as f:
            json.dump(per_frame_timings, f, ensure_ascii=False, indent=2)

        stage_names = [
            "frame_load",
            "preprocessing",
            "yolo_inference",
            "grounding_dino_inference",
            "event_router",
            "sam2_inference",
            "postprocessing",
            "temporal_tracking",
            "vlm_explanation",
            "risk_engine",
            "visualization",
            "json_write",
            "total_per_frame",
        ]
        summary_by_stage: dict[str, dict[str, float]] = {}
        for stage in stage_names:
            values = [float(row.get(f"{stage}_ms", 0.0)) for row in per_frame_timings]
            summary_by_stage[stage] = self._stage_stats(values)

        # Bottleneck = stage with the highest average latency, excluding total_per_frame.
        candidate_stages = [
            s
            for s in stage_names
            if s not in {"total_per_frame"}
        ]
        bottleneck_stage = max(
            candidate_stages,
            key=lambda s: summary_by_stage[s]["avg_ms"],
            default="n/a",
        )
        bottleneck_avg_ms = (
            summary_by_stage[bottleneck_stage]["avg_ms"] if bottleneck_stage in summary_by_stage else 0.0
        )

        avg_fps = (processed_frames / total_runtime_sec) if total_runtime_sec > 0 else 0.0
        avg_latency_ms = summary_by_stage["total_per_frame"]["avg_ms"]
        total_runtime_ms = total_runtime_sec * 1000.0

        queue_wait_avg_ms: float | None = None
        if any(row.get("queue_wait_ms") is not None for row in per_frame_timings):
            queue_vals = [float(row.get("queue_wait_ms", 0.0)) for row in per_frame_timings]
            queue_wait_avg_ms = float(np.mean(np.array(queue_vals, dtype=np.float64))) if queue_vals else 0.0

        stage_map_for_metrics = {
            "frame_load_avg_ms": "frame_load",
            "preprocess_avg_ms": "preprocessing",
            "yolo_inference_avg_ms": "yolo_inference",
            "risk_engine_avg_ms": "risk_engine",
            "event_router_avg_ms": "event_router",
            "sam2_inference_avg_ms": "sam2_inference",
            "vlm_inference_avg_ms": "vlm_explanation",
            "output_write_avg_ms": "json_write",
            "end_to_end_avg_ms": "total_per_frame",
        }

        standardized_metrics: dict[str, Any] = {
            "environment_name": self.cfg.environment_name,
            "status": "SUCCESS",
            "run_location": self._infer_run_location(self.cfg.environment_name),
            "actual_runtime": self._infer_actual_runtime(self.cfg.run_type),
            "container_validation": bool("docker" in str(self.cfg.run_type).lower()),
            "container_runtime": "docker" if "docker" in str(self.cfg.run_type).lower() else None,
            "run_type": self.cfg.run_type,
            "detector_model": detector_model,
            "detector_backend": detector_backend,
            "model_zoo_candidates": {
                "detection": ["YOLOv8n", "YOLOv8m"],
                "segmentation": ["SAM2"],
                "vlm_explanation": ["Qwen2.5-VL-7B"],
                "optional_llm_summary": ["A.X-4.0-Light"],
            },
            "vlm_enabled": bool(self.cfg.vlm_enabled),
            "vlm_event_only": bool(
                self.cfg.vlm_only_on_risk_levels
                and {level.upper() for level in self.cfg.vlm_only_on_risk_levels}
                == {"MEDIUM", "HIGH"}
            ),
            "vlm_model": self.cfg.vlm_model_name_or_path,
            "vlm_max_new_tokens": int(self.cfg.vlm_max_new_tokens),
            "force_event_mode": bool(self.cfg.force_event_mode),
            "processed_frames": processed_frames,
            "total_runtime_ms": round(total_runtime_ms, 4),
            "avg_latency_per_frame_ms": round(avg_latency_ms, 4),
            "average_fps": round(avg_fps, 4),
            "frame_load_avg_ms": None,
            "preprocess_avg_ms": None,
            "yolo_inference_avg_ms": None,
            "risk_engine_avg_ms": None,
            "event_router_avg_ms": None,
            "queue_wait_avg_ms": round(queue_wait_avg_ms, 4) if queue_wait_avg_ms is not None else None,
            "sam2_inference_avg_ms": None,
            "vlm_explanation_avg_ms": None,
            "vlm_inference_avg_ms": None,
            "output_write_avg_ms": None,
            "end_to_end_avg_ms": None,
            "end_to_end_p95_ms": None,
            "event_count": event_count,
            "sam2_used_count": sam2_used_count,
            "vlm_used_count": vlm_used_count,
            "bottleneck_stage": bottleneck_stage,
        }
        for metric_key, stage_key in stage_map_for_metrics.items():
            standardized_metrics[metric_key] = round(summary_by_stage[stage_key]["avg_ms"], 4)
        standardized_metrics["vlm_explanation_avg_ms"] = standardized_metrics["vlm_inference_avg_ms"]
        standardized_metrics["end_to_end_p95_ms"] = round(summary_by_stage["total_per_frame"]["p95_ms"], 4)

        note_text = (
            "Mock mode timings are useful for validating pipeline overhead and stage orchestration. "
            "Use real GroundingDINO/SAM2/VLM backends for true model latency benchmarking."
            if mock_mode
            else "Timings include active backend inference paths."
        )
        if self.cfg.force_event_mode:
            note_text += " Forced MEDIUM/HIGH events enabled for bottleneck-load profiling (not accuracy evaluation)."

        summary_payload: dict[str, Any] = {
            **standardized_metrics,
            "processed_frames": processed_frames,
            "total_runtime_sec": round(total_runtime_sec, 6),
            "average_fps": round(avg_fps, 4),
            "average_latency_per_frame_ms": round(avg_latency_ms, 4),
            "bottleneck_stage": bottleneck_stage,
            "bottleneck_avg_ms": round(bottleneck_avg_ms, 4),
            "detector_type": detector_type,
            "detector_model": detector_model,
            "detector_backend": detector_backend,
            "event_count": event_count,
            "sam2_used_count": sam2_used_count,
            "vlm_used_count": vlm_used_count,
            "standardized_metrics": standardized_metrics,
            "stage_statistics": summary_by_stage,
            "note": note_text,
        }
        summary_payload["notes"] = summary_payload["note"]
        summary_path = os.path.join(output_dir, "performance_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, ensure_ascii=False, indent=2)

        self.logger.info("Profiling enabled. Wrote: %s", performance_path)
        self.logger.info("Profiling summary: %s", summary_path)
        self.logger.info("Processed frames: %d", processed_frames)
        self.logger.info("Total runtime: %.3f sec", total_runtime_sec)
        self.logger.info("Average FPS: %.3f", avg_fps)
        self.logger.info("Average latency/frame: %.3f ms", avg_latency_ms)
        self.logger.info("Bottleneck stage: %s (avg %.3f ms)", bottleneck_stage, bottleneck_avg_ms)

        return summary_payload

    def run(self, input_video: str, output_dir: str, profile: bool = False) -> list[dict[str, Any]]:
        frames_dir = os.path.join(output_dir, "frames")
        overlays_dir = os.path.join(output_dir, "overlays")
        os.makedirs(frames_dir, exist_ok=True)
        os.makedirs(overlays_dir, exist_ok=True)

        alerts: list[dict[str, Any]] = []
        per_frame_timings: list[dict[str, Any]] = []
        previous_event_area_ratio: float | None = None
        consecutive_detected = 0
        event_count = 0
        sam2_used_count = 0
        vlm_used_count = 0

        total_start = time.perf_counter()
        frame_iter = sample_video_frames(input_video, frame_stride=self.cfg.frame_stride)
        with tqdm(desc="Processing frames", unit="frame") as pbar:
            while True:
                total_per_frame_start = time.perf_counter()

                # Stage: frame_load (pull next sampled frame from loader)
                t0 = time.perf_counter()
                try:
                    packet = next(frame_iter)
                except StopIteration:
                    break
                t1 = time.perf_counter()
                frame_load_ms = self._ms(t0, t1)

                # Stage: preprocessing
                t0 = time.perf_counter()
                frame = preprocess_frame(packet.image_bgr)
                t1 = time.perf_counter()
                preprocessing_ms = self._ms(t0, t1)

                # Stage: detector inference (YOLO or GroundingDINO)
                t0 = time.perf_counter()
                detections = self._detect(frame, frame_id=packet.frame_id, timestamp=packet.timestamp)
                t1 = time.perf_counter()
                detector_inference_ms = self._ms(t0, t1)
                yolo_inference_ms = detector_inference_ms if self.detector_type == "yolo" else 0.0
                grounding_dino_inference_ms = (
                    detector_inference_ms if self.detector_type == "groundingdino" else 0.0
                )

                # Stage: postprocessing (detection filtering)
                t0 = time.perf_counter()
                detections = filter_detections(detections, self.cfg.detection_confidence)
                t1 = time.perf_counter()
                detection_filter_ms = self._ms(t0, t1)

                if detections:
                    consecutive_detected += 1
                else:
                    consecutive_detected = 0

                avg_conf = (
                    float(sum(float(d["score"]) for d in detections) / len(detections))
                    if detections
                    else 0.0
                )
                labels = sorted(set(str(d.get("label", "unknown")) for d in detections))

                # Stage: risk_engine (pre-event, detector-only)
                t0 = time.perf_counter()
                risk_level, alert_message = self.risk_engine.evaluate(
                    avg_confidence=avg_conf,
                    mask_area_ratio=0.0,
                    mask_growth=0.0,
                    consecutive_frames=consecutive_detected,
                    region_count=0,
                )
                t1 = time.perf_counter()
                risk_engine_ms = self._ms(t0, t1)

                # Stage: event_router
                t0 = time.perf_counter()
                event_triggered = risk_level in {"MEDIUM", "HIGH"} and len(detections) > 0
                if self.cfg.force_event_mode:
                    event_triggered = True
                    if risk_level == "LOW":
                        risk_level = "MEDIUM"
                        alert_message = "Forced MEDIUM event for bottleneck profiling mode."
                t1 = time.perf_counter()
                event_router_ms = self._ms(t0, t1)

                if event_triggered:
                    event_count += 1

                # Stage: sam2_inference (event-triggered only)
                t0 = time.perf_counter()
                if event_triggered:
                    masks = self.sam2.segment(frame, detections, frame_id=packet.frame_id)
                    sam2_used = True
                    sam2_used_count += 1
                else:
                    masks = []
                    sam2_used = False
                t1 = time.perf_counter()
                sam2_inference_ms = self._ms(t0, t1)

                # Stage: postprocessing (mask merge + area/regions)
                t0 = time.perf_counter()
                if masks:
                    merged_mask = merge_masks(masks, image_shape=frame.shape[:2])
                    area_ratio = mask_area_ratio(merged_mask)
                    region_count = count_regions(merged_mask)
                else:
                    merged_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    area_ratio = 0.0
                    region_count = 0
                t1 = time.perf_counter()
                mask_stats_ms = self._ms(t0, t1)
                postprocessing_ms = detection_filter_ms + mask_stats_ms

                # Stage: temporal_tracking
                t0 = time.perf_counter()
                if sam2_used:
                    growth = compute_growth(area_ratio, previous_event_area_ratio)
                    previous_event_area_ratio = area_ratio
                else:
                    growth = 0.0
                t1 = time.perf_counter()
                temporal_tracking_ms = self._ms(t0, t1)

                # Re-score risk when event path provides SAM2 feedback.
                if sam2_used:
                    t0 = time.perf_counter()
                    risk_level, alert_message = self.risk_engine.evaluate(
                        avg_confidence=avg_conf,
                        mask_area_ratio=area_ratio,
                        mask_growth=growth,
                        consecutive_frames=consecutive_detected,
                        region_count=region_count,
                    )
                    t1 = time.perf_counter()
                    risk_engine_ms += self._ms(t0, t1)

                # Stage: vlm_explanation
                t0 = time.perf_counter()
                if self.vlm.should_explain(risk_level=risk_level, frame_id=packet.frame_id):
                    vlm_used = True
                    vlm_used_count += 1
                    explanation = self.vlm.explain(
                        frame=frame,
                        detections=detections,
                        merged_mask=merged_mask,
                        mask_area_ratio=area_ratio,
                        mask_growth=growth,
                        consecutive_frames=consecutive_detected,
                        risk_level=risk_level,
                    )
                else:
                    vlm_used = False
                    explanation = self.vlm.fallback_explanation(
                        detections=detections,
                        mask_area_ratio=area_ratio,
                        mask_growth=growth,
                        consecutive_frames=consecutive_detected,
                        risk_level=risk_level,
                    )
                t1 = time.perf_counter()
                vlm_explanation_ms = self._ms(t0, t1)

                # Stage: visualization
                t0 = time.perf_counter()
                frame_name = f"frame_{packet.frame_id:06d}.jpg"
                save_image(frame, os.path.join(frames_dir, frame_name))
                overlay = draw_overlay(
                    frame,
                    detections,
                    merged_mask,
                    risk_level=risk_level,
                    alpha=self.cfg.mask_alpha,
                    box_thickness=self.cfg.box_thickness,
                )
                save_image(overlay, os.path.join(overlays_dir, frame_name))
                t1 = time.perf_counter()
                visualization_ms = self._ms(t0, t1)

                # Stage: json_write (building/append frame records in-memory)
                t0 = time.perf_counter()
                alerts.append(
                    {
                        "frame_id": packet.frame_id,
                        "timestamp": round(packet.timestamp, 3),
                        "detector_type": self.detector_type,
                        "detector_model": (
                            self.yolo.source_model if self.detector_type == "yolo" and self.yolo else "GroundingDINO"
                        ),
                        "detector_backend": self.cfg.yolo_backend if self.detector_type == "yolo" else "native",
                        "detected_labels": labels,
                        "confidence": round(avg_conf, 4),
                        "mask_area_ratio": round(area_ratio, 6),
                        "mask_growth": round(growth, 6),
                        "risk_level": risk_level,
                        "event_triggered": event_triggered,
                        "sam2_used": sam2_used,
                        "vlm_used": vlm_used,
                        "vlm_model": self.cfg.vlm_model_name_or_path or self.cfg.vlm_backend,
                        "explanation": explanation,
                        "fallback_used": bool(self.vlm.last_fallback_used) or not vlm_used,
                        "alert_message": alert_message,
                    }
                )
                t1 = time.perf_counter()
                json_write_ms = self._ms(t0, t1)

                total_per_frame_end = time.perf_counter()
                total_per_frame_ms = self._ms(total_per_frame_start, total_per_frame_end)

                if profile:
                    per_frame_timings.append(
                        {
                            "frame_id": packet.frame_id,
                            "timestamp": round(packet.timestamp, 3),
                            "frame_load_ms": round(frame_load_ms, 6),
                            "preprocessing_ms": round(preprocessing_ms, 6),
                            "yolo_inference_ms": round(yolo_inference_ms, 6),
                            "grounding_dino_inference_ms": round(grounding_dino_inference_ms, 6),
                            "event_router_ms": round(event_router_ms, 6),
                            "sam2_inference_ms": round(sam2_inference_ms, 6),
                            "postprocessing_ms": round(postprocessing_ms, 6),
                            "temporal_tracking_ms": round(temporal_tracking_ms, 6),
                            "vlm_explanation_ms": round(vlm_explanation_ms, 6),
                            "risk_engine_ms": round(risk_engine_ms, 6),
                            "visualization_ms": round(visualization_ms, 6),
                            "json_write_ms": round(json_write_ms, 6),
                            "queue_wait_ms": None,
                            "total_per_frame_ms": round(total_per_frame_ms, 6),
                        }
                    )

                pbar.update(1)

        alerts_path = os.path.join(output_dir, "alerts.json")
        with open(alerts_path, "w", encoding="utf-8") as f:
            json.dump(alerts, f, ensure_ascii=False, indent=2)
        total_runtime_sec = time.perf_counter() - total_start

        self.logger.info("Done. Alerts saved to: %s", alerts_path)
        if profile:
            if self.detector_type == "yolo" and self.yolo is not None:
                detector_model = self.yolo.source_model
                detector_backend = self.cfg.yolo_backend
            else:
                detector_model = "GroundingDINO"
                detector_backend = "native"
            self._write_profile_outputs(
                output_dir=output_dir,
                per_frame_timings=per_frame_timings,
                processed_frames=len(alerts),
                total_runtime_sec=total_runtime_sec,
                mock_mode=bool(
                    (self.grounding.mock if self.grounding is not None else self.cfg.yolo_backend == "mock")
                    and self.sam2.mock
                ),
                detector_type=self.detector_type,
                detector_model=detector_model,
                detector_backend=detector_backend,
                event_count=event_count,
                sam2_used_count=sam2_used_count,
                vlm_used_count=vlm_used_count,
            )
        return alerts

