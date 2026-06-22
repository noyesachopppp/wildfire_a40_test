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


class WildfirePipeline:
    def __init__(self, cfg: PipelineConfig, mock: bool = False):
        self.cfg = cfg
        self.logger = get_logger(self.__class__.__name__)
        self.grounding = GroundingDINOWrapper(
            mock=mock,
            checkpoint=cfg.grounding_checkpoint,
            config_path=cfg.grounding_config,
            box_threshold=cfg.grounding_box_threshold,
            text_threshold=cfg.grounding_text_threshold,
            device=cfg.model_device,
        )
        self.sam2 = SAM2Wrapper(
            mock=mock,
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
    ) -> dict[str, Any]:
        performance_path = os.path.join(output_dir, "performance.json")
        with open(performance_path, "w", encoding="utf-8") as f:
            json.dump(per_frame_timings, f, ensure_ascii=False, indent=2)

        stage_names = [
            "frame_load",
            "preprocessing",
            "grounding_dino_inference",
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
        summary_payload: dict[str, Any] = {
            "processed_frames": processed_frames,
            "total_runtime_sec": round(total_runtime_sec, 6),
            "average_fps": round(avg_fps, 4),
            "average_latency_per_frame_ms": round(avg_latency_ms, 4),
            "bottleneck_stage": bottleneck_stage,
            "bottleneck_avg_ms": round(bottleneck_avg_ms, 4),
            "stage_statistics": summary_by_stage,
            "notes": (
                "Mock mode timings are useful for validating pipeline overhead and stage orchestration. "
                "Use real GroundingDINO/SAM2/VLM backends for true model latency benchmarking."
                if mock_mode
                else "Timings include active backend inference paths."
            ),
        }
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
        previous_area_ratio: float | None = None
        consecutive_detected = 0

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

                # Stage: grounding_dino_inference
                t0 = time.perf_counter()
                detections = self.grounding.predict(frame, self.cfg.prompts, frame_id=packet.frame_id)
                t1 = time.perf_counter()
                grounding_dino_inference_ms = self._ms(t0, t1)

                # Stage: postprocessing (detection filtering + mask merge + area/regions)
                t0 = time.perf_counter()
                detections = filter_detections(detections, self.cfg.detection_confidence)
                t1 = time.perf_counter()
                detection_filter_ms = self._ms(t0, t1)

                # Stage: sam2_inference
                t0 = time.perf_counter()
                masks = self.sam2.segment(frame, detections, frame_id=packet.frame_id)
                t1 = time.perf_counter()
                sam2_inference_ms = self._ms(t0, t1)

                t0 = time.perf_counter()
                merged_mask = merge_masks(masks, image_shape=frame.shape[:2])
                area_ratio = mask_area_ratio(merged_mask)
                region_count = count_regions(merged_mask)
                t1 = time.perf_counter()
                mask_stats_ms = self._ms(t0, t1)
                postprocessing_ms = detection_filter_ms + mask_stats_ms

                # Stage: temporal_tracking
                t0 = time.perf_counter()
                growth = compute_growth(area_ratio, previous_area_ratio)
                previous_area_ratio = area_ratio
                if detections:
                    consecutive_detected += 1
                else:
                    consecutive_detected = 0
                t1 = time.perf_counter()
                temporal_tracking_ms = self._ms(t0, t1)

                avg_conf = (
                    float(sum(float(d["score"]) for d in detections) / len(detections))
                    if detections
                    else 0.0
                )
                labels = sorted(set(d["label"] for d in detections))

                # Stage: risk_engine
                t0 = time.perf_counter()
                risk_level, alert_message = self.risk_engine.evaluate(
                    avg_confidence=avg_conf,
                    mask_area_ratio=area_ratio,
                    mask_growth=growth,
                    consecutive_frames=consecutive_detected,
                    region_count=region_count,
                )
                t1 = time.perf_counter()
                risk_engine_ms = self._ms(t0, t1)

                # Stage: vlm_explanation
                t0 = time.perf_counter()
                if self.vlm.should_explain(risk_level=risk_level, frame_id=packet.frame_id):
                    vlm_used = True
                    explanation = self.vlm.explain(
                        detections=detections,
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
                        "detected_labels": labels,
                        "confidence": round(avg_conf, 4),
                        "mask_area_ratio": round(area_ratio, 6),
                        "mask_growth": round(growth, 6),
                        "risk_level": risk_level,
                        "vlm_used": vlm_used,
                        "vlm_model": self.cfg.vlm_model_name_or_path or self.cfg.vlm_backend,
                        "explanation": explanation,
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
                            "grounding_dino_inference_ms": round(grounding_dino_inference_ms, 6),
                            "sam2_inference_ms": round(sam2_inference_ms, 6),
                            "postprocessing_ms": round(postprocessing_ms, 6),
                            "temporal_tracking_ms": round(temporal_tracking_ms, 6),
                            "vlm_explanation_ms": round(vlm_explanation_ms, 6),
                            "risk_engine_ms": round(risk_engine_ms, 6),
                            "visualization_ms": round(visualization_ms, 6),
                            "json_write_ms": round(json_write_ms, 6),
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
            self._write_profile_outputs(
                output_dir=output_dir,
                per_frame_timings=per_frame_timings,
                processed_frames=len(alerts),
                total_runtime_sec=total_runtime_sec,
                mock_mode=self.grounding.mock and self.sam2.mock,
            )
        return alerts

