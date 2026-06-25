from __future__ import annotations

import argparse
import os

import yaml

from src.pipeline.wildfire_pipeline import PipelineConfig, WildfirePipeline
from src.utils.logging_utils import get_logger


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_pipeline_config(
    cfg: dict,
    frame_stride_override: int | None = None,
    detector_override: str | None = None,
    yolo_model_override: str | None = None,
    yolo_backend_override: str | None = None,
    environment_name_override: str | None = None,
    run_type_override: str | None = None,
    vlm_enabled_override: bool | None = None,
    vlm_event_only_override: bool | None = None,
    vlm_model_override: str | None = None,
    vlm_max_new_tokens_override: int | None = None,
    force_medium_high_events_override: bool | None = None,
) -> PipelineConfig:
    frame_stride = (
        frame_stride_override
        if frame_stride_override is not None
        else int(cfg["sampling"]["frame_stride"])
    )
    models_cfg = cfg.get("models", {})
    detector_cfg = models_cfg.get("detector", {})
    yolo_cfg = models_cfg.get("yolo", {})
    selected_detector = str(detector_override or detector_cfg.get("type", "yolo")).lower()
    selected_yolo_model = str(yolo_model_override or yolo_cfg.get("model_name", "yolov8n")).lower()
    selected_yolo_backend = str(yolo_backend_override or yolo_cfg.get("backend", "mock")).lower()
    selected_environment_name = str(environment_name_override or cfg.get("environment_name", "local_mock"))
    selected_run_type = str(run_type_override or cfg.get("run_type", "single_process"))
    vlm_cfg = cfg.get("models", {}).get("vlm", {})
    vlm_enabled = (
        bool(vlm_enabled_override)
        if vlm_enabled_override is not None
        else bool(vlm_cfg.get("enabled", True))
    )
    vlm_only_on_risk_levels = vlm_cfg.get("only_on_risk_levels")
    if vlm_event_only_override is True:
        vlm_only_on_risk_levels = ["MEDIUM", "HIGH"]
    selected_vlm_frame_interval = int(vlm_cfg.get("selected_frame_interval", 1))
    if vlm_event_only_override is True:
        selected_vlm_frame_interval = 1
    selected_vlm_model = vlm_model_override or vlm_cfg.get("model_name_or_path")
    selected_vlm_max_new_tokens = (
        int(vlm_max_new_tokens_override)
        if vlm_max_new_tokens_override is not None
        else int(vlm_cfg.get("max_new_tokens", 64))
    )
    force_event_mode = bool(force_medium_high_events_override)

    return PipelineConfig(
        prompts=list(cfg["prompts"]),
        frame_stride=frame_stride,
        detection_confidence=float(cfg["thresholds"]["detection_confidence"]),
        high_risk_confidence=float(cfg["thresholds"]["high_risk_confidence"]),
        min_mask_area_ratio=float(cfg["thresholds"]["min_mask_area_ratio"]),
        high_growth_ratio=float(cfg["thresholds"]["high_growth_ratio"]),
        box_thickness=int(cfg["visualization"]["box_thickness"]),
        mask_alpha=float(cfg["visualization"]["mask_alpha"]),
        grounding_checkpoint=cfg.get("models", {}).get("grounding_dino", {}).get("checkpoint"),
        grounding_config=cfg.get("models", {}).get("grounding_dino", {}).get("config"),
        grounding_box_threshold=float(
            cfg.get("models", {}).get("grounding_dino", {}).get("box_threshold", 0.35)
        ),
        grounding_text_threshold=float(
            cfg.get("models", {}).get("grounding_dino", {}).get("text_threshold", 0.25)
        ),
        sam2_checkpoint=cfg.get("models", {}).get("sam2", {}).get("checkpoint"),
        sam2_model_config=cfg.get("models", {}).get("sam2", {}).get("model_config"),
        model_device=cfg.get("models", {}).get("device"),
        vlm_enabled=vlm_enabled,
        vlm_backend=str(vlm_cfg.get("backend", "placeholder")),
        vlm_model_name_or_path=selected_vlm_model,
        vlm_device=vlm_cfg.get("device"),
        vlm_torch_dtype=str(vlm_cfg.get("torch_dtype", "float16")),
        vlm_load_in_4bit=bool(vlm_cfg.get("load_in_4bit", False)),
        vlm_max_new_tokens=selected_vlm_max_new_tokens,
        vlm_only_on_risk_levels=vlm_only_on_risk_levels,
        vlm_selected_frame_interval=int(
            selected_vlm_frame_interval
        ),
        detector_type=selected_detector,
        yolo_model=selected_yolo_model,
        yolo_backend=selected_yolo_backend,
        yolo_conf_threshold=float(yolo_cfg.get("conf_threshold", cfg["thresholds"]["detection_confidence"])),
        environment_name=selected_environment_name,
        run_type=selected_run_type,
        force_event_mode=force_event_mode,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wildfire early-warning MVP pipeline")
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--config", default="config.yaml", help="YAML config file path")
    parser.add_argument("--mock", action="store_true", help="Enable mock mode")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable per-stage latency profiling and bottleneck analysis",
    )
    parser.add_argument("--frame-stride", type=int, default=None, help="Override frame stride")
    parser.add_argument(
        "--detector",
        default="yolo",
        choices=["yolo", "groundingdino"],
        help="Detector type for first-stage detection",
    )
    parser.add_argument(
        "--yolo-model",
        default="yolov8n",
        choices=["yolov8n", "yolov8m"],
        help="YOLO model variant",
    )
    parser.add_argument(
        "--yolo-backend",
        default="mock",
        choices=["mock", "ultralytics"],
        help="YOLO backend implementation",
    )
    parser.add_argument(
        "--environment-name",
        default="local_mock",
        help="Name of runtime environment for profiling reports",
    )
    parser.add_argument(
        "--run-type",
        default="single_process",
        help="Run type label for profiling reports",
    )
    parser.add_argument(
        "--enable-vlm",
        action="store_true",
        help="Force-enable VLM path",
    )
    parser.add_argument(
        "--disable-vlm",
        action="store_true",
        help="Force-disable VLM path",
    )
    parser.add_argument(
        "--vlm-event-only",
        action="store_true",
        help="Restrict VLM invocation to MEDIUM/HIGH events",
    )
    parser.add_argument(
        "--vlm-model",
        default=None,
        help="Override VLM model candidate name",
    )
    parser.add_argument(
        "--vlm-max-new-tokens",
        type=int,
        default=None,
        help="Override VLM max_new_tokens",
    )
    parser.add_argument(
        "--force-medium-high-events",
        action="store_true",
        help="Force MEDIUM/HIGH event routing for bottleneck-load profiling only",
    )
    return parser.parse_args()


def main() -> None:
    logger = get_logger("main")
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input video not found: {args.input}")
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")

    cfg_raw = load_config(args.config)
    if args.enable_vlm and args.disable_vlm:
        raise ValueError("Use only one of --enable-vlm or --disable-vlm.")
    vlm_enabled_override = None
    if args.enable_vlm:
        vlm_enabled_override = True
    elif args.disable_vlm:
        vlm_enabled_override = False
    cfg = build_pipeline_config(
        cfg_raw,
        frame_stride_override=args.frame_stride,
        detector_override=args.detector,
        yolo_model_override=args.yolo_model,
        yolo_backend_override=args.yolo_backend,
        environment_name_override=args.environment_name,
        run_type_override=args.run_type,
        vlm_enabled_override=vlm_enabled_override,
        vlm_event_only_override=args.vlm_event_only,
        vlm_model_override=args.vlm_model,
        vlm_max_new_tokens_override=args.vlm_max_new_tokens,
        force_medium_high_events_override=args.force_medium_high_events,
    )

    pipeline = WildfirePipeline(cfg=cfg, mock=args.mock)
    alerts = pipeline.run(input_video=args.input, output_dir=args.output, profile=args.profile)
    logger.info("Processed %d sampled frames.", len(alerts))


if __name__ == "__main__":
    main()

