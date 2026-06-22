from __future__ import annotations

import argparse
import os

import yaml

from src.pipeline.wildfire_pipeline import PipelineConfig, WildfirePipeline
from src.utils.logging_utils import get_logger


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_pipeline_config(cfg: dict, frame_stride_override: int | None = None) -> PipelineConfig:
    frame_stride = (
        frame_stride_override
        if frame_stride_override is not None
        else int(cfg["sampling"]["frame_stride"])
    )
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
    return parser.parse_args()


def main() -> None:
    logger = get_logger("main")
    args = parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input video not found: {args.input}")
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config not found: {args.config}")

    cfg_raw = load_config(args.config)
    cfg = build_pipeline_config(cfg_raw, frame_stride_override=args.frame_stride)

    pipeline = WildfirePipeline(cfg=cfg, mock=args.mock)
    alerts = pipeline.run(input_video=args.input, output_dir=args.output, profile=args.profile)
    logger.info("Processed %d sampled frames.", len(alerts))


if __name__ == "__main__":
    main()

