from __future__ import annotations

from typing import List

import numpy as np

from src.utils.logging_utils import get_logger


class VLMExplainer:
    """
    VLM integration interface.

    TODO:
      - Replace placeholder logic with an actual VLM backend (local or API).
      - Keep the same `explain(...)` signature for plug-and-play replacement.
    """

    def __init__(
        self,
        mode: str = "placeholder",
        enabled: bool = True,
        backend: str = "placeholder",
        model_name_or_path: str | None = None,
        device: str | None = None,
        torch_dtype: str = "float16",
        load_in_4bit: bool = False,
        max_new_tokens: int = 64,
        only_on_risk_levels: list[str] | None = None,
        selected_frame_interval: int = 1,
    ):
        self.logger = get_logger(self.__class__.__name__)
        self.mode = mode
        self.enabled = enabled
        self.backend = backend
        self.model_name_or_path = model_name_or_path
        self.device = device
        self.torch_dtype = torch_dtype
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self.only_on_risk_levels = (
            {level.strip().upper() for level in only_on_risk_levels}
            if only_on_risk_levels
            else None
        )
        self.selected_frame_interval = max(1, int(selected_frame_interval))

        self._hf_model = None
        self._hf_tokenizer = None
        self.last_fallback_used = False

    def should_explain(self, risk_level: str | None, frame_id: int) -> bool:
        if not self.enabled:
            return False
        if self.only_on_risk_levels and (risk_level or "").upper() not in self.only_on_risk_levels:
            return False
        if frame_id % self.selected_frame_interval != 0:
            return False
        return True

    def fallback_explanation(
        self,
        detections: List[dict],
        mask_area_ratio: float,
        mask_growth: float,
        consecutive_frames: int,
        risk_level: str | None = None,
    ) -> str:
        self.last_fallback_used = True
        return self._placeholder_explain(
            detections=detections,
            mask_area_ratio=mask_area_ratio,
            mask_growth=mask_growth,
            consecutive_frames=consecutive_frames,
            risk_level=risk_level,
        )

    def _ensure_hf_model(self) -> None:
        if self._hf_model is not None and self._hf_tokenizer is not None:
            return
        if not self.model_name_or_path:
            raise ValueError("VLM model_name_or_path is required for huggingface backend.")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "Transformers backend import failed for VLM. "
                "Install transformers/torch dependencies."
            ) from exc

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(str(self.torch_dtype).lower(), torch.float16)

        model_kwargs = {"torch_dtype": torch_dtype}
        if self.load_in_4bit:
            model_kwargs["load_in_4bit"] = True
        if self.device:
            model_kwargs["device_map"] = "auto" if self.device == "cuda" else self.device

        self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            **model_kwargs,
        )
        self.logger.info("Loaded VLM backend=%s model=%s", self.backend, self.model_name_or_path)

    def _placeholder_explain(
        self,
        detections: List[dict],
        mask_area_ratio: float,
        mask_growth: float,
        consecutive_frames: int,
        risk_level: str | None = None,
    ) -> str:
        if not detections:
            return "No clear smoke or flame signal is detected in this frame."

        labels = sorted(set(d["label"] for d in detections))
        avg_conf = sum(float(d["score"]) for d in detections) / max(1, len(detections))
        growth_pct = int(mask_growth * 100)
        area_pct = mask_area_ratio * 100.0

        prefix = f"{', '.join(labels)}-like region detected."
        evidence = (
            f"Average confidence is {avg_conf:.2f}, mask area is {area_pct:.2f}% of frame, "
            f"and temporal growth is {growth_pct:+d}% across recent frames."
        )

        if consecutive_frames >= 3 and mask_growth > 0:
            temporal = (
                f"The signal has persisted for {consecutive_frames} consecutive frames "
                "and appears to be expanding."
            )
        elif consecutive_frames >= 2:
            temporal = f"The signal persists for {consecutive_frames} consecutive frames."
        else:
            temporal = "The signal is currently weak or short-lived."

        suffix = f" Current risk estimate: {risk_level}." if risk_level else ""
        return f"{prefix} {evidence} {temporal}{suffix}"

    def explain(
        self,
        detections: List[dict],
        mask_area_ratio: float,
        mask_growth: float,
        consecutive_frames: int,
        risk_level: str | None = None,
        frame: np.ndarray | None = None,
        merged_mask: np.ndarray | None = None,
    ) -> str:
        if self.backend == "placeholder":
            self.last_fallback_used = False
            return self._placeholder_explain(
                detections=detections,
                mask_area_ratio=mask_area_ratio,
                mask_growth=mask_growth,
                consecutive_frames=consecutive_frames,
                risk_level=risk_level,
            )

        if self.backend == "huggingface":
            try:
                self._ensure_hf_model()
                assert self._hf_model is not None
                assert self._hf_tokenizer is not None

                labels = sorted(set(d.get("label", "unknown") for d in detections))
                avg_conf = (
                    sum(float(d.get("score", 0.0)) for d in detections) / max(1, len(detections))
                    if detections
                    else 0.0
                )
                bbox_preview = [d.get("bbox", d.get("box")) for d in detections[:3]]
                mask_pixels = int(np.count_nonzero(merged_mask)) if merged_mask is not None else 0
                frame_shape = tuple(frame.shape[:2]) if frame is not None else None
                prompt = (
                    "You are a wildfire monitoring assistant. "
                    "Summarize this frame risk evidence in 1-2 short sentences.\n"
                    f"risk_level={risk_level}\n"
                    f"labels={labels}\n"
                    f"avg_confidence={avg_conf:.3f}\n"
                    f"bbox_preview={bbox_preview}\n"
                    f"frame_shape={frame_shape}\n"
                    f"mask_area_ratio={mask_area_ratio:.6f}\n"
                    f"mask_pixels={mask_pixels}\n"
                    f"mask_growth={mask_growth:.6f}\n"
                    f"consecutive_frames={consecutive_frames}\n"
                )
                inputs = self._hf_tokenizer(prompt, return_tensors="pt")
                if hasattr(self._hf_model, "device"):
                    inputs = {k: v.to(self._hf_model.device) for k, v in inputs.items()}
                outputs = self._hf_model.generate(
                    **inputs,
                    max_new_tokens=int(self.max_new_tokens),
                    do_sample=False,
                )
                text = self._hf_tokenizer.decode(outputs[0], skip_special_tokens=True)
                if text.startswith(prompt):
                    text = text[len(prompt) :].strip()
                self.last_fallback_used = False
                return text.strip() or self._placeholder_explain(
                    detections=detections,
                    mask_area_ratio=mask_area_ratio,
                    mask_growth=mask_growth,
                    consecutive_frames=consecutive_frames,
                    risk_level=risk_level,
                )
            except Exception as exc:
                self.logger.warning(
                    "VLM huggingface inference failed (%s). Falling back to placeholder explain.",
                    exc,
                )
                self.last_fallback_used = True
                return self._placeholder_explain(
                    detections=detections,
                    mask_area_ratio=mask_area_ratio,
                    mask_growth=mask_growth,
                    consecutive_frames=consecutive_frames,
                    risk_level=risk_level,
                )

        self.logger.warning("Unknown VLM backend '%s'. Using placeholder.", self.backend)
        self.last_fallback_used = True
        return self._placeholder_explain(
            detections=detections,
            mask_area_ratio=mask_area_ratio,
            mask_growth=mask_growth,
            consecutive_frames=consecutive_frames,
            risk_level=risk_level,
        )

