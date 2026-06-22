from __future__ import annotations

import os
from typing import List

import numpy as np

from src.utils.logging_utils import get_logger


class GroundingDINOWrapper:
    """
    GroundingDINO inference adapter.

    Expected output format:
    [
      {"box": [x1, y1, x2, y2], "label": "smoke", "score": 0.82},
      ...
    ]
    """

    def __init__(
        self,
        mock: bool = False,
        checkpoint: str | None = None,
        config_path: str | None = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        device: str | None = None,
    ):
        self.logger = get_logger(self.__class__.__name__)
        self.mock = mock
        self.checkpoint = checkpoint
        self.config_path = config_path
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.device = device
        self.model = None
        self._predict_fn = None

        if not self.mock:
            self._initialize_real_model()

    @staticmethod
    def _is_unset_path(path: str | None) -> bool:
        if path is None:
            return True
        normalized = str(path).strip()
        return not normalized or normalized.upper().startswith("TODO:")

    def _validate_required_files(self) -> None:
        if self._is_unset_path(self.checkpoint):
            raise FileNotFoundError(
                "GroundingDINO checkpoint path is missing in config. "
                "Set models.grounding_dino.checkpoint or run with --mock."
            )
        if self._is_unset_path(self.config_path):
            raise FileNotFoundError(
                "GroundingDINO config path is missing in config. "
                "Set models.grounding_dino.config or run with --mock."
            )
        assert self.checkpoint is not None
        assert self.config_path is not None
        if not os.path.exists(self.checkpoint):
            raise FileNotFoundError(
                f"GroundingDINO checkpoint not found: {self.checkpoint}. "
                "Verify the path or run with --mock."
            )
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"GroundingDINO config not found: {self.config_path}. "
                "Verify the path or run with --mock."
            )

    def _initialize_real_model(self) -> None:
        self._validate_required_files()
        try:
            import torch
            from groundingdino.util.inference import load_model, predict
        except Exception as exc:
            raise RuntimeError(
                "Failed to import GroundingDINO dependencies. "
                "Install GroundingDINO/Torch for real inference or run with --mock."
            ) from exc

        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        assert self.config_path is not None
        assert self.checkpoint is not None
        self.model = load_model(self.config_path, self.checkpoint, device=self.device)
        self._predict_fn = predict
        self.logger.info("GroundingDINO real backend initialized on device=%s", self.device)

    def predict(self, image_bgr: np.ndarray, prompts: list[str], frame_id: int) -> List[dict]:
        """
        Detect text-guided candidate regions.

        In mock mode, synthetic detections are generated so downstream modules can be tested.
        """
        if self.mock:
            return self._mock_predict(image_bgr, prompts, frame_id)

        return self._real_predict(image_bgr, prompts)

    @staticmethod
    def _to_xyxy_abs(box: np.ndarray, width: int, height: int) -> list[int]:
        # GroundingDINO predict() commonly returns normalized cx, cy, w, h.
        cx, cy, bw, bh = [float(v) for v in box.tolist()]
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
        x1 = int(np.clip(round(x1), 0, max(0, width - 1)))
        y1 = int(np.clip(round(y1), 0, max(0, height - 1)))
        x2 = int(np.clip(round(x2), 1, max(1, width - 1)))
        y2 = int(np.clip(round(y2), 1, max(1, height - 1)))
        return [x1, y1, x2, y2]

    def _normalize_label(self, phrase: str, prompts: list[str]) -> str:
        phrase_l = phrase.strip().lower()
        if not phrase_l:
            return "unknown"
        for prompt in prompts:
            prompt_l = prompt.strip().lower()
            if not prompt_l:
                continue
            if prompt_l in phrase_l or phrase_l in prompt_l:
                return prompt_l
        return phrase_l

    def _real_predict(self, image_bgr: np.ndarray, prompts: list[str]) -> List[dict]:
        if self.model is None or self._predict_fn is None:
            raise RuntimeError(
                "GroundingDINO real backend is not initialized. "
                "Provide valid model files or run with --mock."
            )

        h, w = image_bgr.shape[:2]
        caption = " . ".join(p.strip().lower() for p in prompts if p.strip())
        if not caption:
            return []
        caption = f"{caption} ."

        try:
            from groundingdino.datasets import transforms as T
            from PIL import Image
        except Exception as exc:
            raise RuntimeError(
                "Failed to import GroundingDINO preprocessing dependencies. "
                "Install GroundingDINO extras (including Pillow) or run with --mock."
            ) from exc

        image_rgb = image_bgr[:, :, ::-1]
        image_pil = Image.fromarray(image_rgb)
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_tensor, _ = transform(image_pil, None)

        boxes, logits, phrases = self._predict_fn(
            model=self.model,
            image=image_tensor,
            caption=caption,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )

        detections: list[dict] = []
        if boxes is None:
            return detections

        for box, score, phrase in zip(boxes, logits, phrases):
            label = self._normalize_label(str(phrase), prompts)
            xyxy = self._to_xyxy_abs(np.asarray(box), w, h)
            detections.append(
                {
                    "box": xyxy,
                    "label": label,
                    "score": float(score),
                }
            )

        return detections

    def _mock_predict(self, image_bgr: np.ndarray, prompts: list[str], frame_id: int) -> List[dict]:
        h, w = image_bgr.shape[:2]
        t = frame_id // 10

        # Deterministic pseudo-events for reproducible demos.
        # Event 1: "smoke" candidate gradually moves and grows.
        x1 = int(w * 0.15 + min(t, 40) * 2)
        y1 = int(h * 0.20 + min(t, 40))
        x2 = int(min(w - 1, x1 + w * 0.18 + t * 1.2))
        y2 = int(min(h - 1, y1 + h * 0.15 + t * 0.8))
        score_smoke = min(0.9, 0.35 + 0.02 * t)

        detections = [
            {
                "box": [max(0, x1), max(0, y1), max(1, x2), max(1, y2)],
                "label": "smoke",
                "score": float(score_smoke),
            }
        ]

        # Event 2 (intermittent): "flame" candidate with moderate confidence.
        if frame_id % 30 < 20:
            fx1 = int(w * 0.62)
            fy1 = int(h * 0.58)
            fx2 = int(w * 0.72 + (frame_id % 15))
            fy2 = int(h * 0.74 + (frame_id % 10))
            detections.append(
                {
                    "box": [fx1, fy1, min(w - 1, fx2), min(h - 1, fy2)],
                    "label": "flame",
                    "score": float(0.45 + 0.015 * (frame_id % 20)),
                }
            )

        # Only keep labels that are in user prompts.
        prompt_set = set(p.lower() for p in prompts)
        return [d for d in detections if d["label"].lower() in prompt_set]

