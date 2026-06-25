from __future__ import annotations

import os
from typing import List

import cv2
import numpy as np

from src.utils.logging_utils import get_logger


class SAM2Wrapper:
    """
    SAM2 segmentation adapter.

    Input:
      - frame image
      - detection boxes from GroundingDINO
    Output:
      - list of binary masks (uint8, shape HxW)
    """

    def __init__(
        self,
        mock: bool = False,
        checkpoint: str | None = None,
        model_config: str | None = None,
        device: str | None = None,
    ):
        self.logger = get_logger(self.__class__.__name__)
        self.mock = mock
        self.checkpoint = checkpoint
        self.model_config = model_config
        self.device = device
        self.predictor = None
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
                "SAM2 checkpoint path is missing in config. "
                "Set models.sam2.checkpoint or run with --mock."
            )
        if self._is_unset_path(self.model_config):
            raise FileNotFoundError(
                "SAM2 model config path is missing in config. "
                "Set models.sam2.model_config or run with --mock."
            )
        assert self.checkpoint is not None
        assert self.model_config is not None
        if not os.path.exists(self.checkpoint):
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: {self.checkpoint}. "
                "Verify the path or run with --mock."
            )
        if not os.path.exists(self.model_config):
            raise FileNotFoundError(
                f"SAM2 model config not found: {self.model_config}. "
                "Verify the path or run with --mock."
            )

    def _initialize_real_model(self) -> None:
        self._validate_required_files()
        try:
            import torch
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except Exception as exc:
            raise RuntimeError(
                "Failed to import SAM2 dependencies. "
                "Install SAM2/Torch for real inference or run with --mock."
            ) from exc

        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        assert self.model_config is not None
        assert self.checkpoint is not None
        sam_model = build_sam2(self.model_config, self.checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(sam_model)
        self.logger.info("SAM2 real backend initialized on device=%s", self.device)

    def segment(
        self,
        image_bgr: np.ndarray,
        detections: list[dict],
        frame_id: int,
    ) -> List[np.ndarray]:
        if self.mock:
            return self._mock_segment(image_bgr, detections, frame_id)

        return self._real_segment(image_bgr, detections)

    @staticmethod
    def _extract_box(det: dict) -> list[int]:
        raw = det.get("box", det.get("bbox"))
        if raw is None:
            return [0, 0, 1, 1]
        return [int(v) for v in raw]

    def _real_segment(self, image_bgr: np.ndarray, detections: list[dict]) -> List[np.ndarray]:
        if self.predictor is None:
            raise RuntimeError(
                "SAM2 real backend is not initialized. "
                "Provide valid model files or run with --mock."
            )
        if not detections:
            return []

        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        boxes: list[list[float]] = []
        for det in detections:
            x1, y1, x2, y2 = self._extract_box(det)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            boxes.append([float(x1), float(y1), float(x2), float(y2)])
        if not boxes:
            return []

        self.predictor.set_image(image_rgb)
        masks, _, _ = self.predictor.predict(
            point_coords=None,
            point_labels=None,
            box=np.array(boxes, dtype=np.float32),
            multimask_output=False,
        )

        if masks is None:
            return self._box_fill_fallback(image_bgr, detections)

        if masks.ndim == 4:
            # Expected shape can be N x 1 x H x W when multimask_output=False.
            masks = masks[:, 0, :, :]
        elif masks.ndim == 2:
            # Single-mask fallback shape H x W.
            masks = np.expand_dims(masks, axis=0)

        outputs: list[np.ndarray] = []
        for m in masks:
            outputs.append((m > 0).astype(np.uint8))
        return outputs

    def _box_fill_fallback(self, image_bgr: np.ndarray, detections: list[dict]) -> List[np.ndarray]:
        h, w = image_bgr.shape[:2]
        masks: list[np.ndarray] = []
        for det in detections:
            x1, y1, x2, y2 = self._extract_box(det)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            m = np.zeros((h, w), dtype=np.uint8)
            m[y1:y2, x1:x2] = 1
            masks.append(m)
        return masks

    def _mock_segment(self, image_bgr: np.ndarray, detections: list[dict], frame_id: int) -> List[np.ndarray]:
        h, w = image_bgr.shape[:2]
        masks: list[np.ndarray] = []
        for det in detections:
            x1, y1, x2, y2 = self._extract_box(det)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            mask = np.zeros((h, w), dtype=np.uint8)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            base_rx = max(3, (x2 - x1) // 2)
            base_ry = max(3, (y2 - y1) // 2)

            # Simulate temporal growth for smoke-like regions.
            growth = 1.0 + min(0.4, (frame_id % 120) / 300.0)
            rx = int(base_rx * growth)
            ry = int(base_ry * growth)

            cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, color=1, thickness=-1)
            masks.append(mask)
        return masks

