from __future__ import annotations

from typing import Any

import numpy as np

from src.utils.logging_utils import get_logger


class YOLODetector:
    """
    YOLOv8 detector adapter for real-time edge detection.

    Output detection schema:
    {
      "label": "smoke" | "fire" | "flame",
      "score": float,
      "bbox": [x1, y1, x2, y2],
      "box": [x1, y1, x2, y2],  # compatibility key for current pipeline
      "frame_id": int | None,
      "timestamp": float | None,
      "source_model": "YOLOv8n" | "YOLOv8m"
    }
    """

    def __init__(
        self,
        model_name: str = "yolov8n",
        backend: str = "mock",
        conf_threshold: float = 0.35,
        device: str | None = None,
    ):
        self.logger = get_logger(self.__class__.__name__)
        self.model_name = str(model_name).lower()
        self.backend = str(backend).lower()
        self.conf_threshold = float(conf_threshold)
        self.device = device
        self.source_model = "YOLOv8m" if self.model_name == "yolov8m" else "YOLOv8n"
        self._model = None
        self._class_names: dict[int, str] = {}

        if self.backend == "ultralytics":
            self._initialize_ultralytics()
        elif self.backend != "mock":
            self.logger.warning("Unknown YOLO backend '%s'. Falling back to mock.", self.backend)
            self.backend = "mock"

    def _initialize_ultralytics(self) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError(
                "Failed to import ultralytics YOLO backend. Install with `pip install ultralytics` "
                "or use --yolo-backend mock."
            ) from exc

        weight_name = "yolov8m.pt" if self.model_name == "yolov8m" else "yolov8n.pt"
        self._model = YOLO(weight_name)
        names_obj = getattr(self._model, "names", {}) or {}
        if isinstance(names_obj, dict):
            self._class_names = {int(k): str(v) for k, v in names_obj.items()}
        self.logger.info(
            "YOLO ultralytics backend initialized: model=%s weights=%s device=%s",
            self.source_model,
            weight_name,
            self.device or "auto",
        )

    @staticmethod
    def _to_bbox_xyxy(values: Any, width: int, height: int) -> list[int]:
        arr = np.asarray(values, dtype=np.float32).reshape(-1).tolist()
        x1, y1, x2, y2 = [int(round(v)) for v in arr[:4]]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width - 1, x2))
        y2 = max(y1 + 1, min(height - 1, y2))
        return [x1, y1, x2, y2]

    @staticmethod
    def _map_coco_label(raw_label: str) -> str | None:
        # COCO models are not wildfire-specific; this mapping is for architecture/timing validation only.
        label = raw_label.strip().lower()
        if "smoke" in label:
            return "smoke"
        if "flame" in label:
            return "flame"
        if "fire" in label:
            return "fire"
        return None

    def detect(
        self,
        frame: np.ndarray,
        frame_id: int | None = None,
        timestamp: float | None = None,
    ) -> list[dict]:
        if self.backend == "mock":
            return self._mock_detect(frame, frame_id=frame_id, timestamp=timestamp)
        return self._ultralytics_detect(frame, frame_id=frame_id, timestamp=timestamp)

    def _mock_detect(
        self,
        frame: np.ndarray,
        frame_id: int | None = None,
        timestamp: float | None = None,
    ) -> list[dict]:
        h, w = frame.shape[:2]
        fid = int(frame_id or 0)
        phase = (fid // 30) % 6

        # Deterministic pseudo smoke box.
        dx = (fid * 7) % max(40, w // 5)
        x1 = int(w * 0.10 + dx)
        y1 = int(h * 0.18 + (fid * 3) % max(20, h // 12))
        bw = int(w * (0.16 + 0.015 * phase))
        bh = int(h * (0.13 + 0.010 * phase))
        smoke_box = [
            max(0, x1),
            max(0, y1),
            min(w - 1, x1 + bw),
            min(h - 1, y1 + bh),
        ]
        smoke_score = min(0.92, 0.32 + 0.08 * phase)

        detections: list[dict] = [
            {
                "label": "smoke",
                "score": float(smoke_score),
                "bbox": smoke_box,
                "box": smoke_box,
                "frame_id": frame_id,
                "timestamp": timestamp,
                "source_model": self.source_model,
            }
        ]

        # Intermittent flame candidate to create MEDIUM/HIGH event patterns.
        if phase in {3, 4, 5}:
            fx1 = int(w * 0.60)
            fy1 = int(h * 0.55)
            fw = int(w * (0.08 + 0.01 * (phase - 2)))
            fh = int(h * (0.10 + 0.01 * (phase - 2)))
            flame_box = [fx1, fy1, min(w - 1, fx1 + fw), min(h - 1, fy1 + fh)]
            flame_score = min(0.95, 0.50 + 0.10 * (phase - 3))
            detections.append(
                {
                    "label": "flame",
                    "score": float(flame_score),
                    "bbox": flame_box,
                    "box": flame_box,
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "source_model": self.source_model,
                }
            )

        return [d for d in detections if float(d["score"]) >= self.conf_threshold]

    def _ultralytics_detect(
        self,
        frame: np.ndarray,
        frame_id: int | None = None,
        timestamp: float | None = None,
    ) -> list[dict]:
        if self._model is None:
            raise RuntimeError("Ultralytics YOLO backend is not initialized.")

        h, w = frame.shape[:2]
        kwargs: dict[str, Any] = {"conf": self.conf_threshold, "verbose": False}
        if self.device:
            kwargs["device"] = self.device
        results = self._model(frame, **kwargs)

        detections: list[dict] = []
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = getattr(boxes, "xyxy", None)
            conf = getattr(boxes, "conf", None)
            cls = getattr(boxes, "cls", None)
            if xyxy is None or conf is None or cls is None:
                continue
            xyxy_np = xyxy.detach().cpu().numpy()
            conf_np = conf.detach().cpu().numpy()
            cls_np = cls.detach().cpu().numpy()
            for idx in range(len(xyxy_np)):
                score = float(conf_np[idx])
                class_id = int(cls_np[idx])
                raw_label = self._class_names.get(class_id, str(class_id))
                mapped = self._map_coco_label(raw_label)
                if mapped is None:
                    continue
                bbox = self._to_bbox_xyxy(xyxy_np[idx], width=w, height=h)
                detections.append(
                    {
                        "label": mapped,
                        "score": score,
                        "bbox": bbox,
                        "box": bbox,
                        "frame_id": frame_id,
                        "timestamp": timestamp,
                        "source_model": self.source_model,
                    }
                )
        return detections

