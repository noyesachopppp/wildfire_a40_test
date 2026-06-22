from __future__ import annotations

import os
from typing import Iterable

import cv2
import numpy as np


def _color_for_label(label: str) -> tuple[int, int, int]:
    palette = {
        "smoke": (180, 180, 180),
        "flame": (0, 120, 255),
        "fire": (0, 90, 255),
        "smoke plume": (210, 210, 210),
    }
    return palette.get(label.lower(), (0, 255, 255))


def draw_overlay(
    frame_bgr: np.ndarray,
    detections: list[dict],
    merged_mask: np.ndarray,
    risk_level: str,
    alpha: float = 0.35,
    box_thickness: int = 2,
) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    if merged_mask is not None and merged_mask.shape[:2] == (h, w):
        mask_color = np.zeros_like(out)
        mask_color[:, :, 2] = (merged_mask > 0).astype(np.uint8) * 255
        out = cv2.addWeighted(out, 1.0, mask_color, alpha, 0.0)

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        label = str(det["label"])
        score = float(det["score"])
        color = _color_for_label(label)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, box_thickness)
        cv2.putText(
            out,
            f"{label}:{score:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    risk_color = {"LOW": (0, 255, 0), "MEDIUM": (0, 215, 255), "HIGH": (0, 0, 255)}.get(
        risk_level,
        (255, 255, 255),
    )
    cv2.putText(
        out,
        f"RISK: {risk_level}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        risk_color,
        2,
        cv2.LINE_AA,
    )
    return out


def save_image(image_bgr: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image_bgr)

