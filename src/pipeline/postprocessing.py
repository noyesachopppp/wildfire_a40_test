from __future__ import annotations

from typing import Iterable, List

import cv2
import numpy as np


def filter_detections(detections: Iterable[dict], confidence_threshold: float) -> List[dict]:
    """Keep detections whose confidence >= threshold."""
    return [d for d in detections if float(d.get("score", 0.0)) >= confidence_threshold]


def merge_masks(masks: list[np.ndarray], image_shape: tuple[int, int]) -> np.ndarray:
    """Merge multiple binary masks into a single binary mask."""
    h, w = image_shape
    merged = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        if m is None:
            continue
        merged = np.maximum(merged, (m > 0).astype(np.uint8))
    return merged


def mask_area_ratio(mask: np.ndarray) -> float:
    """Ratio of foreground pixels to total pixels."""
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(mask.size)


def count_regions(mask: np.ndarray) -> int:
    """Count connected components excluding background."""
    if np.count_nonzero(mask) == 0:
        return 0
    num_labels, _ = cv2.connectedComponents((mask > 0).astype(np.uint8))
    return max(0, int(num_labels) - 1)


def compute_growth(current_ratio: float, previous_ratio: float | None) -> float:
    """
    Compute relative growth compared to previous frame ratio.

    Returns:
        e.g. 0.37 means +37% growth.
    """
    if previous_ratio is None or previous_ratio <= 0:
        return 0.0
    return (current_ratio - previous_ratio) / previous_ratio

