from typing import Tuple

import cv2
import numpy as np


def preprocess_frame(frame_bgr: np.ndarray, resize_to: Tuple[int, int] | None = None) -> np.ndarray:
    """
    Basic frame preprocessing.

    - Optionally resize for stable throughput.
    - Apply mild denoising to reduce high-frequency noise.
    """
    processed = frame_bgr.copy()
    if resize_to is not None:
        processed = cv2.resize(processed, resize_to, interpolation=cv2.INTER_LINEAR)

    # Keep denoising conservative to avoid blurring smoke edges too much.
    processed = cv2.GaussianBlur(processed, (3, 3), 0)
    return processed

