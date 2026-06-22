from __future__ import annotations

from typing import List


class VLMExplainer:
    """
    VLM integration interface.

    TODO:
      - Replace placeholder logic with an actual VLM backend (local or API).
      - Keep the same `explain(...)` signature for plug-and-play replacement.
    """

    def __init__(self, mode: str = "placeholder"):
        self.mode = mode

    def explain(
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

