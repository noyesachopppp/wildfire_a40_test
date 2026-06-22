from __future__ import annotations


class RuleBasedRiskEngine:
    """Compute LOW/MEDIUM/HIGH wildfire risk from frame-level + temporal signals."""

    def __init__(
        self,
        detection_confidence: float = 0.35,
        high_risk_confidence: float = 0.65,
        min_mask_area_ratio: float = 0.005,
        high_growth_ratio: float = 0.20,
    ):
        self.detection_confidence = detection_confidence
        self.high_risk_confidence = high_risk_confidence
        self.min_mask_area_ratio = min_mask_area_ratio
        self.high_growth_ratio = high_growth_ratio

    def evaluate(
        self,
        avg_confidence: float,
        mask_area_ratio: float,
        mask_growth: float,
        consecutive_frames: int,
        region_count: int,
    ) -> tuple[str, str]:
        # HIGH: strong confidence + growth + persistence
        if (
            avg_confidence >= self.high_risk_confidence
            and consecutive_frames >= 3
            and mask_growth >= self.high_growth_ratio
            and mask_area_ratio >= self.min_mask_area_ratio
        ):
            msg = (
                "Potential wildfire detected near camera region. "
                f"Signal persisted for {consecutive_frames} consecutive frames and "
                f"mask area increased by {mask_growth * 100:.1f}%. Risk level: HIGH. "
                "Operator verification is recommended."
            )
            return "HIGH", msg

        # MEDIUM: meaningful confidence + persistence or meaningful area
        if (
            avg_confidence >= self.detection_confidence
            and (consecutive_frames >= 2 or mask_area_ratio >= self.min_mask_area_ratio * 2.5)
        ):
            msg = (
                "Possible wildfire indicator detected. "
                f"Consecutive frames: {consecutive_frames}, regions: {region_count}, "
                f"mask area: {mask_area_ratio * 100:.2f}%. Risk level: MEDIUM. "
                "Continue monitoring and prepare manual confirmation."
            )
            return "MEDIUM", msg

        msg = (
            "Weak or isolated signal detected. "
            f"Consecutive frames: {consecutive_frames}, confidence: {avg_confidence:.2f}. "
            "Risk level: LOW."
        )
        return "LOW", msg

