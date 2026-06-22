from dataclasses import dataclass
from typing import Generator

import cv2


@dataclass
class FramePacket:
    frame_id: int
    timestamp: float
    fps: float
    image_bgr: "cv2.typing.MatLike"


def sample_video_frames(video_path: str, frame_stride: int = 10) -> Generator[FramePacket, None, None]:
    """
    Sample frames from a video every `frame_stride` frames.

    Args:
        video_path: Input video file path.
        frame_stride: Keep one frame every N frames.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0

    frame_idx = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_idx % frame_stride == 0:
            timestamp = frame_idx / fps
            yield FramePacket(
                frame_id=frame_idx,
                timestamp=timestamp,
                fps=fps,
                image_bgr=frame,
            )
        frame_idx += 1

    cap.release()

