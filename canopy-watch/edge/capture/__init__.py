"""Frame acquisition service.

    from edge.capture import CaptureService
    from edge.config.loader import get_config

    with CaptureService.from_config(get_config()) as capture:
        frame = capture.next_for_tier1()          # sampled, low latency
        original = capture.get_frame(frame.frame_id)   # exact, for Tier2
"""

from .buffer import (
    CaptureError,
    FrameBuffer,
    FrameExpired,
    FrameNotFound,
    FrameNotYet,
    FrameRetrievalError,
)
from .frame import CapturedFrame, make_frame_id, parse_frame_id
from .sampler import LatestFrameQueue, TimeBasedSampler
from .service import DEFAULTS, CaptureService

__all__ = [
    "DEFAULTS",
    "CaptureError",
    "CaptureService",
    "CapturedFrame",
    "FrameBuffer",
    "FrameExpired",
    "FrameNotFound",
    "FrameNotYet",
    "FrameRetrievalError",
    "LatestFrameQueue",
    "TimeBasedSampler",
    "make_frame_id",
    "parse_frame_id",
]