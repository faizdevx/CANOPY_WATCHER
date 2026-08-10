"""The unit the Capture Service owns.

A driver `Frame` is "some pixels the camera produced". A `CapturedFrame` is
"pixels this node has taken responsibility for" — it carries an identity that
survives camera reconnects and three timestamps that answer three different
questions.

Timestamps, and why there are three:

    media_timestamp     when the sensor exposed the frame. Nullable — only the
                        CSI paths actually have one. NEVER faked from the
                        receive time: absent is debuggable, fabricated is not.
    received_timestamp  time.monotonic() when capture read it. Immune to NTP
                        steps, so this is what latency math uses. Meaningless
                        after a reboot and meaningless across nodes.
    wall_clock          UTC epoch seconds. The only one that can leave the node —
                        the Stream Processor correlates across cameras, and
                        events can sit in the outbox for hours while offline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

#: cam01-000184392 — source id, then a zero-padded sequence
FRAME_ID_PATTERN = re.compile(r"^(?P<source>.+)-(?P<seq>\d{9,})$")
SEQUENCE_WIDTH = 9


def make_frame_id(source_id: str, sequence: int) -> str:
    return f"{source_id}-{sequence:0{SEQUENCE_WIDTH}d}"


def parse_frame_id(frame_id: str) -> Optional[Tuple[str, int]]:
    """(source_id, sequence), or None if the string isn't a frame id at all."""
    if not isinstance(frame_id, str):
        return None
    match = FRAME_ID_PATTERN.match(frame_id)
    if not match:
        return None
    return match.group("source"), int(match.group("seq"))


@dataclass(frozen=True)
class CapturedFrame:
    frame_id: str
    source_id: str
    sequence: int

    data: np.ndarray                 # RGB HxWx3 uint8, READ-ONLY once buffered
    received_timestamp: float        # monotonic
    wall_clock: float                # unix epoch
    media_timestamp: Optional[float] = None   # sensor time, or None

    driver: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def nbytes(self) -> int:
        return int(self.data.nbytes)

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    def age(self, now: float) -> float:
        """Seconds since this frame was received (pass time.monotonic())."""
        return now - self.received_timestamp

    def describe(self) -> Dict[str, Any]:
        """Everything except the pixels — safe to log or put in an event."""
        return {
            "frame_id": self.frame_id,
            "source_id": self.source_id,
            "sequence": self.sequence,
            "width": self.width,
            "height": self.height,
            "received_timestamp": self.received_timestamp,
            "wall_clock": self.wall_clock,
            "media_timestamp": self.media_timestamp,
            "driver": self.driver,
        }

    def __repr__(self) -> str:
        return (f"CapturedFrame({self.frame_id}, {self.width}x{self.height}, "
                f"media_ts={self.media_timestamp})")


def freeze(arr: np.ndarray) -> np.ndarray:
    """Mark a frame array read-only before it enters the shared buffer.

    `CapturedFrame` is a frozen dataclass, but that does nothing for the numpy
    array inside it. Capture hands out references rather than 6 MB copies, so a
    downstream service drawing a bbox onto a retrieved frame would silently
    corrupt shared history. This turns that into an immediate, obvious error.
    """
    if arr.flags.writeable:
        try:
            arr.flags.writeable = False
        except ValueError:              # a view whose base is writeable
            arr = arr.copy()
            arr.flags.writeable = False
    return arr