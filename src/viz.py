
# Drawing code (boxes, zone polygon, and a tiny legend).

from __future__ import annotations
from typing import Dict, List, Tuple

import cv2
import numpy as np

PointI = Tuple[int, int]


# OpenCV uses BGR, not RGB.
PINK = (255, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)


def draw_zone(frame: np.ndarray, polygon: List[PointI]) -> None:
    """Restricted area polygon in pink."""
    pts = np.array(polygon, dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=PINK, thickness=2)


def _person_box_color(inside: bool, status: str) -> Tuple[int, int, int]:
    """
    Your rules:
      - Default person: GREEN
      - Inside & intruder: YELLOW
      - Inside & loitering: RED
      - Outside: GREEN (always)
    """
    if not inside:
        return GREEN

    if status == "loitering":
        return RED
    if status == "intruder":
        return YELLOW
    return GREEN


def draw_person(
    frame: np.ndarray,
    bbox_xyxy: Tuple[int, int, int, int],
    track_id: int,
    inside: bool,
    status: str,
    time_in_zone: float,
) -> None:
    """Draw bbox + ID + timer text."""
    x1, y1, x2, y2 = bbox_xyxy
    color = _person_box_color(inside, status)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Nice compact label
    label = f"ID {track_id} | {time_in_zone:.1f}s"
    if inside and status == "intruder":
        label += " | INTRUSION"
    elif inside and status == "loitering":
        label += " | LOITERING"

    # Put label above the box if possible
    y_text = max(18, y1 - 6)
    cv2.putText(frame, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def draw_legend(frame: np.ndarray) -> None:
    """Small legend so your demo is self-explanatory."""
    x, y = 10, 22
    cv2.putText(frame, "Legend:", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

    y += 22
    cv2.putText(frame, "GREEN = normal", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2)
    y += 20
    cv2.putText(frame, "YELLOW = intrusion", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, YELLOW, 2)
    y += 20
    cv2.putText(frame, "RED = loitering", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, RED, 2)
    y += 20
    cv2.putText(frame, "PINK = restricted zone", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, PINK, 2)
