
# Run this from the project root like:
#   python -m src.run --video testing.mp4 --zones configs/zones.json

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from .geometry import bbox_center_xyxy, point_in_polygon
from .event_logic import EventEngine
from .viz import draw_zone, draw_person, draw_legend


def load_zone(zones_path: str) -> Tuple[str, List[Tuple[int, int]]]:
    """Read zones.json and return (zone_name, polygon_points)."""
    with open(zones_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    zone_name = data.get("zone_name", "restricted")
    poly = data.get("polygon", [])
    polygon = [(int(x), int(y)) for x, y in poly]

    if len(polygon) < 3:
        raise ValueError("zones.json polygon must have at least 3 points")

    return zone_name, polygon


def read_video_meta(video_path: str) -> Tuple[float, int, int]:
    """
    Grab fps/width/height from OpenCV.
    Some videos lie about FPS, so we fallback to 30 if needed.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return float(fps), width, height


def make_writer(out_path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """Create a video writer (mp4v works on most Windows setups)."""
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video writer at: {out_path}")

    return writer


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8 + ByteTrack zone events (intrusion/loitering).")
    parser.add_argument("--video", required=True, help="Input video file (mp4)")
    parser.add_argument("--zones", required=True, help="Path to configs/zones.json")
    parser.add_argument("--out", default="outputs/out.mp4", help="Output annotated video path")
    parser.add_argument("--log", default="outputs/events.json", help="Output events JSON path")

    parser.add_argument("--model", default="yolov8n.pt", help="YOLO model (yolov8n.pt is fast)")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Tracker config name/path for Ultralytics")

    parser.add_argument("--intrusion_sec", type=float, default=1.0, help="Intrusion threshold (seconds)")
    parser.add_argument("--loiter_sec", type=float, default=15.0, help="Loitering threshold (seconds)")
    parser.add_argument("--missing_tol", type=float, default=1.0, help="Allowed tracker dropout (seconds)")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")

    args = parser.parse_args()

    zone_name, polygon = load_zone(args.zones)
    fps, width, height = read_video_meta(args.video)
    writer = make_writer(args.out, fps, width, height)

    print(f"[INFO] Zone: {zone_name} | points={len(polygon)}")
    print(f"[INFO] Video: fps={fps:.2f}, size={width}x{height}")
    print(f"[INFO] Output: {args.out}")

    # YOLO model
    model = YOLO(args.model)

    # Event engine
    engine = EventEngine(
        intrusion_sec=args.intrusion_sec,
        loiter_sec=args.loiter_sec,
        missing_tolerance_sec=args.missing_tol,
    )

    all_events = []
    frame_idx = 0

    # Stream results frame-by-frame
    results_stream = model.track(
        source=args.video,
        stream=True,
        persist=True,
        tracker=args.tracker,
        conf=args.conf,
        verbose=False,
    )

    for res in results_stream:
        # Time in seconds based on frame index (simple + consistent)
        now = frame_idx / fps
        frame_idx += 1

        frame = res.orig_img.copy()

        # Draw zone polygon in pink
        draw_zone(frame, polygon)

        # Collect observations for event logic
        observations = []          # (track_id, inside_zone)
        draw_list = []             # keep bbox info to draw after we compute status

        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes

            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)

            # IDs can be None for early frames; we skip until they're ready
            if boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)
            else:
                ids = None

            if ids is not None:
                for i in range(len(xyxy)):
                    # COCO "person" is class 0
                    if cls[i] != 0:
                        continue

                    tid = int(ids[i])
                    x1, y1, x2, y2 = xyxy[i].astype(int).tolist()

                    cx, cy = bbox_center_xyxy(x1, y1, x2, y2)
                    inside = point_in_polygon((cx, cy), polygon)

                    observations.append((tid, inside))
                    draw_list.append((tid, inside, (x1, y1, x2, y2)))

        # Update events
        time_in_zone, status, new_events = engine.update(now, observations)
        if new_events:
            all_events.extend(new_events)

        # Draw persons using your color rules
        for tid, inside, bbox in draw_list:
            st = status.get(tid, "normal")
            t_in = time_in_zone.get(tid, 0.0)
            draw_person(frame, bbox, tid, inside, st, t_in)

        draw_legend(frame)
        writer.write(frame)

    writer.release()

    # Save events log
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(all_events, f, indent=2)

    print(f"[DONE] Saved video: {args.out}")
    print(f"[DONE] Saved events: {args.log} (count={len(all_events)})")


if __name__ == "__main__":
    main()
