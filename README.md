# YOLOv8 + ByteTrack Zone Events (Intrusion & Loitering)

A small, clean computer vision project that detects and tracks people in video, then triggers **zone-based events**:
- **Intrusion**: person stays inside a restricted polygon zone for > 1s
- **Loitering**: person stays inside the zone for > 15s

Outputs:
- Annotated video (boxes, IDs, timers, zone overlay)
- `events.json` with timestamped alerts

## Demo
(Add a GIF or short mp4 here later)

## Features
- YOLOv8 person detection (pretrained)
- ByteTrack tracking (stable IDs)
- Polygon zone rules (from `configs/zones.json`)
- Human-friendly overlays + color rules:
  - Zone: **pink**
  - Normal: **green**
  - Intrusion: **yellow**
  - Loitering: **red**
  - Back to green when outside

## Install
```bash
pip install -r requirements.txt
