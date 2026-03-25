# YOLOv8 + ByteTrack Zone Events (Intrusion & Loitering)
**Creator:** Adil Salah


<img width="873" height="468" alt="video-events-mvp result2" src="https://github.com/user-attachments/assets/e6633703-0977-401d-84ab-81b79e84427d" />

<img width="1001" height="521" alt="video-events-mvp result1" src="https://github.com/user-attachments/assets/5efecd9e-6bc5-45d1-9f9f-704c3328e243" />

<img width="1347" height="833" alt="Screenshot 2026-03-25 181123" src="https://github.com/user-attachments/assets/22d240e0-f76c-427d-b22a-93babf59c428" />


A compact computer vision project that detects and tracks people in video, then triggers **zone-based events** based on how long a person remains inside a defined polygon zone.

## Overview
This project uses **YOLOv8** for person detection and **ByteTrack** for stable multi-object tracking. It monitors a custom polygon zone and generates alerts for:

- **Intrusion**: a person remains inside the restricted zone for more than **1 second**
- **Loitering**: a person remains inside the zone for more than **15 seconds**

## Outputs
The system produces:
- An **annotated video** with bounding boxes, track IDs, timers, and zone overlay
- An `events.json` file containing timestamped alerts

## Demo
_Add demo video, screenshots, or GIF here._

## Features
- **YOLOv8** person detection using pretrained weights
- **ByteTrack** multi-object tracking for stable IDs
- **Polygon-based zone rules** loaded from `configs/zones.json`
- Clear visual overlays with color-coded states:
  - **Zone:** pink
  - **Normal:** green
  - **Intrusion:** yellow
  - **Loitering:** red
  - Returns to **green** when the person exits the zone

## Installation
Install the required dependencies:

```bash
pip install -r requirements.txt
