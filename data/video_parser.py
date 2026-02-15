"""Parse structured metadata from video filenames.

Naming convention
─────────────────
Composite:  m001-20260202-1770014818_video.mp4
Cropped:    m001-20260202-1770014818_video_front_camera.mp4

Reused verbatim from the AV Video Search Agent repo.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

KNOWN_CAMERAS = [
    "front_camera",
    "left_fisheye",
    "right_fisheye",
    "bev_visualization",
    "telemetry_hud",
    "rear_camera",
]

_PATTERN = re.compile(
    r"^(?P<vehicle>[a-zA-Z0-9]+)"
    r"-(?P<date>\d{8})"
    r"-(?P<ts>\d+)"
    r"_video"
    r"(?:_(?P<camera>[a-z_]+))?"
    r"\.[a-zA-Z0-9]+$"
)


def parse_video_filename(filename: str) -> dict | None:
    basename = Path(filename).name
    m = _PATTERN.match(basename)
    if m is None:
        return None

    vehicle_id = m.group("vehicle")
    date_raw = m.group("date")
    ts = int(m.group("ts"))
    camera_raw = m.group("camera") or None

    try:
        date_str = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
    except (IndexError, ValueError):
        return None

    try:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        datetime_utc = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OSError, OverflowError, ValueError):
        datetime_utc = f"{date_str}T00:00:00Z"

    camera = camera_raw if camera_raw else None
    source_id = f"{vehicle_id}-{date_raw}-{ts}"

    return {
        "vehicle_id": vehicle_id,
        "date": date_str,
        "timestamp": ts,
        "datetime_utc": datetime_utc,
        "camera": camera,
        "source_id": source_id,
    }
