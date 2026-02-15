"""Frame extraction from cropped videos.

Extracts N evenly-spaced frames from each camera video,
grouped by source_id (timestamp).
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.config import settings
from data.models import FrameInfo
from data.video_parser import parse_video_filename

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def discover_timestamps(video_dir: str | None = None) -> dict[str, dict[str, str]]:
    """Scan directory and group videos by source_id.

    Returns
    -------
    dict[str, dict[str, str]]
        source_id → {camera: video_path, ...}
    """
    video_dir = video_dir or settings.storage.video_dir
    base = Path(video_dir)
    if not base.exists():
        logger.warning("Video directory does not exist: %s", base)
        return {}

    groups: dict[str, dict[str, str]] = defaultdict(dict)

    for path in sorted(base.iterdir()):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        parsed = parse_video_filename(path.name)
        if parsed is None or parsed["camera"] is None:
            continue
        # Skip non-real cameras
        if parsed["camera"] in ("bev_visualization", "telemetry_hud"):
            continue
        groups[parsed["source_id"]][parsed["camera"]] = str(path)

    logger.info("Discovered %d timestamps across %d video files",
                len(groups), sum(len(v) for v in groups.values()))
    return dict(groups)


def extract_frames(
    video_path: str,
    n_frames: int = 5,
) -> list[tuple[Image.Image, FrameInfo]]:
    """Extract N evenly-spaced frames from a video.

    Returns list of (PIL.Image, FrameInfo) tuples.
    """
    parsed = parse_video_filename(Path(video_path).name)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Cannot open video: %s", video_path)
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 10.0

    if total_frames < n_frames:
        indices = list(range(total_frames))
    else:
        # Evenly spaced, excluding very first and last frame
        indices = [
            int(i * (total_frames - 1) / (n_frames - 1))
            for i in range(n_frames)
        ]

    results: list[tuple[Image.Image, FrameInfo]] = []

    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        info = FrameInfo(
            video_path=video_path,
            source_id=parsed["source_id"] if parsed else Path(video_path).stem,
            camera=parsed["camera"] if parsed else "",
            frame_idx=frame_idx,
            timestamp_s=round(frame_idx / video_fps, 2),
        )
        results.append((pil_image, info))

    cap.release()
    return results


def extract_all_frames_for_timestamp(
    camera_paths: dict[str, str],
    n_frames: int = 5,
    cameras: list[str] | None = None,
) -> list[tuple[Image.Image, FrameInfo]]:
    """Extract frames from all cameras for one timestamp.

    Parameters
    ----------
    camera_paths : dict[str, str]
        camera_name → video_path
    n_frames : int
        Frames per camera video.
    cameras : list[str] | None
        Which cameras to include. None = use config default.

    Returns
    -------
    list of (PIL.Image, FrameInfo)
    """
    cameras = cameras or settings.pipeline.cameras
    all_frames = []

    for cam in cameras:
        if cam not in camera_paths:
            continue
        frames = extract_frames(camera_paths[cam], n_frames)
        all_frames.extend(frames)

    return all_frames


def save_frame_thumbnail(
    image: Image.Image,
    cache_dir: str,
    source_id: str,
    camera: str,
    frame_idx: int,
    max_size: tuple[int, int] = (320, 240),
) -> str:
    """Save a frame as a thumbnail and return the path."""
    thumbs_dir = Path(cache_dir) / "thumbnails" / source_id
    thumbs_dir.mkdir(parents=True, exist_ok=True)

    thumb = image.copy()
    thumb.thumbnail(max_size, Image.LANCZOS)

    filename = f"{camera}_{frame_idx:04d}.jpg"
    path = thumbs_dir / filename
    thumb.save(path, "JPEG", quality=85)
    return str(path)
