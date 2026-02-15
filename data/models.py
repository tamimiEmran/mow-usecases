"""Core data models for the classification pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FrameInfo:
    """A single extracted frame."""
    video_path: str
    source_id: str
    camera: str
    frame_idx: int
    timestamp_s: float


@dataclass
class ClassificationResult:
    """Zero-shot or few-shot classification for one frame."""
    frame: FrameInfo
    scores: dict[str, float]       # label → probability
    predicted_label: str
    confidence: float


@dataclass
class TimestampResult:
    """Aggregated classification for one timestamp (all cameras, all frames)."""
    source_id: str
    vehicle_id: str
    date: str
    datetime_utc: str
    frame_results: list[ClassificationResult]
    avg_scores: dict[str, float]   # label → mean probability across all frames
    predicted_label: str
    confidence: float              # confidence of predicted_label
    # Paths for UI display
    camera_frames: dict[str, list[str]]  # camera → list of frame image paths


@dataclass
class VerificationLabel:
    """Human verification label for a timestamp."""
    source_id: str
    label: str                     # "positive" | "negative" | "skip"
    notes: str = ""


@dataclass
class UseCase:
    """Definition of a classification use case (one tab)."""
    name: str
    description: str
    labels: list[str]              # classification labels
    target_label: str              # which label is the "positive" class
    few_shot_labels: list[str] | None = None  # override labels for few-shot


# ── Pre-defined use cases ────────────────────────────────────────

BUS_STOP_USECASE = UseCase(
    name="Bus Stop Violations",
    description="Detect vehicles parked at bus stops",
    labels=[
        "a bus stop with a car or vehicle parked at it",
        "a bus stop with no car or vehicle parked at it",
        "a road scene with no bus stop visible",
    ],
    target_label="a bus stop with a car or vehicle parked at it",
)

SIGN_OBSTRUCTION_USECASE = UseCase(
    name="Sign Obstruction",
    description="Traffic signs obscured by vegetation or objects",
    labels=[
        "a traffic sign partially or fully obscured by trees or vegetation",
        "a clearly visible traffic sign with no obstruction",
        "a road scene with no traffic signs",
    ],
    target_label="a traffic sign partially or fully obscured by trees or vegetation",
)

PEDESTRIAN_CROSSING_USECASE = UseCase(
    name="Pedestrian Crossings",
    description="Pedestrians crossing outside crosswalks",
    labels=[
        "a pedestrian jaywalking or crossing outside a crosswalk",
        "a pedestrian using a crosswalk properly",
        "a road scene with no pedestrians crossing",
    ],
    target_label="a pedestrian jaywalking or crossing outside a crosswalk",
)

CONSTRUCTION_USECASE = UseCase(
    name="Construction Zones",
    description="Roadwork, construction zones, lane closures",
    labels=[
        "a road construction zone with barriers cones or workers",
        "a normal road with no construction",
        "a road scene with minor obstructions",
    ],
    target_label="a road construction zone with barriers cones or workers",
)

ALL_USECASES = [
    BUS_STOP_USECASE,
    SIGN_OBSTRUCTION_USECASE,
    PEDESTRIAN_CROSSING_USECASE,
    CONSTRUCTION_USECASE,
]
