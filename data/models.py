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
    description="Detect vehicles illegally parked or stopped at bus stops",
    labels=[
        "a bus stop with a car or vehicle parked at it",
        "a bus stop with no car or vehicle parked at it",
        "a road scene with no bus stop visible",
    ],
    target_label="a bus stop with a car or vehicle parked at it",
)

SIGN_MAINTENANCE_USECASE = UseCase(
    name="Sign Maintenance",
    description="Traffic signs that need maintenance — damaged, faded, tilted, obscured by vegetation, or missing",
    labels=[
        "a traffic sign that is damaged bent faded or needs repair",
        "a traffic sign obscured or hidden by overgrown vegetation",
        "a clearly visible traffic sign in good condition",
        "a road scene with no traffic signs",
    ],
    target_label="a traffic sign that is damaged bent faded or needs repair",
)

STREET_VANDALISM_USECASE = UseCase(
    name="Street Damage & Vandalism",
    description="Graffiti, vandalism on walls or street furniture, damaged infrastructure, illegal dumping",
    labels=[
        "graffiti or vandalism on a wall building or street furniture",
        "damaged or broken street infrastructure like benches signs or poles",
        "illegal dumping or trash accumulation on the street",
        "a clean street scene with no vandalism or damage",
    ],
    target_label="graffiti or vandalism on a wall building or street furniture",
)

ALL_USECASES = [
    BUS_STOP_USECASE,
    SIGN_MAINTENANCE_USECASE,
    STREET_VANDALISM_USECASE,
]
