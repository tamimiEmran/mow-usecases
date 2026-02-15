# AV Video Classifier — MVP

Focused pipeline for classifying autonomous vehicle fleet video by scenario type.
Uses SigLIP zero-shot classification → human verification → few-shot refinement.

## Architecture

```
av-classify/
├── app/
│   ├── config.py          # All settings (env-configurable)
│   └── main.py            # Entry point
├── data/
│   ├── video_parser.py    # Filename → metadata parser
│   └── models.py          # Dataclasses
├── pipeline/
│   ├── frame_extractor.py # Extract N evenly-spaced frames per video
│   ├── classifier.py      # SigLIP zero-shot + few-shot prototype classifier
│   ├── cache.py           # JSON disk cache for embeddings + results
│   └── runner.py          # Orchestrates full pipeline
├── ui/
│   └── gradio_app.py      # Tabbed Gradio UI (5 tabs, 1 implemented)
├── config/
│   └── crop_config.json   # Camera crop regions for 1920×1080 composites
└── cache/                  # Runtime: embeddings, results, labels
```

## Pipeline

1. **Frame Extraction** — 5 evenly-spaced frames × 6 cameras = 30 frames per timestamp
2. **Zero-Shot Classification** — SigLIP encodes frames + text labels, cosine similarity → class probabilities
3. **Aggregation** — Per-timestamp scores averaged across all 30 frames, sorted by target class confidence
4. **Human Verification** — Top candidates shown with thumbnails; user marks ✓/✗
5. **Few-Shot Refinement** — Verified examples become class prototypes; re-classify using nearest-prototype

## Usage

```bash
# Install
pip install -r requirements.txt --break-system-packages

# Launch UI
python app/main.py

# Or specify paths
VIDEO_DIR=/path/to/cropped CACHE_DIR=/path/to/cache python app/main.py
```

## Tabs (planned)

1. **Bus Stop Violations** — Cars parked at bus stops (implemented)
2. **Sign Obstruction** — Traffic signs obscured by vegetation
3. **Pedestrian Crossings** — Jaywalking / unsafe crossings
4. **Construction Zones** — Roadwork and lane closures
5. **Custom Query** — User-defined zero-shot labels
