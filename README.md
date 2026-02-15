# AV Video Classifier — MVP

Focused pipeline for classifying autonomous vehicle fleet video by scenario type.
Uses **E5-V** (unified multimodal embeddings) for zero-shot and few-shot classification
in the same vector space.

## How It Works

E5-V (built on LLaVA-Next 8B) puts text, images, and text+images into a single
embedding space. This means:

- **Zero-shot**: Encode text labels → cosine similarity against image embeddings
- **Few-shot**: Encode text + verified example images together → one composed vector →
  cosine similarity against the *same* cached image embeddings

No re-embedding needed for few-shot. The image embeddings computed once in step 1
are reused throughout.

## Architecture

```
av-classify/
├── app/
│   ├── config.py           # All settings (env-configurable)
│   └── main.py             # Entry point
├── data/
│   ├── video_parser.py     # Filename → metadata parser
│   └── models.py           # Dataclasses + pre-defined use cases
├── pipeline/
│   ├── frame_extractor.py  # Extract N evenly-spaced frames per video
│   ├── classifier.py       # E5-V: text, image, and composed embeddings
│   ├── cache.py            # Disk cache for embeddings + results
│   └── runner.py           # Orchestrates full pipeline
├── ui/
│   └── gradio_app.py       # Tabbed Gradio UI
├── config/
│   └── crop_config.json    # Camera crop regions for 1920×1080 composites
└── cache/                   # Runtime: embeddings, results, labels
```

## Pipeline

1. **Frame Extraction** — 5 evenly-spaced frames × 4 cameras = 20 frames per timestamp
2. **E5-V Image Embedding** — Each frame → embedding vector (cached to disk)
3. **Zero-Shot Classification** — E5-V text embeddings for labels → cosine similarity → rank
4. **Human Verification** — Top candidates shown with thumbnails; user marks ✅/❌/⏭️
5. **Few-Shot Re-ranking** — E5-V composed embedding (text + example images) → one query vector → re-rank using same cached image embeddings

## Zero-Shot vs Few-Shot

| | Zero-Shot | Few-Shot |
|---|---|---|
| **Query** | Text label only | Text label + verified example images |
| **How** | `encode_text(label)` vs `encode_images(frames)` | `encode_composed(label, examples)` vs same `encode_images(frames)` |
| **Speed** | Fast | Equally fast (re-ranking is just cosine similarity) |
| **Accuracy** | Good for broad filtering | Better — query is grounded in real footage |
| **Re-embed?** | N/A | No — reuses cached image embeddings |

## Usage

```bash
# Install
pip install -r requirements.txt --break-system-packages

# Launch UI
python app/main.py

# Custom paths
VIDEO_DIR=/path/to/cropped CACHE_DIR=/path/to/cache python app/main.py
```

## Env Configuration

| Variable | Default | Description |
|---|---|---|
| `VIDEO_DIR` | `/workspace/video-inference/videos/cropped` | Cropped video directory |
| `CACHE_DIR` | `/workspace/cache` | Cache for embeddings, results, labels |
| `MODEL` | `royokong/e5-v` | E5-V model (or any LLaVA-Next checkpoint) |
| `DEVICE` | `cuda` | Compute device |
| `BATCH_SIZE` | `8` | Image embedding batch size |
| `MAX_POS_EXAMPLES` | `4` | Max positive images in composed query |
| `MAX_NEG_EXAMPLES` | `4` | Max negative images in composed query |
| `FRAMES_PER_VIDEO` | `5` | Frames extracted per camera video |

## Model

**E5-V** (`royokong/e5-v`) — LLaVA-Next 8B fine-tuned for universal multimodal embeddings.
~16GB VRAM in fp16. Produces embeddings for:

- Text only: `"<sent>\nSummary above sentence in one word: "`
- Image only: `"<image>\nSummary above image in one word: "`
- Composed: `"<image>\n{text}\nSummary above image in one word: "`

All three share the same vector space. The embedding is the last token's hidden state.

## Tabs

1. **Bus Stop Violations** — Cars parked at bus stops
2. **Sign Obstruction** — Traffic signs obscured by vegetation
3. **Pedestrian Crossings** — Jaywalking / unsafe crossings
4. **Construction Zones** — Roadwork and lane closures
5. **Custom Query** — User-defined zero-shot labels
