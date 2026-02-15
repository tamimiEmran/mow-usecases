"""AV Video Classifier — Entry Point.

Usage:
    python app/main.py
    VIDEO_DIR=/path/to/videos python app/main.py
    MODEL=royokong/e5-v python app/main.py
"""

import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    from app.config import settings
    from ui.gradio_app import create_ui

    logger.info("Starting AV Video Classifier")
    logger.info("  Video dir:    %s", settings.storage.video_dir)
    logger.info("  Cache dir:    %s", settings.storage.cache_dir)
    logger.info("  Model:        %s", settings.model.model_name)
    logger.info("  Device:       %s", settings.model.device)
    logger.info("  Frames/vid:   %d", settings.pipeline.frames_per_video)
    logger.info("  Few-shot:     %d pos / %d neg max examples",
                settings.fewshot.max_positive_examples,
                settings.fewshot.max_negative_examples)

    app = create_ui()
    app.launch(
        server_name=settings.server.host,
        server_port=settings.server.port,
        share=settings.server.share,
        allowed_paths=[settings.storage.cache_dir],
    )


if __name__ == "__main__":
    main()
