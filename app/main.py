"""AV Video Classifier — Entry Point.

Usage:
    python app/main.py
    VIDEO_DIR=/path/to/videos python app/main.py
"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    from app.config import settings
    from ui.gradio_app import create_ui

    logger.info("Starting AV Video Classifier")
    logger.info("  Video dir:  %s", settings.storage.video_dir)
    logger.info("  Cache dir:  %s", settings.storage.cache_dir)
    logger.info("  Model:      %s", settings.model.model_name)
    logger.info("  Device:     %s", settings.model.device)
    logger.info("  Frames/vid: %d", settings.pipeline.frames_per_video)

    app = create_ui()
    app.launch(
        server_name=settings.server.host,
        server_port=settings.server.port,
        share=settings.server.share,
    )


if __name__ == "__main__":
    main()
