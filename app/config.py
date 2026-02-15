"""Application configuration.

All settings centralised and env-configurable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    return os.environ.get(key, str(default)).lower() in ("true", "1", "yes")


@dataclass
class StorageConfig:
    video_dir: str = field(default_factory=lambda: _env("VIDEO_DIR", "/workspace/video-inference/videos/cropped"))
    cache_dir: str = field(default_factory=lambda: _env("CACHE_DIR", "/workspace/cache"))
    crop_config: str = field(default_factory=lambda: _env("CROP_CONFIG", "config/crop_config.json"))

    @property
    def embeddings_dir(self) -> str:
        return os.path.join(self.cache_dir, "embeddings")

    @property
    def results_dir(self) -> str:
        return os.path.join(self.cache_dir, "results")

    @property
    def labels_dir(self) -> str:
        return os.path.join(self.cache_dir, "labels")


@dataclass
class ModelConfig:
    """LLaVA-Next multimodal model for unified embeddings.

    Uses last-token hidden state as embedding vector for text, image,
    and composed (text+image) inputs — all in the same vector space.
    """
    model_name: str = field(default_factory=lambda: _env("MODEL", "llava-hf/llava-v1.6-mistral-7b-hf"))
    device: str = field(default_factory=lambda: _env("DEVICE", "cuda"))
    batch_size: int = field(default_factory=lambda: _env_int("BATCH_SIZE", 8))


@dataclass
class FewShotConfig:
    """Few-shot composed embedding settings."""
    # How many verified positive example images to include in the composed query
    max_positive_examples: int = field(default_factory=lambda: _env_int("MAX_POS_EXAMPLES", 4))
    # How many verified negative example images for the negative query
    max_negative_examples: int = field(default_factory=lambda: _env_int("MAX_NEG_EXAMPLES", 4))


@dataclass
class PipelineConfig:
    frames_per_video: int = field(default_factory=lambda: _env_int("FRAMES_PER_VIDEO", 100))
    cameras: list[str] = field(default_factory=lambda: [
        "front_camera", "left_fisheye", "right_fisheye", "rear_camera",
    ])
    verification_top_k: int = field(default_factory=lambda: _env_int("VERIFY_TOP_K", 50))


@dataclass
class ServerConfig:
    host: str = field(default_factory=lambda: _env("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("PORT", 7860))
    share: bool = field(default_factory=lambda: _env_bool("GRADIO_SHARE", True))


@dataclass
class Settings:
    storage: StorageConfig = field(default_factory=StorageConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    fewshot: FewShotConfig = field(default_factory=FewShotConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    server: ServerConfig = field(default_factory=ServerConfig)


settings = Settings()
