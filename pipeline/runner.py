"""Pipeline runner — orchestrates the full classification workflow.

1. Discover timestamps
2. Extract frames + compute E5-V image embeddings (cached)
3. Zero-shot: E5-V text embeddings for labels, cosine similarity
4. Human verification: mark positive/negative examples
5. Few-shot: E5-V composed embedding (text + example images) → re-rank

Key insight: image embeddings from step 2 are reused in both steps 3 and 5
because E5-V puts text, image, and composed embeddings in the same space.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from app.config import settings
from data.models import UseCase
from pipeline.cache import Cache
from pipeline.classifier import (
    encode_composed,
    encode_images,
    few_shot_classify,
    zero_shot_classify,
)
from pipeline.frame_extractor import (
    discover_timestamps,
    extract_all_frames_for_timestamp,
    save_frame_thumbnail,
)

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Runs the classification pipeline for a given use case."""

    def __init__(self, cache: Cache | None = None) -> None:
        self.cache = cache or Cache(settings.storage.cache_dir)

    # ── Step 1: Embed all timestamps ─────────────────────────────

    def embed_all(
        self,
        video_dir: str | None = None,
        progress: Callable[[float, str], None] | None = None,
    ) -> dict:
        """Extract frames and compute E5-V image embeddings for all timestamps.

        Skips timestamps that already have cached embeddings.
        These embeddings are reused for both zero-shot and few-shot.
        """
        timestamps = discover_timestamps(video_dir)
        total = len(timestamps)
        if total == 0:
            return {"total": 0, "embedded": 0, "skipped": 0, "time_s": 0}

        embedded = 0
        skipped = 0
        t0 = time.time()

        for i, (source_id, camera_paths) in enumerate(timestamps.items()):
            if self.cache.has_embeddings(source_id):
                skipped += 1
                if progress:
                    progress((i + 1) / total, f"Skip {source_id} (cached)")
                continue

            try:
                frames_data = extract_all_frames_for_timestamp(
                    camera_paths,
                    n_frames=settings.pipeline.frames_per_video,
                )

                if not frames_data:
                    skipped += 1
                    continue

                images = [img for img, _ in frames_data]
                infos = [
                    {
                        "video_path": fi.video_path,
                        "source_id": fi.source_id,
                        "camera": fi.camera,
                        "frame_idx": fi.frame_idx,
                        "timestamp_s": fi.timestamp_s,
                    }
                    for _, fi in frames_data
                ]

                # Save thumbnails
                for img, fi in frames_data:
                    save_frame_thumbnail(
                        img, str(self.cache.thumbnails_dir.parent),
                        fi.source_id, fi.camera, fi.frame_idx,
                    )

                # Compute E5-V image embeddings
                embeddings = encode_images(
                    images,
                    model_name=settings.model.model_name,
                    device=settings.model.device,
                    batch_size=settings.model.batch_size,
                )

                self.cache.save_embeddings(source_id, embeddings, infos)
                embedded += 1

                if progress:
                    progress(
                        (i + 1) / total,
                        f"Embedded {source_id} ({len(images)} frames)",
                    )

            except Exception as e:
                logger.error("Failed to embed %s: %s", source_id, e)
                if progress:
                    progress((i + 1) / total, f"FAILED {source_id}: {e}")

        elapsed = round(time.time() - t0, 1)
        logger.info("Embedding complete: %d embedded, %d skipped in %.1fs",
                    embedded, skipped, elapsed)

        return {
            "total": total,
            "embedded": embedded,
            "skipped": skipped,
            "time_s": elapsed,
        }

    # ── Step 2: Zero-shot classify ───────────────────────────────

    def run_zero_shot(
        self,
        usecase: UseCase,
        progress: Callable[[float, str], None] | None = None,
    ) -> list[dict]:
        """Zero-shot: E5-V text embeddings vs cached image embeddings.

        Returns sorted list of timestamp results (highest target confidence first).
        """
        sources = self.cache.list_embedded_sources()
        total = len(sources)
        if total == 0:
            logger.warning("No embedded sources found. Run embed_all first.")
            return []

        all_results = []
        t0 = time.time()

        for i, source_id in enumerate(sources):
            loaded = self.cache.load_embeddings(source_id)
            if loaded is None:
                continue

            embeddings, frame_infos = loaded

            frame_scores = zero_shot_classify(
                embeddings,
                labels=usecase.labels,
                model_name=settings.model.model_name,
                device=settings.model.device,
            )

            # Aggregate: mean across all frames
            avg_scores = {}
            for label in usecase.labels:
                avg_scores[label] = float(np.mean([
                    fs[label] for fs in frame_scores
                ]))

            predicted = max(avg_scores, key=avg_scores.get)
            confidence = avg_scores[predicted]
            target_confidence = avg_scores[usecase.target_label]

            fi0 = frame_infos[0] if frame_infos else {}
            from data.video_parser import parse_video_filename
            parsed = parse_video_filename(
                Path(fi0.get("video_path", "")).name
            ) if fi0.get("video_path") else None

            result = {
                "source_id": source_id,
                "vehicle_id": parsed["vehicle_id"] if parsed else "",
                "date": parsed["date"] if parsed else "",
                "datetime_utc": parsed["datetime_utc"] if parsed else "",
                "avg_scores": avg_scores,
                "predicted_label": predicted,
                "confidence": confidence,
                "target_confidence": target_confidence,
                "n_frames": len(frame_infos),
                "thumbnails": self.cache.list_thumbnails(source_id),
            }
            all_results.append(result)

            if progress:
                progress((i + 1) / total, f"Classified {source_id}")

        all_results.sort(key=lambda r: r["target_confidence"], reverse=True)

        elapsed = round(time.time() - t0, 1)
        logger.info("Zero-shot complete: %d timestamps in %.1fs",
                    len(all_results), elapsed)

        self.cache.save_results(usecase.name, "zeroshot", all_results)
        return all_results

    # ── Step 3: Few-shot classify ────────────────────────────────

    def _select_example_thumbnail(
        self,
        thumbnails: list[str],
        prefer_camera: str = "front_camera",
    ) -> str | None:
        """Pick the best single thumbnail from a timestamp's thumbnails."""
        if not thumbnails:
            return None
        # Prefer front camera middle frame
        front = [t for t in thumbnails if prefer_camera in t]
        if front:
            return front[len(front) // 2]
        return thumbnails[len(thumbnails) // 2]

    def run_few_shot(
        self,
        usecase: UseCase,
        progress: Callable[[float, str], None] | None = None,
    ) -> list[dict]:
        """Few-shot: build composed embedding from text + verified examples.

        Workflow:
        1. Collect thumbnail images from verified positive/negative timestamps
        2. Create composed E5-V embedding: target_description + positive images → query vector
        3. (Optional) Create composed negative embedding: "not target" + negative images
        4. Compare query vector against ALL cached image embeddings via cosine similarity
        5. Re-rank all timestamps

        The cached image embeddings from step 1 are reused — no re-embedding needed!
        """
        labels = self.cache.load_labels(usecase.name)
        if not labels:
            logger.warning("No verification labels found for '%s'", usecase.name)
            return []

        # Gather example images
        positive_images: list[Image.Image] = []
        negative_images: list[Image.Image] = []

        max_pos = settings.fewshot.max_positive_examples
        max_neg = settings.fewshot.max_negative_examples

        pos_count = 0
        neg_count = 0

        for source_id, label in labels.items():
            if label == "positive" and pos_count < max_pos:
                thumb = self._select_example_thumbnail(
                    self.cache.list_thumbnails(source_id)
                )
                if thumb:
                    try:
                        positive_images.append(Image.open(thumb).convert("RGB"))
                        pos_count += 1
                    except Exception as e:
                        logger.warning("Cannot load thumbnail %s: %s", thumb, e)

            elif label == "negative" and neg_count < max_neg:
                thumb = self._select_example_thumbnail(
                    self.cache.list_thumbnails(source_id)
                )
                if thumb:
                    try:
                        negative_images.append(Image.open(thumb).convert("RGB"))
                        neg_count += 1
                    except Exception as e:
                        logger.warning("Cannot load thumbnail %s: %s", thumb, e)

        if not positive_images:
            logger.warning("No positive example images found")
            return []

        logger.info(
            "Building composed embeddings: %d positive, %d negative images",
            len(positive_images), len(negative_images),
        )

        if progress:
            progress(0.05, "Building positive query embedding...")

        # Build composed positive query: text + positive example images
        positive_query = encode_composed(
            text=usecase.target_label,
            images=positive_images,
            model_name=settings.model.model_name,
            device=settings.model.device,
        )

        # Build composed negative query if we have negative examples
        negative_query = None
        if negative_images:
            if progress:
                progress(0.1, "Building negative query embedding...")

            # Use first non-target label as negative text
            neg_text = [l for l in usecase.labels if l != usecase.target_label]
            negative_query = encode_composed(
                text=neg_text[0] if neg_text else "not " + usecase.target_label,
                images=negative_images,
                model_name=settings.model.model_name,
                device=settings.model.device,
            )

        # Classify all timestamps against composed query
        sources = self.cache.list_embedded_sources()
        all_results = []
        t0 = time.time()

        for i, source_id in enumerate(sources):
            loaded = self.cache.load_embeddings(source_id)
            if loaded is None:
                continue

            embeddings, frame_infos = loaded

            # Few-shot classify using composed embeddings
            frame_scores = few_shot_classify(
                image_embeddings=embeddings,
                query_embedding=positive_query,
                negative_embedding=negative_query,
                temperature=0.5,
            )

            # Aggregate across frames
            avg_pos = float(np.mean([fs["positive"] for fs in frame_scores]))
            avg_neg = float(np.mean([fs["negative"] for fs in frame_scores]))

            fi0 = frame_infos[0] if frame_infos else {}
            from data.video_parser import parse_video_filename
            parsed = parse_video_filename(
                Path(fi0.get("video_path", "")).name
            ) if fi0.get("video_path") else None

            human_label = labels.get(source_id)

            result = {
                "source_id": source_id,
                "vehicle_id": parsed["vehicle_id"] if parsed else "",
                "date": parsed["date"] if parsed else "",
                "datetime_utc": parsed["datetime_utc"] if parsed else "",
                "avg_scores": {"positive": avg_pos, "negative": avg_neg},
                "predicted_label": "positive" if avg_pos > avg_neg else "negative",
                "confidence": max(avg_pos, avg_neg),
                "target_confidence": avg_pos,
                "n_frames": len(frame_infos),
                "thumbnails": self.cache.list_thumbnails(source_id),
                "human_label": human_label,
            }
            all_results.append(result)

            if progress:
                progress(
                    0.15 + 0.85 * (i + 1) / len(sources),
                    f"Classified {source_id}",
                )

        all_results.sort(key=lambda r: r["target_confidence"], reverse=True)

        elapsed = round(time.time() - t0, 1)
        logger.info(
            "Few-shot complete: %d timestamps in %.1fs "
            "(using %d pos + %d neg examples)",
            len(all_results), elapsed,
            len(positive_images), len(negative_images),
        )

        self.cache.save_results(usecase.name, "fewshot", all_results)
        return all_results

    # ── Utilities ────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Return current pipeline status."""
        timestamps = discover_timestamps()
        cache_stats = self.cache.stats()
        return {
            "total_timestamps": len(timestamps),
            "total_videos": sum(len(v) for v in timestamps.values()),
            **cache_stats,
        }
