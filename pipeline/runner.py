"""Pipeline runner — orchestrates the full classification workflow.

1. Discover timestamps
2. Extract frames + compute embeddings (cached)
3. Zero-shot classify all timestamps
4. (After human verification) Build prototypes → few-shot re-classify
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np

from app.config import settings
from data.models import UseCase
from pipeline.cache import Cache
from pipeline.classifier import (
    build_prototypes,
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
        """Extract frames and compute embeddings for all timestamps.

        Skips timestamps that already have cached embeddings.

        Returns
        -------
        dict with total, embedded, skipped, time_s
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
                # Extract frames
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

                # Compute embeddings
                embeddings = encode_images(
                    images,
                    model_name=settings.model.model_name,
                    device=settings.model.device,
                    batch_size=settings.model.batch_size,
                )

                # Cache
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
        """Run zero-shot classification on all embedded timestamps.

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

            # Classify
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

            # Parse metadata from first frame info
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

        # Sort by target class confidence (descending)
        all_results.sort(key=lambda r: r["target_confidence"], reverse=True)

        elapsed = round(time.time() - t0, 1)
        logger.info("Zero-shot complete: %d timestamps in %.1fs", len(all_results), elapsed)

        # Cache results
        self.cache.save_results(usecase.name, "zeroshot", all_results)

        return all_results

    # ── Step 3: Few-shot classify ────────────────────────────────

    def run_few_shot(
        self,
        usecase: UseCase,
        progress: Callable[[float, str], None] | None = None,
    ) -> list[dict]:
        """Run few-shot classification using verified examples as prototypes.

        Requires labels saved via cache.save_labels().

        Returns sorted list of timestamp results.
        """
        labels = self.cache.load_labels(usecase.name)
        if not labels:
            logger.warning("No verification labels found for '%s'", usecase.name)
            return []

        # Build prototype embeddings from labeled examples
        positive_embs = []
        negative_embs = []

        for source_id, label in labels.items():
            loaded = self.cache.load_embeddings(source_id)
            if loaded is None:
                continue
            embeddings, _ = loaded

            if label == "positive":
                # Use all frame embeddings from this timestamp
                for emb in embeddings:
                    positive_embs.append(emb)
            elif label == "negative":
                for emb in embeddings:
                    negative_embs.append(emb)

        if not positive_embs:
            logger.warning("No positive examples to build prototypes")
            return []
        if not negative_embs:
            logger.warning("No negative examples — using zero-shot fallback for negative class")
            # Use text embedding as fallback negative prototype
            from pipeline.classifier import encode_texts
            neg_text = [l for l in usecase.labels if l != usecase.target_label]
            if neg_text:
                negative_embs = list(encode_texts(
                    neg_text[:1],
                    settings.model.model_name,
                    settings.model.device,
                ))

        prototypes = build_prototypes({
            "positive": positive_embs,
            "negative": negative_embs,
        })

        if len(prototypes) < 2:
            logger.warning("Need at least 2 prototype classes")
            return []

        # Classify all timestamps
        sources = self.cache.list_embedded_sources()
        all_results = []
        t0 = time.time()

        for i, source_id in enumerate(sources):
            loaded = self.cache.load_embeddings(source_id)
            if loaded is None:
                continue

            embeddings, frame_infos = loaded

            frame_scores = few_shot_classify(
                embeddings,
                prototypes=prototypes,
                temperature=0.5,
            )

            avg_scores = {}
            for label in prototypes:
                avg_scores[label] = float(np.mean([
                    fs[label] for fs in frame_scores
                ]))

            predicted = max(avg_scores, key=avg_scores.get)
            confidence = avg_scores[predicted]
            target_confidence = avg_scores.get("positive", 0.0)

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
                "avg_scores": avg_scores,
                "predicted_label": predicted,
                "confidence": confidence,
                "target_confidence": target_confidence,
                "n_frames": len(frame_infos),
                "thumbnails": self.cache.list_thumbnails(source_id),
                "human_label": human_label,
            }
            all_results.append(result)

            if progress:
                progress((i + 1) / len(sources), f"Classified {source_id}")

        all_results.sort(key=lambda r: r["target_confidence"], reverse=True)

        elapsed = round(time.time() - t0, 1)
        logger.info("Few-shot complete: %d timestamps in %.1fs", len(all_results), elapsed)

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
