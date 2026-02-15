"""Disk cache for embeddings, classification results, and human labels.

Stores:
- Image embeddings per source_id (numpy .npz)
- Classification results per use-case (JSON)
- Human verification labels per use-case (JSON)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class Cache:
    """Manages all cached data for the classification pipeline."""

    def __init__(self, cache_dir: str) -> None:
        self.base = Path(cache_dir)
        self.embeddings_dir = self.base / "embeddings"
        self.results_dir = self.base / "results"
        self.labels_dir = self.base / "labels"
        self.thumbnails_dir = self.base / "thumbnails"

        for d in [self.embeddings_dir, self.results_dir,
                  self.labels_dir, self.thumbnails_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # ── Embeddings ───────────────────────────────────────────────

    def save_embeddings(
        self,
        source_id: str,
        embeddings: np.ndarray,
        frame_infos: list[dict],
    ) -> None:
        """Save embeddings + metadata for a timestamp."""
        path = self.embeddings_dir / f"{source_id}.npz"
        np.savez_compressed(
            path,
            embeddings=embeddings,
        )
        meta_path = self.embeddings_dir / f"{source_id}.json"
        meta_path.write_text(json.dumps(frame_infos, indent=2, default=str))

    def load_embeddings(self, source_id: str) -> tuple[np.ndarray, list[dict]] | None:
        """Load cached embeddings. Returns (embeddings, frame_infos) or None."""
        path = self.embeddings_dir / f"{source_id}.npz"
        meta_path = self.embeddings_dir / f"{source_id}.json"

        if not path.exists() or not meta_path.exists():
            return None

        try:
            data = np.load(path)
            embeddings = data["embeddings"]
            frame_infos = json.loads(meta_path.read_text())
            return embeddings, frame_infos
        except Exception as e:
            logger.warning("Failed to load embeddings for %s: %s", source_id, e)
            return None

    def has_embeddings(self, source_id: str) -> bool:
        return (self.embeddings_dir / f"{source_id}.npz").exists()

    def list_embedded_sources(self) -> list[str]:
        """List all source_ids with cached embeddings."""
        return [
            p.stem for p in sorted(self.embeddings_dir.glob("*.npz"))
        ]

    # ── Classification Results ───────────────────────────────────

    def _results_path(self, usecase_name: str, mode: str) -> Path:
        safe = usecase_name.lower().replace(" ", "_")
        return self.results_dir / f"{safe}_{mode}.json"

    def save_results(
        self,
        usecase_name: str,
        mode: str,  # "zeroshot" | "fewshot"
        results: list[dict],
    ) -> None:
        path = self._results_path(usecase_name, mode)
        path.write_text(json.dumps(results, indent=2, default=str))
        logger.info("Saved %d %s results for '%s'", len(results), mode, usecase_name)

    def load_results(
        self,
        usecase_name: str,
        mode: str,
    ) -> list[dict] | None:
        path = self._results_path(usecase_name, mode)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception as e:
            logger.warning("Failed to load results: %s", e)
            return None

    # ── Human Labels ─────────────────────────────────────────────

    def _labels_path(self, usecase_name: str) -> Path:
        safe = usecase_name.lower().replace(" ", "_")
        return self.labels_dir / f"{safe}_labels.json"

    def save_labels(
        self,
        usecase_name: str,
        labels: dict[str, str],  # source_id → "positive"|"negative"|"skip"
    ) -> None:
        path = self._labels_path(usecase_name)
        # Merge with existing
        existing = self.load_labels(usecase_name) or {}
        existing.update(labels)
        path.write_text(json.dumps(existing, indent=2))
        logger.info("Saved %d labels for '%s' (%d total)",
                    len(labels), usecase_name, len(existing))

    def load_labels(self, usecase_name: str) -> dict[str, str] | None:
        path = self._labels_path(usecase_name)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    def count_labels(self, usecase_name: str) -> dict[str, int]:
        """Count labels by type."""
        labels = self.load_labels(usecase_name)
        if not labels:
            return {"positive": 0, "negative": 0, "skip": 0, "total": 0}
        counts = {"positive": 0, "negative": 0, "skip": 0}
        for v in labels.values():
            counts[v] = counts.get(v, 0) + 1
        counts["total"] = len(labels)
        return counts

    # ── Thumbnails ───────────────────────────────────────────────

    def get_thumbnail_dir(self, source_id: str) -> Path:
        d = self.thumbnails_dir / source_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def list_thumbnails(self, source_id: str) -> list[str]:
        d = self.thumbnails_dir / source_id
        if not d.exists():
            return []
        return sorted(str(p) for p in d.glob("*.jpg"))

    # ── Stats ────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "embedded_sources": len(self.list_embedded_sources()),
            "result_files": len(list(self.results_dir.glob("*.json"))),
            "label_files": len(list(self.labels_dir.glob("*.json"))),
            "thumbnail_dirs": len(list(self.thumbnails_dir.iterdir()))
            if self.thumbnails_dir.exists() else 0,
        }
