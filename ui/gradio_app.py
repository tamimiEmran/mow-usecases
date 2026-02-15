"""Gradio UI — tabbed interface for video classification.

3 predefined use-case tabs + 1 custom tab, each with the full pipeline:
1. Zero-shot classification
2. Verify top candidates (with video viewer)
3. Few-shot re-ranking

Video viewer integrated into verification and available standalone per tab.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

import gradio as gr

from app.config import settings
from data.models import ALL_USECASES, UseCase
from pipeline.cache import Cache
from pipeline.runner import PipelineRunner
from pipeline.frame_extractor import discover_timestamps

logger = logging.getLogger(__name__)

# ── Singleton pipeline runner ────────────────────────────────────

_runner: PipelineRunner | None = None
_cache: Cache | None = None
_timestamps: dict[str, dict[str, str]] | None = None


def _get_runner() -> PipelineRunner:
    global _runner, _cache
    if _runner is None:
        _cache = Cache(settings.storage.cache_dir)
        _runner = PipelineRunner(_cache)
    return _runner


def _get_cache() -> Cache:
    global _cache
    if _cache is None:
        _cache = Cache(settings.storage.cache_dir)
    return _cache


def _get_timestamps() -> dict[str, dict[str, str]]:
    """Cached lookup: source_id → {camera: video_path}."""
    global _timestamps
    if _timestamps is None:
        _timestamps = discover_timestamps()
    return _timestamps


def _refresh_timestamps():
    global _timestamps
    _timestamps = discover_timestamps()
    return _timestamps


def _find_videos_for_source(source_id: str) -> dict[str, str]:
    """Return {camera_name: video_path} for a source_id."""
    ts = _get_timestamps()
    return ts.get(source_id, {})


# ═══════════════════════════════════════════════════════════════════
# Shared: Embedding step
# ═══════════════════════════════════════════════════════════════════

def _run_embedding(progress=gr.Progress()):
    """Embed all timestamps (shared across use cases)."""
    _refresh_timestamps()
    runner = _get_runner()

    def prog(frac, msg):
        progress(frac, desc=msg)

    result = runner.embed_all(progress=prog)

    summary = (
        f"### Embedding Complete\n\n"
        f"| Metric | Count |\n|---|---|\n"
        f"| Total timestamps | {result['total']} |\n"
        f"| Newly embedded | {result['embedded']} |\n"
        f"| Skipped (cached) | {result['skipped']} |\n"
        f"| Time | {result['time_s']}s |"
    )
    return summary, _status_text()


# ═══════════════════════════════════════════════════════════════════
# Video Viewer
# ═══════════════════════════════════════════════════════════════════

def _load_video_for_source(source_id: str, camera: str) -> str | None:
    """Get video path for a given source + camera."""
    if not source_id:
        return None
    videos = _find_videos_for_source(source_id.strip())
    return videos.get(camera)


def _get_available_cameras(source_id: str) -> list[str]:
    """List cameras with videos for this source_id."""
    if not source_id:
        return []
    videos = _find_videos_for_source(source_id.strip())
    return list(videos.keys())


def _build_video_viewer_handler():
    """Create handler that loads a video given source_id + camera selection."""

    def load_video(source_id: str, camera: str):
        if not source_id or not camera:
            return None, "Select a source and camera."
        path = _load_video_for_source(source_id, camera)
        if path and Path(path).exists():
            return path, f"Playing `{camera}` for `{source_id}`"
        return None, f"No video found for `{source_id}` / `{camera}`"

    def update_camera_choices(source_id: str):
        cams = _get_available_cameras(source_id)
        if not cams:
            return gr.update(choices=[], value=None)
        default = "front_camera" if "front_camera" in cams else cams[0]
        return gr.update(choices=cams, value=default)

    return load_video, update_camera_choices


# ═══════════════════════════════════════════════════════════════════
# Per use-case handlers
# ═══════════════════════════════════════════════════════════════════

def _make_zero_shot_handler(usecase: UseCase):
    def handler(progress=gr.Progress()):
        runner = _get_runner()

        def prog(frac, msg):
            progress(frac, desc=msg)

        results = runner.run_zero_shot(usecase, progress=prog)

        if not results:
            return "No results — run embedding first.", [], _status_text()

        table_data = []
        for r in results:
            table_data.append([
                r["source_id"],
                r["vehicle_id"],
                r["date"],
                f"{r['target_confidence']:.3f}",
                r["predicted_label"][:50],
                r["n_frames"],
            ])

        top = results[0]
        summary = (
            f"### Zero-Shot: {usecase.name}\n\n"
            f"Classified **{len(results)}** timestamps against target: "
            f"`{usecase.target_label[:60]}`\n\n"
            f"Top match: **{top['source_id']}** "
            f"(confidence: {top['target_confidence']:.3f})"
        )

        return summary, table_data, _status_text()

    return handler


def _make_load_verification_handler(usecase: UseCase):
    def handler(verify_index: int):
        cache = _get_cache()
        results = cache.load_results(usecase.name, "zeroshot")

        if not results:
            return (
                "Run zero-shot first.",
                None,
                0,
                "",
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
            )

        idx = int(verify_index) if verify_index else 0
        idx = max(0, min(idx, len(results) - 1))

        r = results[idx]
        thumbs = r.get("thumbnails", [])

        labels = cache.load_labels(usecase.name) or {}
        existing = labels.get(r["source_id"], "")

        label_badge = ""
        if existing:
            icon = {"positive": "✅", "negative": "❌", "skip": "⏭️"}.get(existing, "🏷️")
            label_badge = f"\n\n{icon} Already labeled: **{existing}**"

        info = (
            f"### Candidate {idx + 1} / {len(results)}\n\n"
            f"**Source:** `{r['source_id']}`\n\n"
            f"Vehicle: `{r['vehicle_id']}` · Date: `{r['date']}`\n\n"
            f"Target confidence: **{r['target_confidence']:.3f}** · "
            f"Predicted: `{r['predicted_label'][:50]}`"
            f"{label_badge}"
        )

        return (
            info,
            thumbs if thumbs else None,
            idx,
            r["source_id"],
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )

    return handler


def _make_label_handler(usecase: UseCase, label_type: str):
    def handler(verify_index: int, current_source_id: str):
        cache = _get_cache()
        results = cache.load_results(usecase.name, "zeroshot")

        if not results or not current_source_id:
            return verify_index, "No results loaded."

        cache.save_labels(usecase.name, {current_source_id: label_type})
        next_idx = min(int(verify_index) + 1, len(results) - 1)

        counts = cache.count_labels(usecase.name)
        icon = {"positive": "✅", "negative": "❌", "skip": "⏭️"}[label_type]
        msg = (
            f"{icon} Labeled `{current_source_id}` as **{label_type}**\n\n"
            f"Progress: {counts['positive']} positive · "
            f"{counts['negative']} negative · {counts['skip']} skipped"
        )

        return next_idx, msg

    return handler


def _make_few_shot_handler(usecase: UseCase):
    def handler(progress=gr.Progress()):
        runner = _get_runner()
        cache = _get_cache()

        counts = cache.count_labels(usecase.name)
        if counts["positive"] == 0:
            return (
                "Need at least **1 positive** example. Verify some candidates first.",
                [],
                _status_text(),
            )

        def prog(frac, msg):
            progress(frac, desc=msg)

        results = runner.run_few_shot(usecase, progress=prog)

        if not results:
            return "Few-shot failed — check logs.", [], _status_text()

        table_data = []
        for r in results:
            human = r.get("human_label", "")
            marker = {"positive": "✅", "negative": "❌", "skip": "⏭️"}.get(human, "")
            table_data.append([
                r["source_id"],
                r["vehicle_id"],
                r["date"],
                f"{r['target_confidence']:.3f}",
                r["predicted_label"],
                marker,
            ])

        neg_info = ""
        if counts.get("negative", 0) > 0:
            neg_info = f" + {counts['negative']} negative images"

        top = results[0]
        summary = (
            f"### Few-Shot: {usecase.name}\n\n"
            f"Composed query: `{usecase.target_label[:60]}` "
            f"+ {counts['positive']} positive images{neg_info}\n\n"
            f"Re-ranked **{len(results)}** timestamps\n\n"
            f"Top match: **{top['source_id']}** "
            f"(confidence: {top['target_confidence']:.3f})\n\n"
            f"*Same cached image embeddings — only the query changed.*"
        )

        return summary, table_data, _status_text()

    return handler


# ═══════════════════════════════════════════════════════════════════
# Status
# ═══════════════════════════════════════════════════════════════════

def _status_text() -> str:
    try:
        runner = _get_runner()
        status = runner.get_status()
        cache = _get_cache()

        lines = [
            "### Status\n",
            f"**{status['total_timestamps']}** timestamps · "
            f"**{status['total_videos']}** videos\n",
            f"**{status['embedded_sources']}** embedded\n",
            f"Model: `{settings.model.model_name.split('/')[-1]}`\n",
        ]

        for uc in ALL_USECASES:
            counts = cache.count_labels(uc.name)
            if counts["total"] > 0:
                lines.append(
                    f"{uc.name}: {counts['positive']}✅ "
                    f"{counts['negative']}❌ {counts['skip']}⏭️\n"
                )

        # Check custom labels too
        custom_counts = cache.count_labels("Custom")
        if custom_counts and custom_counts["total"] > 0:
            lines.append(
                f"Custom: {custom_counts['positive']}✅ "
                f"{custom_counts['negative']}❌ {custom_counts['skip']}⏭️\n"
            )

        return "\n".join(lines)
    except Exception as e:
        return f"Status unavailable: {e}"


# ═══════════════════════════════════════════════════════════════════
# Video viewer widget (reusable)
# ═══════════════════════════════════════════════════════════════════

def _build_video_viewer(prefix: str = ""):
    """Build a video viewer sub-component. Returns (source_input, camera_dropdown, video, info)."""

    load_video_fn, update_cameras_fn = _build_video_viewer_handler()

    with gr.Accordion("📹 Video Viewer", open=False):
        gr.Markdown("View the full source video for any timestamp.")
        with gr.Row():
            vid_source = gr.Textbox(
                label="Source ID",
                placeholder="e.g. m002-20260202-1770014818",
                scale=3,
            )
            vid_camera = gr.Dropdown(
                label="Camera",
                choices=["front_camera", "left_fisheye", "right_fisheye", "rear_camera"],
                value="front_camera",
                scale=2,
            )
            vid_load_btn = gr.Button("▶ Load", scale=1, size="sm")

        vid_player = gr.Video(label="Video", height=400)
        vid_info = gr.Markdown("")

        # When source changes, update available cameras
        vid_source.change(
            fn=update_cameras_fn,
            inputs=[vid_source],
            outputs=[vid_camera],
        )

        vid_load_btn.click(
            fn=load_video_fn,
            inputs=[vid_source, vid_camera],
            outputs=[vid_player, vid_info],
        )

    return vid_source, vid_camera, vid_player


# ═══════════════════════════════════════════════════════════════════
# Build a use-case tab (predefined or custom)
# ═══════════════════════════════════════════════════════════════════

def _build_usecase_tab(usecase: UseCase):
    """Full pipeline tab: zero-shot → verify → few-shot + video viewer."""

    with gr.Tab(usecase.name):
        gr.Markdown(f"### {usecase.name}\n{usecase.description}")

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown(
                    f"**Target:** `{usecase.target_label[:80]}`"
                )
            with gr.Column(scale=1):
                label_pills = " · ".join(f"`{l[:40]}`" for l in usecase.labels)
                gr.Markdown(f"**Labels:** {label_pills}")

        # ── Step 1: Zero-Shot ────────────────────────────────
        with gr.Accordion("① Zero-Shot Classification", open=True):
            gr.Markdown(
                "Encodes text labels and frame images into the same vector space, "
                "then ranks all timestamps by cosine similarity to the target."
            )
            zs_btn = gr.Button("▶ Run Zero-Shot", variant="primary")
            zs_summary = gr.Markdown("")
            zs_table = gr.Dataframe(
                headers=["Source ID", "Vehicle", "Date",
                         "Target Conf", "Predicted", "Frames"],
                datatype=["str", "str", "str", "str", "str", "number"],
                interactive=False,
                wrap=True,
            )

        # ── Step 2: Verify ───────────────────────────────────
        with gr.Accordion("② Verify Top Candidates", open=False):
            gr.Markdown(
                "Review zero-shot results. Mark **positive** (genuinely shows the target) "
                "or **negative** (false positive). These examples power few-shot re-ranking."
            )
            with gr.Row():
                verify_idx = gr.Number(
                    value=0, label="Candidate #", precision=0, minimum=0,
                )
                load_btn = gr.Button("Load Candidate", size="sm")

            verify_info = gr.Markdown("Click **Load Candidate** to start reviewing.")
            verify_gallery = gr.Gallery(
                label="Frame Thumbnails",
                columns=4,
                height=240,
                object_fit="contain",
            )

            current_source = gr.Textbox(visible=False)

            with gr.Row():
                confirm_btn = gr.Button("✅ Positive", variant="primary",
                                       interactive=False)
                reject_btn = gr.Button("❌ Negative", variant="stop",
                                      interactive=False)
                skip_btn = gr.Button("⏭️ Skip", interactive=False)

            label_status = gr.Markdown("")

            # Inline video viewer for verification — pre-fills source_id
            gr.Markdown("---")
            gr.Markdown("**Preview full video for this candidate:**")

            load_video_fn, update_cameras_fn = _build_video_viewer_handler()

            with gr.Row():
                verify_vid_camera = gr.Dropdown(
                    label="Camera",
                    choices=["front_camera", "left_fisheye", "right_fisheye", "rear_camera"],
                    value="front_camera",
                    scale=2,
                )
                verify_vid_btn = gr.Button("▶ Play Video", scale=1, size="sm")

            verify_vid_player = gr.Video(label="Candidate Video", height=360)
            verify_vid_info = gr.Markdown("")

            verify_vid_btn.click(
                fn=load_video_fn,
                inputs=[current_source, verify_vid_camera],
                outputs=[verify_vid_player, verify_vid_info],
            )

        # ── Step 3: Few-Shot ─────────────────────────────────
        with gr.Accordion("③ Few-Shot Re-ranking", open=False):
            gr.Markdown(
                "Creates a **composed embedding** from the target text + your verified "
                "example images → one query vector. All timestamps are re-ranked "
                "against this composed query using cached image embeddings."
            )
            fs_btn = gr.Button("▶ Run Few-Shot", variant="primary")
            fs_summary = gr.Markdown("")
            fs_table = gr.Dataframe(
                headers=["Source ID", "Vehicle", "Date",
                         "Target Conf", "Predicted", "Label"],
                datatype=["str", "str", "str", "str", "str", "str"],
                interactive=False,
                wrap=True,
            )

        # ── Video Viewer (standalone) ────────────────────────
        vid_source, _, _ = _build_video_viewer()

        # ── Wire events ──────────────────────────────────────
        status_out = gr.Markdown()

        zs_handler = _make_zero_shot_handler(usecase)
        zs_btn.click(
            fn=zs_handler,
            outputs=[zs_summary, zs_table, status_out],
        )

        load_handler = _make_load_verification_handler(usecase)
        load_btn.click(
            fn=load_handler,
            inputs=[verify_idx],
            outputs=[verify_info, verify_gallery, verify_idx,
                     current_source, confirm_btn, reject_btn, skip_btn],
        )

        # Auto-update camera choices when candidate loads
        current_source.change(
            fn=update_cameras_fn,
            inputs=[current_source],
            outputs=[verify_vid_camera],
        )

        for btn, ltype in [(confirm_btn, "positive"),
                           (reject_btn, "negative"),
                           (skip_btn, "skip")]:
            label_fn = _make_label_handler(usecase, ltype)
            btn.click(
                fn=label_fn,
                inputs=[verify_idx, current_source],
                outputs=[verify_idx, label_status],
            ).then(
                fn=load_handler,
                inputs=[verify_idx],
                outputs=[verify_info, verify_gallery, verify_idx,
                         current_source, confirm_btn, reject_btn, skip_btn],
            )

        fs_handler = _make_few_shot_handler(usecase)
        fs_btn.click(
            fn=fs_handler,
            outputs=[fs_summary, fs_table, status_out],
        )


def _build_custom_tab():
    """Custom query tab with full pipeline: zero-shot → verify → few-shot."""

    with gr.Tab("Custom Query"):
        gr.Markdown(
            "### Custom Classification\n\n"
            "Define your own labels and run the full pipeline: "
            "zero-shot → verify → few-shot."
        )

        with gr.Row():
            with gr.Column(scale=2):
                custom_labels = gr.Textbox(
                    label="Classification labels (one per line)",
                    lines=4,
                    placeholder=(
                        "a vehicle making an illegal U-turn\n"
                        "a vehicle turning legally\n"
                        "a road scene with no turning vehicles"
                    ),
                )
            with gr.Column(scale=1):
                custom_target = gr.Textbox(
                    label="Target label (positive class)",
                    placeholder="a vehicle making an illegal U-turn",
                )
                custom_name = gr.Textbox(
                    label="Query name (for caching)",
                    value="Custom",
                    placeholder="Custom",
                )

        # ── Step 1: Zero-Shot ────────────────────────────────
        with gr.Accordion("① Zero-Shot Classification", open=True):
            custom_zs_btn = gr.Button("▶ Run Zero-Shot", variant="primary")
            custom_zs_summary = gr.Markdown("*Enter labels above, then run.*")
            custom_zs_table = gr.Dataframe(
                headers=["Source ID", "Vehicle", "Date",
                         "Target Conf", "Predicted", "Frames"],
                datatype=["str", "str", "str", "str", "str", "number"],
                interactive=False,
                wrap=True,
            )

        # ── Step 2: Verify ───────────────────────────────────
        with gr.Accordion("② Verify Top Candidates", open=False):
            gr.Markdown(
                "Review zero-shot results. Mark positive or negative for few-shot."
            )
            with gr.Row():
                custom_verify_idx = gr.Number(
                    value=0, label="Candidate #", precision=0, minimum=0,
                )
                custom_load_btn = gr.Button("Load Candidate", size="sm")

            custom_verify_info = gr.Markdown("Run zero-shot first, then load candidates.")
            custom_verify_gallery = gr.Gallery(
                label="Frame Thumbnails",
                columns=4,
                height=240,
                object_fit="contain",
            )

            custom_current_source = gr.Textbox(visible=False)

            with gr.Row():
                custom_confirm_btn = gr.Button("✅ Positive", variant="primary",
                                               interactive=False)
                custom_reject_btn = gr.Button("❌ Negative", variant="stop",
                                              interactive=False)
                custom_skip_btn = gr.Button("⏭️ Skip", interactive=False)

            custom_label_status = gr.Markdown("")

            # Inline video viewer
            gr.Markdown("---")
            gr.Markdown("**Preview full video for this candidate:**")

            load_video_fn, update_cameras_fn = _build_video_viewer_handler()

            with gr.Row():
                custom_vid_camera = gr.Dropdown(
                    label="Camera",
                    choices=["front_camera", "left_fisheye", "right_fisheye", "rear_camera"],
                    value="front_camera",
                    scale=2,
                )
                custom_vid_btn = gr.Button("▶ Play Video", scale=1, size="sm")

            custom_vid_player = gr.Video(label="Candidate Video", height=360)
            custom_vid_info = gr.Markdown("")

            custom_vid_btn.click(
                fn=load_video_fn,
                inputs=[custom_current_source, custom_vid_camera],
                outputs=[custom_vid_player, custom_vid_info],
            )

            custom_current_source.change(
                fn=update_cameras_fn,
                inputs=[custom_current_source],
                outputs=[custom_vid_camera],
            )

        # ── Step 3: Few-Shot ─────────────────────────────────
        with gr.Accordion("③ Few-Shot Re-ranking", open=False):
            gr.Markdown(
                "Composed embedding from target text + verified examples → re-rank."
            )
            custom_fs_btn = gr.Button("▶ Run Few-Shot", variant="primary")
            custom_fs_summary = gr.Markdown("")
            custom_fs_table = gr.Dataframe(
                headers=["Source ID", "Vehicle", "Date",
                         "Target Conf", "Predicted", "Label"],
                datatype=["str", "str", "str", "str", "str", "str"],
                interactive=False,
                wrap=True,
            )

        # ── Video Viewer (standalone) ────────────────────────
        _build_video_viewer()

        # ── Wire events ──────────────────────────────────────
        custom_status_out = gr.Markdown()

        def _run_custom_zs(labels_text, target, name, progress=gr.Progress()):
            if not labels_text.strip() or not target.strip():
                return "Enter labels and a target label.", [], ""

            labels_list = [l.strip() for l in labels_text.strip().split("\n") if l.strip()]
            if len(labels_list) < 2:
                return "Need at least 2 labels.", [], ""
            if target not in labels_list:
                labels_list.insert(0, target)

            uc_name = name.strip() or "Custom"
            custom_uc = UseCase(
                name=uc_name,
                description="User-defined query",
                labels=labels_list,
                target_label=target,
            )

            runner = _get_runner()
            results = runner.run_zero_shot(
                custom_uc,
                progress=lambda f, m: progress(f, desc=m),
            )

            if not results:
                return "No results — run embedding first.", [], _status_text()

            table = [
                [r["source_id"], r["vehicle_id"], r["date"],
                 f"{r['target_confidence']:.3f}",
                 r["predicted_label"][:50], r["n_frames"]]
                for r in results
            ]

            top = results[0]
            summary = (
                f"### Zero-Shot: {uc_name}\n\n"
                f"Classified **{len(results)}** timestamps\n\n"
                f"Top match: **{top['source_id']}** "
                f"(confidence: {top['target_confidence']:.3f})"
            )

            return summary, table, _status_text()

        custom_zs_btn.click(
            fn=_run_custom_zs,
            inputs=[custom_labels, custom_target, custom_name],
            outputs=[custom_zs_summary, custom_zs_table, custom_status_out],
        )

        # Verification handlers for custom
        def _custom_load_verify(verify_index, name):
            cache = _get_cache()
            uc_name = name.strip() or "Custom"
            results = cache.load_results(uc_name, "zeroshot")

            if not results:
                return (
                    "Run zero-shot first.",
                    None, 0, "",
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                )

            idx = int(verify_index) if verify_index else 0
            idx = max(0, min(idx, len(results) - 1))
            r = results[idx]
            thumbs = r.get("thumbnails", [])

            labels = cache.load_labels(uc_name) or {}
            existing = labels.get(r["source_id"], "")

            label_badge = ""
            if existing:
                icon = {"positive": "✅", "negative": "❌", "skip": "⏭️"}.get(existing, "🏷️")
                label_badge = f"\n\n{icon} Already labeled: **{existing}**"

            info = (
                f"### Candidate {idx + 1} / {len(results)}\n\n"
                f"**Source:** `{r['source_id']}`\n\n"
                f"Vehicle: `{r['vehicle_id']}` · Date: `{r['date']}`\n\n"
                f"Target confidence: **{r['target_confidence']:.3f}** · "
                f"Predicted: `{r['predicted_label'][:50]}`"
                f"{label_badge}"
            )

            return (
                info,
                thumbs if thumbs else None,
                idx,
                r["source_id"],
                gr.update(interactive=True),
                gr.update(interactive=True),
                gr.update(interactive=True),
            )

        custom_load_btn.click(
            fn=_custom_load_verify,
            inputs=[custom_verify_idx, custom_name],
            outputs=[custom_verify_info, custom_verify_gallery, custom_verify_idx,
                     custom_current_source, custom_confirm_btn, custom_reject_btn,
                     custom_skip_btn],
        )

        def _make_custom_label_handler(label_type: str):
            def handler(verify_index, current_source_id, name):
                cache = _get_cache()
                uc_name = name.strip() or "Custom"
                results = cache.load_results(uc_name, "zeroshot")

                if not results or not current_source_id:
                    return verify_index, "No results loaded."

                cache.save_labels(uc_name, {current_source_id: label_type})
                next_idx = min(int(verify_index) + 1, len(results) - 1)

                counts = cache.count_labels(uc_name)
                icon = {"positive": "✅", "negative": "❌", "skip": "⏭️"}[label_type]
                msg = (
                    f"{icon} Labeled `{current_source_id}` as **{label_type}**\n\n"
                    f"Progress: {counts['positive']} positive · "
                    f"{counts['negative']} negative · {counts['skip']} skipped"
                )
                return next_idx, msg

            return handler

        for btn, ltype in [(custom_confirm_btn, "positive"),
                           (custom_reject_btn, "negative"),
                           (custom_skip_btn, "skip")]:
            label_fn = _make_custom_label_handler(ltype)
            btn.click(
                fn=label_fn,
                inputs=[custom_verify_idx, custom_current_source, custom_name],
                outputs=[custom_verify_idx, custom_label_status],
            ).then(
                fn=_custom_load_verify,
                inputs=[custom_verify_idx, custom_name],
                outputs=[custom_verify_info, custom_verify_gallery, custom_verify_idx,
                         custom_current_source, custom_confirm_btn, custom_reject_btn,
                         custom_skip_btn],
            )

        # Few-shot for custom
        def _run_custom_fs(labels_text, target, name, progress=gr.Progress()):
            if not labels_text.strip() or not target.strip():
                return "Enter labels and target first.", [], ""

            labels_list = [l.strip() for l in labels_text.strip().split("\n") if l.strip()]
            if target not in labels_list:
                labels_list.insert(0, target)

            uc_name = name.strip() or "Custom"
            custom_uc = UseCase(
                name=uc_name,
                description="User-defined query",
                labels=labels_list,
                target_label=target,
            )

            cache = _get_cache()
            counts = cache.count_labels(uc_name)
            if counts["positive"] == 0:
                return (
                    "Need at least **1 positive** example. Verify some candidates first.",
                    [], _status_text(),
                )

            runner = _get_runner()
            results = runner.run_few_shot(
                custom_uc,
                progress=lambda f, m: progress(f, desc=m),
            )

            if not results:
                return "Few-shot failed — check logs.", [], _status_text()

            table = []
            for r in results:
                human = r.get("human_label", "")
                marker = {"positive": "✅", "negative": "❌", "skip": "⏭️"}.get(human, "")
                table.append([
                    r["source_id"], r["vehicle_id"], r["date"],
                    f"{r['target_confidence']:.3f}",
                    r["predicted_label"], marker,
                ])

            top = results[0]
            neg_info = ""
            if counts.get("negative", 0) > 0:
                neg_info = f" + {counts['negative']} negative images"

            summary = (
                f"### Few-Shot: {uc_name}\n\n"
                f"Composed query: `{target[:60]}` "
                f"+ {counts['positive']} positive images{neg_info}\n\n"
                f"Re-ranked **{len(results)}** timestamps\n\n"
                f"Top match: **{top['source_id']}** "
                f"(confidence: {top['target_confidence']:.3f})"
            )

            return summary, table, _status_text()

        custom_fs_btn.click(
            fn=_run_custom_fs,
            inputs=[custom_labels, custom_target, custom_name],
            outputs=[custom_fs_summary, custom_fs_table, custom_status_out],
        )


# ═══════════════════════════════════════════════════════════════════
# Main UI
# ═══════════════════════════════════════════════════════════════════

def create_ui() -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.sky,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Source Sans Pro"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    )

    with gr.Blocks(
        title="AV Video Classifier",
        theme=theme,
        css="""
        .gradio-container { max-width: 1400px !important; }
        footer { display: none !important; }
        """,
    ) as app:

        gr.Markdown(
            "# 🚗 AV Video Classifier\n\n"
            "Zero-shot → verify → few-shot classification pipeline "
            "for autonomous vehicle fleet video.\n\n"
            f"**Model:** `{settings.model.model_name.split('/')[-1]}` — "
            f"unified text / image / composed embeddings"
        )

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Accordion("Embed Videos (shared across all tabs)", open=True):
                    gr.Markdown(
                        f"📁 `{settings.storage.video_dir}`\n\n"
                        f"{settings.pipeline.frames_per_video} frames × "
                        f"{len(settings.pipeline.cameras)} cameras = "
                        f"**{settings.pipeline.frames_per_video * len(settings.pipeline.cameras)} "
                        f"frames** per timestamp"
                    )
                    embed_btn = gr.Button("▶ Embed All Videos", variant="primary")
                    embed_output = gr.Markdown("")

            with gr.Column(scale=1):
                status_panel = gr.Markdown(value=_status_text, every=15)

        embed_btn.click(
            fn=_run_embedding,
            outputs=[embed_output, status_panel],
        )

        with gr.Tabs():
            for usecase in ALL_USECASES:
                _build_usecase_tab(usecase)

            _build_custom_tab()

    return app
