"""Gradio UI — tabbed interface for video classification.

Tab per use case. Each tab has the same workflow:
1. Embed videos (shared across tabs) — E5-V image embeddings
2. Run zero-shot classification — E5-V text vs image embeddings
3. Verify top candidates — human marks positive/negative
4. Run few-shot — E5-V composed (text + example images) vs cached image embeddings
"""

from __future__ import annotations

import logging
from pathlib import Path

import gradio as gr

from app.config import settings
from data.models import (
    ALL_USECASES,
    UseCase,
)
from pipeline.cache import Cache
from pipeline.runner import PipelineRunner

logger = logging.getLogger(__name__)

# ── Singleton pipeline runner ────────────────────────────────────

_runner: PipelineRunner | None = None
_cache: Cache | None = None


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


# ═══════════════════════════════════════════════════════════════════
# Shared: Embedding step
# ═══════════════════════════════════════════════════════════════════

def _run_embedding(progress=gr.Progress()):
    """Embed all timestamps (shared across use cases)."""
    runner = _get_runner()

    def prog(frac, msg):
        progress(frac, desc=msg)

    result = runner.embed_all(progress=prog)

    summary = (
        f"**Embedding complete**\n\n"
        f"- Total timestamps: {result['total']}\n"
        f"- Newly embedded: {result['embedded']}\n"
        f"- Skipped (cached): {result['skipped']}\n"
        f"- Time: {result['time_s']}s"
    )
    return summary, _status_text()


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
            return "No results. Run embedding first.", [], _status_text()

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

        summary = (
            f"**Zero-shot results for: {usecase.name}**\n\n"
            f"- Timestamps classified: {len(results)}\n"
            f"- Target class: {usecase.target_label[:60]}\n"
            f"- Top match: {results[0]['source_id']} "
            f"(conf: {results[0]['target_confidence']:.3f})"
            if results else "No results."
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

        info = (
            f"**#{idx + 1} / {len(results)}** — `{r['source_id']}`\n\n"
            f"Vehicle: {r['vehicle_id']} | Date: {r['date']}\n\n"
            f"Target confidence: **{r['target_confidence']:.3f}**\n\n"
            f"Predicted: {r['predicted_label'][:60]}\n\n"
            f"{'🏷️ Already labeled: **' + existing + '**' if existing else ''}"
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
        msg = (
            f"Labeled `{current_source_id}` as **{label_type}**\n\n"
            f"Labels so far: {counts['positive']} positive, "
            f"{counts['negative']} negative, {counts['skip']} skipped"
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
                "Need at least 1 positive example. Verify some candidates first.",
                [],
                _status_text(),
            )

        def prog(frac, msg):
            progress(frac, desc=msg)

        results = runner.run_few_shot(usecase, progress=prog)

        if not results:
            return "Few-shot failed. Check logs.", [], _status_text()

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

        summary = (
            f"**Few-shot results for: {usecase.name}**\n\n"
            f"- Composed query: `{usecase.target_label[:60]}` "
            f"+ {counts['positive']} positive images"
            f"{f' + ' + str(counts.get('negative', 0)) + ' negative images' if counts.get('negative', 0) > 0 else ''}\n"
            f"- Timestamps re-ranked: {len(results)}\n"
            f"- Top match: {results[0]['source_id']} "
            f"(conf: {results[0]['target_confidence']:.3f})\n\n"
            f"*Same cached image embeddings — only the query changed.*"
            if results else "No results."
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
            "### Pipeline Status\n",
            f"- **{status['total_timestamps']}** timestamps "
            f"({status['total_videos']} videos)",
            f"- **{status['embedded_sources']}** embedded",
            f"- Model: `{settings.model.model_name}`",
        ]

        for uc in ALL_USECASES:
            counts = cache.count_labels(uc.name)
            if counts["total"] > 0:
                lines.append(
                    f"- {uc.name}: {counts['positive']}✅ "
                    f"{counts['negative']}❌ {counts['skip']}⏭️"
                )

        return "\n".join(lines)
    except Exception as e:
        return f"Status unavailable: {e}"


# ═══════════════════════════════════════════════════════════════════
# Build a use-case tab
# ═══════════════════════════════════════════════════════════════════

def _build_usecase_tab(usecase: UseCase):
    with gr.Tab(usecase.name):
        gr.Markdown(f"### {usecase.name}\n{usecase.description}")
        gr.Markdown(
            f"**Labels:** {' · '.join(f'`{l[:50]}`' for l in usecase.labels)}\n\n"
            f"**Target:** `{usecase.target_label[:60]}`"
        )

        # ── Step 1: Zero-Shot ────────────────────────────────
        with gr.Accordion("Step 1: Zero-Shot Classification", open=True):
            gr.Markdown(
                "E5-V encodes your text labels and all frame images into the "
                "same vector space, then ranks by cosine similarity."
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
        with gr.Accordion("Step 2: Verify Top Candidates", open=False):
            gr.Markdown(
                "Review zero-shot candidates. **Positive** = genuinely shows the "
                "target scenario. **Negative** = false positive.\n\n"
                "These example images will be combined with the text description "
                "into a single composed embedding for few-shot re-ranking."
            )
            with gr.Row():
                verify_idx = gr.Number(
                    value=0, label="Candidate #", precision=0, minimum=0,
                )
                load_btn = gr.Button("Load Candidate", size="sm")

            verify_info = gr.Markdown("Click 'Load Candidate' to start")
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

        # ── Step 3: Few-Shot ─────────────────────────────────
        with gr.Accordion("Step 3: Few-Shot Re-ranking", open=False):
            gr.Markdown(
                "E5-V creates a **composed embedding** from the target text + "
                "your verified example images → one query vector.\n\n"
                "This single vector captures both *what you described* and "
                "*what it actually looks like in your footage*. "
                "All timestamps are re-ranked against this composed query "
                "using the same cached image embeddings — instant re-ranking, "
                "no re-embedding needed."
            )
            fs_btn = gr.Button("▶ Run Few-Shot", variant="primary")
            fs_summary = gr.Markdown("")
            fs_table = gr.Dataframe(
                headers=["Source ID", "Vehicle", "Date",
                         "Target Conf", "Predicted", "Human Label"],
                datatype=["str", "str", "str", "str", "str", "str"],
                interactive=False,
                wrap=True,
            )

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


# ═══════════════════════════════════════════════════════════════════
# Main UI
# ═══════════════════════════════════════════════════════════════════

def create_ui() -> gr.Blocks:
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
    )

    with gr.Blocks(
        title="AV Video Classifier",
        theme=theme,
    ) as app:

        gr.Markdown(
            "# 🚗 AV Video Classifier\n"
            "Zero-shot → verify → few-shot classification pipeline "
            "for autonomous vehicle fleet video.\n\n"
            f"**Model:** `{settings.model.model_name}` — unified text/image/composed embeddings"
        )

        with gr.Row():
            with gr.Column(scale=3):
                with gr.Accordion("Embed Videos (shared across all tabs)", open=True):
                    gr.Markdown(
                        f"Video directory: `{settings.storage.video_dir}`\n\n"
                        f"Frames per video: {settings.pipeline.frames_per_video} × "
                        f"{len(settings.pipeline.cameras)} cameras = "
                        f"{settings.pipeline.frames_per_video * len(settings.pipeline.cameras)} "
                        f"frames per timestamp"
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

            # Tab 5: Custom Query
            with gr.Tab("Custom Query"):
                gr.Markdown(
                    "### Custom Zero-Shot Query\n\n"
                    "Define your own classification labels."
                )
                custom_labels = gr.Textbox(
                    label="Labels (one per line)",
                    lines=4,
                    placeholder=(
                        "a vehicle making an illegal U-turn\n"
                        "a vehicle turning legally\n"
                        "a road scene with no turning vehicles"
                    ),
                )
                custom_target = gr.Textbox(
                    label="Target label (positive class)",
                    placeholder="a vehicle making an illegal U-turn",
                )
                custom_btn = gr.Button("▶ Run Custom Classification", variant="primary")
                custom_output = gr.Markdown("*Enter labels above*")
                custom_table = gr.Dataframe(
                    headers=["Source ID", "Vehicle", "Date",
                             "Target Conf", "Predicted", "Frames"],
                    interactive=False,
                )

                def _run_custom(labels_text, target, progress=gr.Progress()):
                    if not labels_text.strip() or not target.strip():
                        return "Enter labels and target.", []

                    labels_list = [l.strip() for l in labels_text.strip().split("\n") if l.strip()]
                    if len(labels_list) < 2:
                        return "Need at least 2 labels.", []
                    if target not in labels_list:
                        labels_list.insert(0, target)

                    custom_uc = UseCase(
                        name="Custom",
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
                        return "No results.", []

                    table = [
                        [r["source_id"], r["vehicle_id"], r["date"],
                         f"{r['target_confidence']:.3f}",
                         r["predicted_label"][:50], r["n_frames"]]
                        for r in results
                    ]
                    return f"Classified {len(results)} timestamps.", table

                custom_btn.click(
                    fn=_run_custom,
                    inputs=[custom_labels, custom_target],
                    outputs=[custom_output, custom_table],
                )

    return app
