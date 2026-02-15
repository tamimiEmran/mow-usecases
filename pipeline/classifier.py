"""LLaVA-Next Mistral multimodal classifier.

Uses llava-hf/llava-v1.6-mistral-7b-hf for all embeddings in one vector space:
- Text-only embeddings (zero-shot labels)
- Image-only embeddings (candidate frames)
- Composed text+image embeddings (few-shot queries with example images)

Embedding is the last token's hidden state, L2-normalised.
All three share the same embedding space, so cosine similarity works across
any combination.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)

# Module-level model cache
_model = None
_processor = None
_tokenizer = None
_load_failed = False

# ── Prompt templates (Mistral [INST] format for LLaVA-Next Mistral) ──

# Mistral chat format: [INST] content [/INST]
TEXT_PROMPT = "[INST] <sent>\nSummary above sentence in one word: [/INST]"

IMAGE_PROMPT = "[INST] <image>\nSummary above image in one word: [/INST]"

# For composed: image(s) + text instruction → single embedding
COMPOSED_PROMPT_SINGLE = "[INST] <image>\n{text}\nSummary above image in one word: [/INST]"

COMPOSED_PROMPT_MULTI = "[INST] {image_tokens}\n{text}\nSummary above image in one word: [/INST]"


def _ensure_processor_patch_size(processor, model=None):
    """Ensure the processor has patch_size and vision_feature_select_strategy set.

    This is the root cause of the 'unsupported operand type(s) for //: int and NoneType'
    error — the processor tries to compute image_size // patch_size during image
    processing, and if patch_size is None, it crashes.
    """
    needs_patch_size = getattr(processor, 'patch_size', None) is None
    needs_strategy = getattr(processor, 'vision_feature_select_strategy', None) is None

    if not needs_patch_size and not needs_strategy:
        return

    # Try to pull from the model's config
    patch_size = None
    strategy = None

    if model is not None:
        config = getattr(model, 'config', None)
        if config is not None:
            # Check model.config.vision_config.patch_size
            vision_config = getattr(config, 'vision_config', None)
            if vision_config is not None:
                patch_size = getattr(vision_config, 'patch_size', None)
            # Check model.config.vision_feature_select_strategy
            strategy = getattr(config, 'vision_feature_select_strategy', None)

    # Fallback defaults for CLIP-ViT-L/14-336px (used by all LLaVA-Next variants)
    if patch_size is None:
        patch_size = 14
    if strategy is None:
        strategy = "default"

    if needs_patch_size:
        processor.patch_size = patch_size
        logger.info("Set processor.patch_size = %d", patch_size)
    if needs_strategy:
        processor.vision_feature_select_strategy = strategy
        logger.info("Set processor.vision_feature_select_strategy = '%s'", strategy)


def load_model(model_name: str, device: str = "cuda"):
    """Load LLaVA-Next model + processor (cached)."""
    global _model, _processor, _tokenizer, _load_failed

    if _model is not None:
        return _model, _processor, _tokenizer

    if _load_failed:
        raise RuntimeError("Model loading previously failed. Restart the app to retry.")

    from transformers import LlavaNextForConditionalGeneration, LlavaNextProcessor

    try:
        logger.info("Loading processor from %s", model_name)
        _processor = LlavaNextProcessor.from_pretrained(model_name)

        logger.info("Loading model weights: %s", model_name)
        _model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,      # was `dtype` — wrong kwarg, silently ignored
            low_cpu_mem_usage=True,
        ).to(device).eval()

        # Critical: ensure patch_size is set BEFORE any image processing
        _ensure_processor_patch_size(_processor, _model)

        _tokenizer = _processor.tokenizer
        logger.info("LLaVA-Next loaded on %s (patch_size=%s)",
                     device, getattr(_processor, 'patch_size', '?'))
    except Exception as e:
        _load_failed = True
        _model = _processor = _tokenizer = None
        logger.error("Failed to load model: %s", e)
        raise

    return _model, _processor, _tokenizer


def unload_model():
    """Free GPU memory."""
    global _model, _processor, _tokenizer, _load_failed
    if _model is not None:
        del _model, _processor, _tokenizer
        _model = _processor = _tokenizer = None
        _load_failed = False
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded")


def _get_last_token_embedding(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    **kwargs,
) -> np.ndarray:
    """Run model forward and extract the last token's hidden state as embedding."""
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
    # Last hidden state, last token for each item in batch
    hidden = outputs.hidden_states[-1]  # (batch, seq_len, dim)

    # Find last non-padding token per sequence
    seq_lengths = attention_mask.sum(dim=1) - 1  # (batch,)
    embeddings = []
    for i in range(hidden.shape[0]):
        emb = hidden[i, seq_lengths[i], :].float().cpu().numpy()
        embeddings.append(emb)

    result = np.stack(embeddings, axis=0)
    # L2 normalise
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    result = result / np.maximum(norms, 1e-8)
    return result


# ── Public API ───────────────────────────────────────────────────


def encode_texts(
    texts: list[str],
    model_name: str,
    device: str = "cuda",
) -> np.ndarray:
    """Encode text labels into embeddings.

    Returns
    -------
    np.ndarray shape (N, dim), float32, L2-normalised.
    """
    model, processor, tokenizer = load_model(model_name, device)
    all_embeddings = []

    for text in texts:
        prompt = TEXT_PROMPT.replace("<sent>", text)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        emb = _get_last_token_embedding(model, **inputs)
        all_embeddings.append(emb)

    return np.concatenate(all_embeddings, axis=0)


def encode_images(
    images: list[Image.Image],
    model_name: str,
    device: str = "cuda",
    batch_size: int = 8,
) -> np.ndarray:
    """Encode images into embeddings.

    Returns
    -------
    np.ndarray shape (N, dim), float32, L2-normalised.
    """
    model, processor, tokenizer = load_model(model_name, device)
    all_embeddings = []

    for start in range(0, len(images), batch_size):
        batch_imgs = images[start:start + batch_size]
        batch_embs = []

        for img in batch_imgs:
            inputs = processor(images=img, text=IMAGE_PROMPT, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            emb = _get_last_token_embedding(model, **inputs)
            batch_embs.append(emb)

        all_embeddings.extend(batch_embs)

    return np.concatenate(all_embeddings, axis=0)


def encode_composed(
    text: str,
    images: list[Image.Image],
    model_name: str,
    device: str = "cuda",
) -> np.ndarray:
    """Encode a composed text+image(s) query into a single embedding.

    This is the key function for few-shot: combine the target description
    text with verified example images into one query vector.

    Parameters
    ----------
    text : str
        The target description (e.g., "a bus stop with a car parked at it").
    images : list[Image.Image]
        One or more example images to condition on.

    Returns
    -------
    np.ndarray shape (1, dim), float32, L2-normalised.
    """
    model, processor, tokenizer = load_model(model_name, device)

    if len(images) == 1:
        prompt = COMPOSED_PROMPT_SINGLE.format(text=text)
        inputs = processor(images=images[0], text=prompt, return_tensors="pt")
    else:
        # Multiple images: create multiple <image> tokens
        image_tokens = "<image>\n" * len(images)
        prompt = COMPOSED_PROMPT_MULTI.format(
            image_tokens=image_tokens.strip(),
            text=text,
        )
        inputs = processor(images=images, text=prompt, return_tensors="pt")

    inputs = {k: v.to(device) for k, v in inputs.items()}
    emb = _get_last_token_embedding(model, **inputs)
    return emb


# ── Classification functions ─────────────────────────────────────


def zero_shot_classify(
    image_embeddings: np.ndarray,
    labels: list[str],
    model_name: str,
    device: str = "cuda",
    temperature: float = 1.0,
) -> list[dict[str, float]]:
    """Zero-shot classification via cosine similarity.

    Parameters
    ----------
    image_embeddings : (N, dim) — already computed image embeddings
    labels : text labels to compare against
    temperature : softmax temperature (lower = sharper)

    Returns
    -------
    list of dicts, each {label: probability}
    """
    text_embeddings = encode_texts(labels, model_name, device)

    # Cosine similarity: (N, dim) @ (dim, L) → (N, L)
    similarities = image_embeddings @ text_embeddings.T

    # Softmax over labels
    logits = similarities / temperature
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    results = []
    for i in range(len(probs)):
        scores = {label: float(probs[i, j]) for j, label in enumerate(labels)}
        results.append(scores)

    return results


def few_shot_classify(
    image_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    negative_embedding: np.ndarray | None = None,
    temperature: float = 0.5,
) -> list[dict[str, float]]:
    """Few-shot classification using composed query embedding.

    Parameters
    ----------
    image_embeddings : (N, dim) — candidate image embeddings (cached)
    query_embedding : (1, dim) — composed text+images "positive" embedding
    negative_embedding : (1, dim) or None — composed "negative" embedding
    temperature : softmax temperature

    Returns
    -------
    list of dicts, each {"positive": prob, "negative": prob}
    """
    labels = ["positive"]
    protos = [query_embedding[0]]

    if negative_embedding is not None:
        labels.append("negative")
        protos.append(negative_embedding[0])
    else:
        sims = image_embeddings @ query_embedding.T  # (N, 1)
        results = []
        for i in range(len(sims)):
            score = float(sims[i, 0])
            prob = 1.0 / (1.0 + np.exp(-score / temperature))
            results.append({"positive": prob, "negative": 1.0 - prob})
        return results

    proto_matrix = np.stack(protos, axis=0)  # (L, dim)
    norms = np.linalg.norm(proto_matrix, axis=1, keepdims=True)
    proto_matrix = proto_matrix / np.maximum(norms, 1e-8)

    similarities = image_embeddings @ proto_matrix.T  # (N, L)

    logits = similarities / temperature
    exp_logits = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    results = []
    for i in range(len(probs)):
        scores = {label: float(probs[i, j]) for j, label in enumerate(labels)}
        results.append(scores)

    return results
