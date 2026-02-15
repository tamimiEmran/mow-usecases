"""E5-V unified multimodal classifier.

Uses royokong/e5-v (LLaVA-Next 8B) for all embeddings in one vector space:
- Text-only embeddings (zero-shot labels)
- Image-only embeddings (candidate frames)
- Composed text+image embeddings (few-shot queries with example images)

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

# ── Prompt templates (Llama-3 chat format for E5-V) ─────────────

_LLAMA3_TPL = (
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{content}"
    "<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n \n"
)

TEXT_PROMPT = _LLAMA3_TPL.format(
    content="<sent>\nSummary above sentence in one word: "
)

IMAGE_PROMPT = _LLAMA3_TPL.format(
    content="<image>\nSummary above image in one word: "
)

# For composed: image(s) + text instruction → single embedding
# The text goes before "Summary above image in one word:" to guide what
# aspect of the image(s) to focus on.
COMPOSED_PROMPT_SINGLE = _LLAMA3_TPL.format(
    content="<image>\n{text}\nSummary above image in one word: "
)

COMPOSED_PROMPT_MULTI = _LLAMA3_TPL.format(
    content="{image_tokens}\n{text}\nSummary above image in one word: "
)


def load_model(model_name: str, device: str = "cuda"):
    """Load E5-V model + processor (cached).

    E5-V is a LoRA fine-tune of LLaVA-Next. The processor (image preprocessor)
    must be loaded from the base model, while the model weights come from E5-V.
    """
    global _model, _processor, _tokenizer, _load_failed

    if _model is not None:
        return _model, _processor, _tokenizer

    if _load_failed:
        raise RuntimeError("Model loading previously failed. Restart the app to retry.")

    from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

    # The base model that has the processor config (preprocessor_config.json)
    BASE_MODEL = "llava-hf/llama3-llava-next-8b-hf"

    try:
        logger.info("Loading processor from base model: %s", BASE_MODEL)
        _processor = LlavaNextProcessor.from_pretrained(BASE_MODEL)

        logger.info("Loading E5-V model weights: %s", model_name)
        _model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
        ).to(device).eval()
        _tokenizer = _processor.tokenizer
        logger.info("E5-V loaded on %s", device)
    except Exception as e:
        _load_failed = True
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
        logger.info("E5-V model unloaded")


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
    """Encode text labels into E5-V embeddings.

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
    """Encode images into E5-V embeddings.

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
            inputs = processor(IMAGE_PROMPT, img, return_tensors="pt")
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
    """Encode a composed text+image(s) query into a single E5-V embedding.

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
        inputs = processor(prompt, images[0], return_tensors="pt")
    else:
        # Multiple images: create multiple <image> tokens
        image_tokens = "<image>\n" * len(images)
        prompt = COMPOSED_PROMPT_MULTI.format(
            image_tokens=image_tokens.strip(),
            text=text,
        )
        # LlavaNext processor handles list of images mapped to <image> tokens
        inputs = processor(prompt, images, return_tensors="pt")

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
    image_embeddings : (N, dim) — already computed E5-V image embeddings
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
    # Build prototype matrix
    labels = ["positive"]
    protos = [query_embedding[0]]

    if negative_embedding is not None:
        labels.append("negative")
        protos.append(negative_embedding[0])
    else:
        # Without explicit negative, use raw similarity as score
        sims = image_embeddings @ query_embedding.T  # (N, 1)
        results = []
        for i in range(len(sims)):
            score = float(sims[i, 0])
            # Convert similarity to pseudo-probability with sigmoid
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
