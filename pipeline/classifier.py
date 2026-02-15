"""SigLIP-based zero-shot and few-shot classifier.

Zero-shot: encode text labels + images, pick highest cosine similarity.
Few-shot: prototype-based — mean embedding per class from verified examples,
          classify by nearest prototype.
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


def load_model(model_name: str, device: str = "cuda"):
    """Load SigLIP model + processor (cached)."""
    global _model, _processor

    if _model is not None:
        return _model, _processor

    from transformers import AutoModel, AutoProcessor

    logger.info("Loading classifier model: %s", model_name)
    _processor = AutoProcessor.from_pretrained(model_name)
    _model = AutoModel.from_pretrained(model_name).to(device).eval()
    logger.info("Model loaded on %s", device)

    return _model, _processor


def unload_model():
    """Free GPU memory."""
    global _model, _processor
    if _model is not None:
        del _model, _processor
        _model = _processor = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded")


def encode_images(
    images: list[Image.Image],
    model_name: str,
    device: str = "cuda",
    batch_size: int = 32,
) -> np.ndarray:
    """Encode images into SigLIP embeddings.

    Returns
    -------
    np.ndarray shape (N, dim), float32, L2-normalised.
    """
    model, processor = load_model(model_name, device)
    all_embeddings = []

    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size]
        inputs = processor(images=batch, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.inference_mode():
            features = model.get_image_features(**inputs)

        emb = features.cpu().numpy().astype(np.float32)
        all_embeddings.append(emb)

    result = np.concatenate(all_embeddings, axis=0)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    result = result / np.maximum(norms, 1e-8)
    return result


def encode_texts(
    texts: list[str],
    model_name: str,
    device: str = "cuda",
) -> np.ndarray:
    """Encode text labels into SigLIP embeddings.

    Returns
    -------
    np.ndarray shape (N, dim), float32, L2-normalised.
    """
    model, processor = load_model(model_name, device)
    inputs = processor(text=texts, padding="max_length", return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        features = model.get_text_features(**inputs)

    result = features.cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    result = result / np.maximum(norms, 1e-8)
    return result


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
    image_embeddings : (N, dim)
    labels : list of text labels
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
    prototypes: dict[str, np.ndarray],
    temperature: float = 0.5,
) -> list[dict[str, float]]:
    """Few-shot prototype classification.

    Parameters
    ----------
    image_embeddings : (N, dim)
    prototypes : {label: (dim,) mean embedding}
    temperature : softmax temperature

    Returns
    -------
    list of dicts, each {label: probability}
    """
    labels = list(prototypes.keys())
    proto_matrix = np.stack([prototypes[l] for l in labels], axis=0)  # (L, dim)

    # Normalise prototypes
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


def build_prototypes(
    embeddings_by_label: dict[str, list[np.ndarray]],
) -> dict[str, np.ndarray]:
    """Build prototype (mean) embeddings from labeled examples.

    Parameters
    ----------
    embeddings_by_label : {label: [array(dim,), ...]}

    Returns
    -------
    {label: array(dim,)}
    """
    prototypes = {}
    for label, emb_list in embeddings_by_label.items():
        if not emb_list:
            continue
        stacked = np.stack(emb_list, axis=0)  # (K, dim)
        mean = stacked.mean(axis=0)
        # Re-normalise
        norm = np.linalg.norm(mean)
        if norm > 1e-8:
            mean = mean / norm
        prototypes[label] = mean
        logger.info("Prototype for '%s': %d examples", label, len(emb_list))

    return prototypes
