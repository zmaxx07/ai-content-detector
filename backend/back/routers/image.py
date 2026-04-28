"""
Image Detection Router
POST /api/v1/detect/image   (multipart/form-data, field: file)
"""

import logging
import time
from io import BytesIO
from typing import Tuple as tuple
from typing import List as list
from fastapi import APIRouter, File, UploadFile, HTTPException

from models.schemas import (
    ImageDetectResponse, BreakdownScore, RawModelScore,
    DetectionSignal, DatasetInfo,
)
from services.model_manager import ModelManager

log = logging.getLogger("router.image")
router = APIRouter()

ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"}
MAX_FILE_MB = 10

IMAGE_DATASET = DatasetInfo(
    name="CIFAKE + GenImage + AIGCDetectBenchmark",
    source="Kaggle / GitHub Research",
    samples="1,200,000+ image pairs",
    hf_dataset="birdy654/CIFAKE + GenImage-dataset/GenImage",
    ai_models_covered=["Stable Diffusion", "DALL-E", "Midjourney", "BigGAN", "StyleGAN", "FLUX"],
)


def _extract_image_features(image_bytes: bytes, mime: str) -> dict:
    """Extract basic image metadata features."""
    try:
        from PIL import Image
        import struct

        img = Image.open(BytesIO(image_bytes))
        width, height = img.size
        mode = img.mode
        aspect = round(width / height, 2)

        # Check for common AI image size signatures
        common_ai_sizes = {(512, 512), (768, 768), (1024, 1024), (512, 768), (768, 512)}
        is_ai_size = (width, height) in common_ai_sizes

        # Estimate color richness
        if img.mode in ("RGB", "RGBA"):
            img_small = img.resize((64, 64))
            pixels = list(img_small.getdata())
            unique_colors = len(set(pixels[:500]))
        else:
            unique_colors = -1

        return {
            "width": width,
            "height": height,
            "aspect_ratio": aspect,
            "mode": mode,
            "file_size_kb": round(len(image_bytes) / 1024, 1),
            "is_common_ai_resolution": is_ai_size,
            "unique_color_sample": unique_colors,
            "format": mime,
        }
    except Exception as e:
        log.debug(f"Image feature extraction error: {e}")
        return {"file_size_kb": round(len(image_bytes) / 1024, 1), "format": mime}


def _build_image_signals(ml_ai: int, features: dict) -> tuple[list[DetectionSignal], int]:
    signals = []
    adj = 0

    # ML score signal
    signals.append(DetectionSignal(
        label="ML Model Score",
        value=f"Image classifier: {ml_ai}% AI probability",
        type="ai" if ml_ai > 55 else "human",
    ))

    # Resolution check
    if features.get("is_common_ai_resolution"):
        adj += 8
        signals.append(DetectionSignal(
            label="Image Resolution",
            value=f"{features['width']}×{features['height']} — matches common AI generation resolution",
            type="ai",
        ))
    else:
        signals.append(DetectionSignal(
            label="Image Resolution",
            value=f"{features.get('width','?')}×{features.get('height','?')} — non-standard AI resolution",
            type="neutral",
        ))

    # Aspect ratio
    ar = features.get("aspect_ratio", 1.0)
    if ar == 1.0:
        adj += 5
        signals.append(DetectionSignal(
            label="Aspect Ratio",
            value="Perfect 1:1 square — common AI generation default",
            type="ai",
        ))
    else:
        signals.append(DetectionSignal(
            label="Aspect Ratio",
            value=f"Ratio {ar} — non-square proportions",
            type="neutral",
        ))

    # File size heuristic
    size_kb = features.get("file_size_kb", 0)
    if size_kb < 50:
        adj += 4
        signals.append(DetectionSignal(
            label="File Size",
            value=f"{size_kb}KB — small file size, common in AI-generated images",
            type="ai",
        ))

    return signals, max(-15, min(15, adj))


@router.post("/detect/image", response_model=ImageDetectResponse, summary="Detect AI-generated images")
async def detect_image(file: UploadFile = File(..., description="Image file (JPEG/PNG/WebP/GIF)")):
    t_start = time.time()

    # Validate file type
    mime = file.content_type or "image/jpeg"
    if mime not in ALLOWED_MIME:
        raise HTTPException(400, detail=f"Unsupported file type: {mime}. Allowed: {ALLOWED_MIME}")

    # Read bytes
    image_bytes = await file.read()
    size_mb = len(image_bytes) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        raise HTTPException(413, detail=f"File too large ({size_mb:.1f}MB). Max {MAX_FILE_MB}MB.")

    log.info(f"Image detection: {file.filename}, {size_mb:.2f}MB, {mime}")

    # ── ML Scoring ────────────────────────────────────────────
    try:
        ml_result = await ModelManager.predict_image(image_bytes, mime)
    except Exception as e:
        log.error(f"Image ML model failed: {e}")
        raise HTTPException(503, detail=f"Image model unavailable: {e}")

    ai_raw = ml_result["ai_score"]
    human_raw = ml_result["human_score"]

    # ── Feature extraction ────────────────────────────────────
    features = _extract_image_features(image_bytes, mime)
    signals, adj = _build_image_signals(ai_raw, features)

    # Blended score
    ai_blended = max(0, min(100, ai_raw + adj))
    human_blended = 100 - ai_blended

    # Verdict
    if ai_blended >= 75:
        verdict, conf = "AI-Generated", ai_blended
    elif ai_blended >= 55:
        verdict, conf = "Likely AI", ai_blended
    elif ai_blended <= 25:
        verdict, conf = "Human-Created", human_blended
    else:
        verdict, conf = "Likely Human", human_blended

    raw_scores = [
        RawModelScore(label=s.get("label", "?"), score=round(s.get("score", 0), 4))
        for s in ml_result["raw_scores"]
    ]

    elapsed = round((time.time() - t_start) * 1000)
    log.info(f"Image detection done: verdict={verdict}, time={elapsed}ms")

    return ImageDetectResponse(
        verdict=verdict,
        confidence=conf,
        summary=(
            f"The image was classified as '{verdict}' with {conf}% confidence. "
            f"ML model score: {ai_raw}% AI probability. "
            f"Resolution {features.get('width','?')}×{features.get('height','?')}, "
            f"size {features.get('file_size_kb','?')}KB."
        ),
        breakdown=BreakdownScore(
            ai_score=ai_blended,
            human_score=human_blended,
            raw_scores=raw_scores,
        ),
        signals=signals,
        image_features=features,
        model_used=ml_result["model"],
        dataset_info=IMAGE_DATASET,
        processing_time_ms=elapsed,
    )
