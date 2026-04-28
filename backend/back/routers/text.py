"""
Text Detection Router
POST /api/v1/detect/text
"""

import logging
import time
from typing import Tuple as tuple
from fastapi import APIRouter, HTTPException

from models.schemas import (
    TextDetectRequest, TextDetectResponse,
    BreakdownScore, RawModelScore, DetectionSignal, DatasetInfo,
)
from services.model_manager import ModelManager
from services.source_fetcher import fetch_human_samples, compute_vocabulary_similarity
from services.linguistic_analyzer import analyze

log = logging.getLogger("router.text")
router = APIRouter()

TEXT_DATASET = DatasetInfo(
    name="GPT-2 Output Dataset + AI Text Detection Pile",
    source="OpenAI / Hugging Face Hub",
    samples="630,000 labeled documents",
    hf_dataset="openai-community/gpt2 + artem9k/ai-text-detection-pile",
    ai_models_covered=["GPT-2 (all sizes)", "GPT-3", "ChatGPT", "GPT-J", "GPT-NeoX"],
)


def _build_verdict(ai_score: int, confidence_threshold: int = 60) -> tuple[str, int]:
    if ai_score >= 80:
        return "AI-Generated", ai_score
    if ai_score >= confidence_threshold:
        return "Likely AI", ai_score
    if ai_score <= 20:
        return "Human-Written", 100 - ai_score
    return "Likely Human", 100 - ai_score


@router.post("/detect/text", response_model=TextDetectResponse, summary="Detect AI-generated text")
async def detect_text(req: TextDetectRequest):
    t_start = time.time()
    log.info(f"Text detection request: {len(req.text)} chars, topic='{req.topic}'")

    # ── Step 1: ML model scoring ──────────────────────────────
    try:
        ml_result = await ModelManager.predict_text(req.text)
    except Exception as e:
        log.error(f"ML model failed: {e}")
        raise HTTPException(503, detail=f"ML model unavailable: {e}")

    ai_raw = ml_result["ai_score"]
    human_raw = ml_result["human_score"]

    # ── Step 2: Linguistic analysis ───────────────────────────
    ling = analyze(req.text)

    # Blend ML score with linguistic adjustment
    ai_blended = max(0, min(100, ai_raw + ling.ai_score_adjustment))
    human_blended = 100 - ai_blended

    # ── Step 3: Fetch live human references ───────────────────
    human_samples = []
    if req.fetch_human_references:
        try:
            raw_samples = await fetch_human_samples(req.topic)
            # Annotate with similarity score
            for s in raw_samples:
                s.similarity = compute_vocabulary_similarity(req.text, s.text)
            human_samples = sorted(raw_samples, key=lambda x: -(x.similarity or 0))
        except Exception as e:
            log.warning(f"Human source fetch failed: {e}")

    # ── Step 4: Comparison note ───────────────────────────────
    comparison_note = None
    if human_samples:
        top = human_samples[0]
        sim = top.similarity or 0
        if sim > 40:
            comparison_note = (
                f"Vocabulary overlap with {top.source} ({sim}%) is high, "
                "suggesting similar domain language to human-written text."
            )
        elif sim < 15:
            comparison_note = (
                f"Low vocabulary overlap with {top.source} ({sim}%) — "
                "the writing style diverges from reference human texts."
            )
        else:
            comparison_note = (
                f"Moderate vocabulary overlap with human references "
                f"(top match: {top.source}, {sim}%)."
            )

    # ── Step 5: Verdict ───────────────────────────────────────
    verdict, confidence = _build_verdict(ai_blended)

    # ── Build response ────────────────────────────────────────
    raw_scores = [
        RawModelScore(label=s.get("label", "?"), score=round(s.get("score", 0), 4))
        for s in ml_result["raw_scores"]
    ]

    signals = ling.signals + [
        DetectionSignal(
            label="ML Model (RoBERTa)",
            value=f"Raw AI probability: {ai_raw}% | Human: {human_raw}%",
            type="ai" if ai_raw > 50 else "human",
        )
    ]
    if human_samples:
        top_sim = human_samples[0].similarity or 0
        signals.append(DetectionSignal(
            label="Human Reference Similarity",
            value=f"Best match to live human text: {top_sim}% vocabulary overlap",
            type="human" if top_sim > 35 else "neutral",
        ))

    elapsed_ms = round((time.time() - t_start) * 1000)
    log.info(f"Text detection done: verdict={verdict}, confidence={confidence}%, time={elapsed_ms}ms")

    return TextDetectResponse(
        verdict=verdict,
        confidence=confidence,
        summary=(
            f"The text was classified as '{verdict}' with {confidence}% confidence. "
            f"The RoBERTa ML model scored it {ai_raw}% AI. "
            f"{ling.reasoning}"
        ),
        breakdown=BreakdownScore(
            ai_score=ai_blended,
            human_score=human_blended,
            raw_scores=raw_scores,
        ),
        linguistic_features=ling.features,
        signals=signals,
        human_references=human_samples,
        comparison_note=comparison_note,
        key_evidence=_pick_key_evidence(ling, ai_raw),
        model_used=ml_result["model"],
        dataset_info=TEXT_DATASET,
        processing_time_ms=elapsed_ms,
    )


def _pick_key_evidence(ling, ai_raw: int) -> str:
    if ling.features.ai_phrases_found:
        return f"AI phrase markers detected: {', '.join(ling.features.ai_phrases_found[:3])}"
    if ai_raw >= 75:
        return f"RoBERTa ML model gave {ai_raw}% AI probability — strong signal"
    if ling.features.sentence_length_variance < 8:
        return "Sentence lengths are unusually uniform — a strong AI writing pattern"
    if ling.features.human_informal_signals >= 5:
        return f"High informal language density ({ling.features.human_informal_signals} signals) — human indicator"
    return f"Blended score of {ai_raw}% from ML + {ling.ai_score_adjustment:+d}% linguistic adjustment"
