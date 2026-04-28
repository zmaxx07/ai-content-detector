"""
Code Detection Router
POST /api/v1/detect/code
"""

import logging
import time
from typing import Tuple as tuple
from typing import List as list
from fastapi import APIRouter, HTTPException

from models.schemas import (
    CodeDetectRequest, CodeDetectResponse,
    BreakdownScore, RawModelScore, DetectionSignal, DatasetInfo,
)
from services.model_manager import ModelManager
from services.linguistic_analyzer import detect_language, analyze_code_features

log = logging.getLogger("router.code")
router = APIRouter()

CODE_DATASET = DatasetInfo(
    name="CodeSearchNet + GitHub Code + AI-CodeBench",
    source="GitHub / Hugging Face",
    samples="2,000,000+ code snippets",
    hf_dataset="code_search_net / bigcode/the-stack",
    ai_models_covered=["GitHub Copilot", "ChatGPT", "CodeLlama", "StarCoder", "Claude"],
)


def _build_code_signals(ml_ai: int, code_features: dict) -> tuple[list[DetectionSignal], int]:
    signals = []
    adj = 0

    # ML signal
    signals.append(DetectionSignal(
        label="ML Model (RoBERTa)",
        value=f"Code authorship model: {ml_ai}% AI probability",
        type="ai" if ml_ai > 55 else "human",
    ))

    # Comment density — AI over-comments
    cr = code_features.get("comment_ratio_pct", 0)
    if cr > 40:
        adj += 10
        signals.append(DetectionSignal(
            label="Comment Density",
            value=f"{cr}% of lines are comments — AI tendency to over-explain",
            type="ai",
        ))
    elif cr < 5:
        adj -= 5
        signals.append(DetectionSignal(
            label="Comment Density",
            value=f"{cr}% comment density — minimal comments, human style",
            type="human",
        ))
    else:
        signals.append(DetectionSignal(
            label="Comment Density",
            value=f"{cr}% comment density — normal range",
            type="neutral",
        ))

    # Identifier length — AI uses verbose names
    avg_id = code_features.get("avg_identifier_length", 0)
    if avg_id > 14:
        adj += 8
        signals.append(DetectionSignal(
            label="Variable Naming",
            value=f"Avg identifier length: {avg_id:.1f} chars — very verbose, AI signature",
            type="ai",
        ))
    elif avg_id < 6:
        adj -= 5
        signals.append(DetectionSignal(
            label="Variable Naming",
            value=f"Short identifiers ({avg_id:.1f} chars avg) — human shorthand style",
            type="human",
        ))
    else:
        signals.append(DetectionSignal(
            label="Variable Naming",
            value=f"Average identifier length: {avg_id:.1f} chars",
            type="neutral",
        ))

    # Error handling ratio
    ehr = code_features.get("error_handling_ratio_pct", 0)
    if ehr > 60:
        adj += 7
        signals.append(DetectionSignal(
            label="Error Handling",
            value=f"{ehr}% of functions have try/except — AI boilerplate tendency",
            type="ai",
        ))

    # Docstrings — AI adds them to everything
    if code_features.get("has_docstrings"):
        adj += 5
        signals.append(DetectionSignal(
            label="Docstrings Present",
            value="Docstrings detected — AI commonly adds these to all functions",
            type="ai",
        ))

    # Human markers (TODOs, FIXMEs)
    hcm = code_features.get("human_code_markers", 0)
    if hcm > 0:
        adj -= 8
        signals.append(DetectionSignal(
            label="Human Code Markers",
            value=f"{hcm} TODO/FIXME/HACK comments — strong human indicator",
            type="human",
        ))

    return signals, max(-15, min(15, adj))


@router.post("/detect/code", response_model=CodeDetectResponse, summary="Detect AI-generated code")
async def detect_code(req: CodeDetectRequest):
    t_start = time.time()
    log.info(f"Code detection: {len(req.code)} chars, lang={req.language}")

    # Detect language if not provided
    detected_lang = req.language or detect_language(req.code)

    # Extract code features
    code_features = analyze_code_features(req.code)
    log.info(f"Code features: {code_features}")

    # ML scoring
    try:
        ml_result = await ModelManager.predict_code(req.code)
    except Exception as e:
        log.error(f"Code ML model failed: {e}")
        raise HTTPException(503, detail=f"Code ML model unavailable: {e}")

    ai_raw = ml_result["ai_score"]
    human_raw = ml_result["human_score"]

    # Build signals + adjustment
    signals, adj = _build_code_signals(ai_raw, code_features)

    ai_blended = max(0, min(100, ai_raw + adj))
    human_blended = 100 - ai_blended

    # Verdict
    if ai_blended >= 75:
        verdict, conf = "AI-Generated", ai_blended
    elif ai_blended >= 55:
        verdict, conf = "Likely AI", ai_blended
    elif ai_blended <= 25:
        verdict, conf = "Human-Written", human_blended
    else:
        verdict, conf = "Likely Human", human_blended

    raw_scores = [
        RawModelScore(label=s.get("label", "?"), score=round(s.get("score", 0), 4))
        for s in ml_result["raw_scores"]
    ]

    elapsed = round((time.time() - t_start) * 1000)
    log.info(f"Code detection done: verdict={verdict}, lang={detected_lang}, time={elapsed}ms")

    return CodeDetectResponse(
        verdict=verdict,
        confidence=conf,
        summary=(
            f"The code snippet was classified as '{verdict}' with {conf}% confidence. "
            f"Language detected: {detected_lang or 'unknown'}. "
            f"Comment density: {code_features['comment_ratio_pct']}%. "
            f"ML model gave {ai_raw}% AI probability, adjusted to {ai_blended}% after code feature analysis."
        ),
        breakdown=BreakdownScore(
            ai_score=ai_blended,
            human_score=human_blended,
            raw_scores=raw_scores,
        ),
        signals=signals,
        code_features=code_features,
        detected_language=detected_lang,
        model_used=ml_result["model"],
        dataset_info=CODE_DATASET,
        processing_time_ms=elapsed,
    )
