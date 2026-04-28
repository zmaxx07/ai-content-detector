"""
Pydantic schemas for all API request and response types.
"""

from __future__ import annotations
from typing import Optional
from typing import List as list
from pydantic import BaseModel, Field, validator


# ── Shared ────────────────────────────────────────────────────

class RawModelScore(BaseModel):
    label: str
    score: float

class BreakdownScore(BaseModel):
    ai_score: int   = Field(..., ge=0, le=100)
    human_score: int = Field(..., ge=0, le=100)
    raw_scores: list[RawModelScore] = []

class DetectionSignal(BaseModel):
    label: str
    value: str
    type: str   # "ai" | "human" | "neutral"

class HumanSample(BaseModel):
    source: str
    url: str
    text: str
    title: Optional[str] = None
    author: Optional[str] = None
    sample_type: str = "article"
    icon: str = "📄"
    similarity: Optional[int] = None    # vocab overlap % with input

class DatasetInfo(BaseModel):
    name: str
    source: str
    samples: str
    hf_dataset: str
    ai_models_covered: list[str]

class LinguisticFeatures(BaseModel):
    word_count: int
    unique_words: int
    avg_sentence_length: float
    lexical_diversity: float            # unique/total ratio
    sentence_length_variance: int
    human_informal_signals: int
    ai_phrases_found: list[str]
    punctuation_score: float
    paragraph_count: int
    avg_paragraph_length: float


# ── Text ──────────────────────────────────────────────────────

class TextDetectRequest(BaseModel):
    text: str = Field(..., min_length=40, max_length=50_000)
    topic: str = Field("technology", max_length=100)
    fetch_human_references: bool = True
    use_claude_enrichment: bool = False  # set True if Anthropic key is set

    @validator("text")
    def strip_text(cls, v):
        return v.strip()

class TextDetectResponse(BaseModel):
    verdict: str
    confidence: int
    summary: str
    breakdown: BreakdownScore
    linguistic_features: LinguisticFeatures
    signals: list[DetectionSignal]
    human_references: list[HumanSample]
    comparison_note: Optional[str] = None
    key_evidence: Optional[str] = None
    model_used: str
    dataset_info: DatasetInfo
    processing_time_ms: int


# ── Image ─────────────────────────────────────────────────────

class ImageDetectResponse(BaseModel):
    verdict: str
    confidence: int
    summary: str
    breakdown: BreakdownScore
    signals: list[DetectionSignal]
    image_features: dict
    model_used: str
    dataset_info: DatasetInfo
    processing_time_ms: int


# ── Code ──────────────────────────────────────────────────────

class CodeDetectRequest(BaseModel):
    code: str = Field(..., min_length=10, max_length=50_000)
    language: Optional[str] = None      # "python", "javascript", etc.
    use_claude_enrichment: bool = False

class CodeDetectResponse(BaseModel):
    verdict: str
    confidence: int
    summary: str
    breakdown: BreakdownScore
    signals: list[DetectionSignal]
    code_features: dict
    detected_language: Optional[str]
    model_used: str
    dataset_info: DatasetInfo
    processing_time_ms: int


# ── Sources ───────────────────────────────────────────────────

class SourcesResponse(BaseModel):
    topic: str
    samples: List[HumanSample]
    total_fetched: int
    sources_used: list[str]
    fetch_time_ms: int


# ── Health ────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    models_loaded: dict
    inference_mode: str
    version: str
    uptime_seconds: float
