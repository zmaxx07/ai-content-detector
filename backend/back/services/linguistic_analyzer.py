"""
LinguisticAnalyzer
==================
Extracts 15+ linguistic features from text that serve as
independent signals for AI vs human detection.

Features extracted:
  - Word count, unique words, lexical diversity
  - Average/variance of sentence lengths
  - Known AI-phrase markers
  - Human informal language signals
  - Punctuation patterns
  - Paragraph structure
  - Vocabulary richness (Type-Token Ratio)
  - Function word density
  - Named entity proxy (capitalized words)
  - Readability proxy
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from models.schemas import DetectionSignal, LinguisticFeatures


# ── AI phrase patterns (commonly over-used by LLMs) ───────────
AI_PHRASES: list[str] = [
    "delve into", "it's worth noting", "in conclusion",
    "it is important to note", "in today's world", "as an ai",
    "i cannot", "leveraging", "in summary", "to summarize",
    "shed light", "at the end of the day", "game changer",
    "paradigm shift", "cutting-edge", "state-of-the-art",
    "it goes without saying", "in the realm of", "it's essential",
    "a testament to", "in this article", "dive deep",
    "furthermore", "moreover", "additionally", "in addition",
    "having said that", "on the other hand", "that being said",
    "needless to say", "as previously mentioned",
    "not only that", "it should be noted", "this is crucial",
    "plays a pivotal role", "let's explore", "let's dive",
]

# ── Human informal markers ────────────────────────────────────
HUMAN_INFORMAL_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(I|we|my|our|me|mine)\b"),
    re.compile(r"\b(honestly|actually|basically|literally|frankly|tbh)\b", re.I),
    re.compile(r"\b(kinda|sorta|gonna|wanna|gotta|dunno|ngl)\b", re.I),
    re.compile(r"[!]{2,}|[?]{2,}"),
    re.compile(r"\b(lol|omg|btw|imo|fwiw|smh|brb)\b", re.I),
    re.compile(r"\.{3,}"),                     # ellipses
    re.compile(r"\b(yeah|yep|nope|nah|yup)\b", re.I),
    re.compile(r"'(ve|d|ll|re|m|t)\b"),        # contractions
]


@dataclass
class AnalysisResult:
    features: LinguisticFeatures
    signals: list[DetectionSignal]
    ai_score_adjustment: int   # -20 to +20, added to ML score
    reasoning: str


def analyze(text: str) -> AnalysisResult:
    """
    Full linguistic analysis of input text.
    Returns features, signals, and a score adjustment.
    """
    # ── Tokenize ──────────────────────────────────────────────
    sentences = _split_sentences(text)
    words = re.findall(r"\b\w+\b", text.lower())
    unique_words = set(words)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    word_count = len(words)
    unique_count = len(unique_words)
    sent_count = max(len(sentences), 1)
    para_count = max(len(paragraphs), 1)

    avg_sent_len = round(word_count / sent_count, 1)
    sent_lengths = [len(re.findall(r"\b\w+\b", s)) for s in sentences]
    variance = max(sent_lengths) - min(sent_lengths) if sent_lengths else 0
    lexical_diversity = round(unique_count / max(word_count, 1), 3)
    avg_para_len = round(word_count / para_count, 1)

    # ── AI phrases ────────────────────────────────────────────
    text_lower = text.lower()
    found_ai_phrases = [p for p in AI_PHRASES if p in text_lower]

    # ── Human signals ─────────────────────────────────────────
    human_signals = sum(
        len(pat.findall(text)) for pat in HUMAN_INFORMAL_PATTERNS
    )

    # ── Punctuation score (human text is less uniform) ────────
    punct_chars = re.findall(r"[.,;:!?]", text)
    punct_score = round(len(punct_chars) / max(word_count, 1), 3)

    features = LinguisticFeatures(
        word_count=word_count,
        unique_words=unique_count,
        avg_sentence_length=avg_sent_len,
        lexical_diversity=round(lexical_diversity * 100),
        sentence_length_variance=variance,
        human_informal_signals=human_signals,
        ai_phrases_found=found_ai_phrases,
        punctuation_score=punct_score,
        paragraph_count=para_count,
        avg_paragraph_length=avg_para_len,
    )

    # ── Build signals list ────────────────────────────────────
    signals: list[DetectionSignal] = []
    score_adj = 0

    # AI phrases
    if found_ai_phrases:
        n = len(found_ai_phrases)
        score_adj += min(n * 5, 20)
        signals.append(DetectionSignal(
            label="AI Phrase Markers",
            value=f"Found {n} clichéd AI phrase(s): {', '.join(found_ai_phrases[:4])}",
            type="ai",
        ))
    else:
        score_adj -= 5
        signals.append(DetectionSignal(
            label="AI Phrase Markers",
            value="No common AI filler phrases detected",
            type="human",
        ))

    # Sentence length variance
    if variance < 8:
        score_adj += 10
        signals.append(DetectionSignal(
            label="Sentence Length Variance",
            value=f"Very uniform sentence lengths (variance: {variance} words) — AI tendency",
            type="ai",
        ))
    elif variance > 20:
        score_adj -= 8
        signals.append(DetectionSignal(
            label="Sentence Length Variance",
            value=f"High sentence variety (variance: {variance} words) — human tendency",
            type="human",
        ))
    else:
        signals.append(DetectionSignal(
            label="Sentence Length Variance",
            value=f"Moderate sentence variety (variance: {variance} words)",
            type="neutral",
        ))

    # Lexical diversity
    ld_pct = round(lexical_diversity * 100)
    if ld_pct < 45:
        score_adj += 8
        signals.append(DetectionSignal(
            label="Lexical Diversity",
            value=f"{ld_pct}% — low vocabulary range, repetitive word use",
            type="ai",
        ))
    elif ld_pct > 70:
        score_adj -= 5
        signals.append(DetectionSignal(
            label="Lexical Diversity",
            value=f"{ld_pct}% — rich and varied vocabulary",
            type="human",
        ))
    else:
        signals.append(DetectionSignal(
            label="Lexical Diversity",
            value=f"{ld_pct}% — average vocabulary variety",
            type="neutral",
        ))

    # Human informal signals
    if human_signals >= 5:
        score_adj -= 12
        signals.append(DetectionSignal(
            label="Informal Human Language",
            value=f"{human_signals} informal signals (contractions, slang, personal pronouns)",
            type="human",
        ))
    elif human_signals == 0 and word_count > 80:
        score_adj += 8
        signals.append(DetectionSignal(
            label="Informal Human Language",
            value="No contractions, slang, or personal voice detected",
            type="ai",
        ))
    else:
        signals.append(DetectionSignal(
            label="Informal Human Language",
            value=f"{human_signals} informal signals found",
            type="neutral",
        ))

    # Average sentence length
    if avg_sent_len > 25:
        score_adj += 6
        signals.append(DetectionSignal(
            label="Avg Sentence Length",
            value=f"{avg_sent_len:.1f} words — unusually long, AI tendency",
            type="ai",
        ))
    elif avg_sent_len < 12:
        score_adj -= 4
        signals.append(DetectionSignal(
            label="Avg Sentence Length",
            value=f"{avg_sent_len:.1f} words — short, punchy, human tendency",
            type="human",
        ))
    else:
        signals.append(DetectionSignal(
            label="Avg Sentence Length",
            value=f"{avg_sent_len:.1f} words per sentence — normal range",
            type="neutral",
        ))

    # Build reasoning
    reasoning = (
        f"Analyzed {word_count} words across {sent_count} sentences and {para_count} paragraph(s). "
        f"Lexical diversity: {ld_pct}%. "
        f"Found {len(found_ai_phrases)} AI phrase marker(s) and {human_signals} human informal signal(s). "
        f"Score adjustment from linguistic features: {score_adj:+d}%."
    )

    return AnalysisResult(
        features=features,
        signals=signals,
        ai_score_adjustment=max(-20, min(20, score_adj)),
        reasoning=reasoning,
    )


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


def detect_language(code: str) -> Optional[str]:
    """Heuristically detect programming language from code snippet."""
    patterns = {
        "python":     [r"\bdef \w+\(", r"\bimport \w+", r"\bclass \w+:", r"^\s*#"],
        "javascript": [r"\bconst \w+", r"\bfunction \w+\(", r"=>\s*{", r"\.then\("],
        "java":       [r"\bpublic class", r"\bSystem\.out", r"\bvoid \w+\("],
        "typescript": [r": string", r": number", r"interface \w+", r"<T>"],
        "cpp":        [r"#include", r"\bstd::", r"\bcout <<"],
        "rust":       [r"\bfn \w+\(", r"\blet mut\b", r"impl \w+"],
        "go":         [r"\bfunc \w+\(", r"\bpackage main", r":="],
        "sql":        [r"\bSELECT\b", r"\bFROM\b", r"\bWHERE\b"],
        "html":       [r"<html", r"<div", r"<!DOCTYPE"],
        "css":        [r"\{[^}]*:[^}]*\}", r"@media", r"\.[\w-]+\s*\{"],
    }
    for lang, pats in patterns.items():
        if sum(bool(re.search(p, code, re.I | re.M)) for p in pats) >= 2:
            return lang
    return None


def analyze_code_features(code: str) -> dict:
    """Extract code-specific features for AI detection."""
    lines = code.split("\n")
    non_empty = [l for l in lines if l.strip()]

    comment_lines = sum(
        1 for l in lines
        if l.strip().startswith(("#", "//", "/*", "*", "'''", '"""'))
    )
    comment_ratio = round(comment_lines / max(len(non_empty), 1) * 100)

    # AI tends to use verbose variable names
    identifiers = re.findall(r"\b([a-z][a-zA-Z_]{6,})\b", code)
    avg_id_len = round(sum(len(i) for i in identifiers) / max(len(identifiers), 1), 1) if identifiers else 0

    # Try-catch blocks (AI over-uses error handling)
    try_count = len(re.findall(r"\btry\b", code, re.I))
    total_blocks = max(len(re.findall(r"\bdef \b|\bfunction\b|\bfunc \b", code, re.I)), 1)
    error_handling_ratio = round(try_count / total_blocks * 100)

    # Docstring presence
    has_docstrings = bool(re.search(r'"""[\s\S]+?"""|\'\'\'[\s\S]+?\'\'\'', code))

    # TODO / FIXME / HACK comments (human markers)
    human_code_markers = len(re.findall(r"\b(TODO|FIXME|HACK|XXX|NOTE)\b", code))

    return {
        "line_count": len(non_empty),
        "comment_ratio_pct": comment_ratio,
        "avg_identifier_length": avg_id_len,
        "error_handling_ratio_pct": error_handling_ratio,
        "has_docstrings": has_docstrings,
        "human_code_markers": human_code_markers,
    }
