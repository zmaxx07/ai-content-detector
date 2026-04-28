"""
ModelManager
============
Central registry for all ML models.

Supports two inference modes:
  • "local"  — downloads models from HuggingFace Hub and runs on local CPU/GPU
  • "api"    — calls HuggingFace Inference API (no local download needed)

Usage:
    await ModelManager.load_all()          # call once at startup
    scores = await ModelManager.predict_text("some text")
    scores = await ModelManager.predict_image(image_bytes)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import aiohttp

from config import get_settings

log = logging.getLogger("model_manager")
settings = get_settings()

# HuggingFace Inference API base
HF_INFERENCE_BASE = "https://api-inference.huggingface.co/models"


class _LocalTextModel:
    """Loads RoBERTa locally via transformers pipeline."""

    def __init__(self):
        self.pipeline = None
        self.name = settings.text_model_name

    def load(self):
        try:
            from transformers import pipeline
            import torch

            device = self._resolve_device()
            log.info(f"Loading text model '{self.name}' on {device}...")
            self.pipeline = pipeline(
                "text-classification",
                model=self.name,
                device=device,
                truncation=True,
                max_length=settings.max_text_length,
            )
            log.info(f"✅ Text model loaded: {self.name}")
        except Exception as e:
            log.warning(f"Failed to load primary text model: {e}. Trying fallback...")
            try:
                from transformers import pipeline
                self.pipeline = pipeline(
                    "text-classification",
                    model=settings.text_model_fallback,
                    truncation=True,
                    max_length=settings.max_text_length,
                )
                self.name = settings.text_model_fallback
                log.info(f"✅ Fallback text model loaded: {self.name}")
            except Exception as e2:
                log.error(f"Both text models failed: {e2}")

    def predict(self, text: str) -> list[dict]:
        if not self.pipeline:
            return [{"label": "Real", "score": 0.5}, {"label": "Fake", "score": 0.5}]
        result = self.pipeline(text[:settings.max_text_length])
        # pipeline returns list of {label, score}
        return result if isinstance(result, list) else [result]

    def _resolve_device(self) -> int:
        try:
            import torch
            if settings.device == "auto":
                if torch.cuda.is_available():
                    log.info("CUDA detected — using GPU")
                    return 0
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    log.info("MPS (Apple Silicon) detected")
                    return "mps"
                return -1  # CPU
            return 0 if settings.device == "cuda" else -1
        except ImportError:
            return -1


class _LocalImageModel:
    """Loads image classifier locally."""

    def __init__(self):
        self.pipeline = None
        self.name = settings.image_model_name

    def load(self):
        try:
            from transformers import pipeline
            log.info(f"Loading image model '{self.name}'...")
            self.pipeline = pipeline(
                "image-classification",
                model=self.name,
                truncation=True,
            )
            log.info(f"✅ Image model loaded: {self.name}")
        except Exception as e:
            log.warning(f"Image model load failed: {e}. Using fallback...")
            try:
                from transformers import pipeline
                self.pipeline = pipeline(
                    "image-classification",
                    model=settings.image_model_fallback,
                )
                self.name = settings.image_model_fallback
                log.info(f"✅ Fallback image model loaded: {self.name}")
            except Exception as e2:
                log.error(f"Both image models failed: {e2}")

    def predict(self, image_bytes: bytes) -> list[dict]:
        if not self.pipeline:
            return [{"label": "artificial", "score": 0.5}, {"label": "real", "score": 0.5}]
        from io import BytesIO
        from PIL import Image
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        return self.pipeline(img)


# ── Singleton model instances ─────────────────────────────────
_text_model: Optional[_LocalTextModel] = None
_image_model: Optional[_LocalImageModel] = None
_startup_time: float = time.time()


class ModelManager:
    """
    Static interface to all models.
    Automatically routes between local and API inference.
    """

    @staticmethod
    async def load_all():
        global _text_model, _image_model, _startup_time
        _startup_time = time.time()

        if settings.inference_mode == "local":
            loop = asyncio.get_event_loop()

            _text_model = _LocalTextModel()
            _image_model = _LocalImageModel()

            # Load in thread pool to avoid blocking event loop
            await loop.run_in_executor(None, _text_model.load)
            await loop.run_in_executor(None, _image_model.load)
        else:
            log.info("Running in API mode — no local model download needed.")

    @staticmethod
    def unload_all():
        global _text_model, _image_model
        _text_model = None
        _image_model = None
        log.info("Models unloaded.")

    @staticmethod
    def status() -> dict:
        mode = settings.inference_mode
        if mode == "local":
            return {
                "text_model": _text_model.name if _text_model and _text_model.pipeline else "not_loaded",
                "image_model": _image_model.name if _image_model and _image_model.pipeline else "not_loaded",
                "code_model": _text_model.name if _text_model and _text_model.pipeline else "not_loaded",
            }
        return {
            "text_model": f"API:{settings.text_model_name}",
            "image_model": f"API:{settings.image_model_name}",
            "code_model": f"API:{settings.code_model_name}",
        }

    @staticmethod
    def uptime() -> float:
        return time.time() - _startup_time

    # ─────────────────────────────────────────────────────────
    #  TEXT PREDICTION
    # ─────────────────────────────────────────────────────────
    @staticmethod
    async def predict_text(text: str) -> dict:
        """Returns {"ai_score": int, "human_score": int, "raw_scores": [...], "model": str}"""
        truncated = text[:settings.max_text_length * 4]  # rough char limit

        if settings.inference_mode == "local" and _text_model:
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(None, _text_model.predict, truncated)
            model_name = _text_model.name
        else:
            raw, model_name = await ModelManager._hf_api_text(truncated)

        return ModelManager._parse_classification_scores(raw, model_name)

    # ─────────────────────────────────────────────────────────
    #  IMAGE PREDICTION
    # ─────────────────────────────────────────────────────────
    @staticmethod
    async def predict_image(image_bytes: bytes, mime_type: str = "image/jpeg") -> dict:
        if settings.inference_mode == "local" and _image_model:
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(None, _image_model.predict, image_bytes)
            model_name = _image_model.name
        else:
            raw, model_name = await ModelManager._hf_api_image(image_bytes, mime_type)

        return ModelManager._parse_image_scores(raw, model_name)

    # ─────────────────────────────────────────────────────────
    #  CODE PREDICTION (text model applied to code)
    # ─────────────────────────────────────────────────────────
    @staticmethod
    async def predict_code(code: str) -> dict:
        # Strip comments to focus on logic patterns
        cleaned = ModelManager._strip_code_comments(code)
        truncated = cleaned[:settings.max_text_length * 4]

        if settings.inference_mode == "local" and _text_model:
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(None, _text_model.predict, truncated)
            model_name = _text_model.name
        else:
            raw, model_name = await ModelManager._hf_api_code(truncated)

        return ModelManager._parse_classification_scores(raw, model_name)

    # ─────────────────────────────────────────────────────────
    #  INTERNAL: HuggingFace Inference API calls
    # ─────────────────────────────────────────────────────────
    @staticmethod
    async def _hf_api_text(text: str, retries: int = 3) -> tuple[list, str]:
        model = settings.text_model_name
        for attempt in range(retries):
            try:
                async with aiohttp.ClientSession() as session:
                    headers = {"Content-Type": "application/json"}
                    if settings.huggingface_token:
                        headers["Authorization"] = f"Bearer {settings.huggingface_token}"

                    async with session.post(
                        f"{HF_INFERENCE_BASE}/{model}",
                        json={"inputs": text, "options": {"wait_for_model": True}},
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=30),
                    ) as resp:
                        if resp.status == 503:
                            # Model loading — wait and retry
                            log.warning(f"HF model loading (attempt {attempt+1}), waiting 10s...")
                            await asyncio.sleep(10)
                            continue
                        if resp.status != 200:
                            # Try fallback model
                            model = settings.text_model_fallback
                            continue
                        data = await resp.json()
                        scores = data[0] if isinstance(data[0], list) else data
                        return scores, model
            except Exception as e:
                log.warning(f"HF API text attempt {attempt+1} failed: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)

        log.error("All HF text API attempts failed — returning neutral scores")
        return [{"label": "Real", "score": 0.5}, {"label": "Fake", "score": 0.5}], model

    @staticmethod
    async def _hf_api_image(image_bytes: bytes, mime_type: str) -> tuple[list, str]:
        model = settings.image_model_name
        for attempt in range(3):
            try:
                async with aiohttp.ClientSession() as session:
                    headers = {"Content-Type": mime_type}
                    if settings.huggingface_token:
                        headers["Authorization"] = f"Bearer {settings.huggingface_token}"

                    async with session.post(
                        f"{HF_INFERENCE_BASE}/{model}",
                        data=image_bytes,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=40),
                    ) as resp:
                        if resp.status == 503:
                            await asyncio.sleep(10)
                            continue
                        if resp.status != 200:
                            model = settings.image_model_fallback
                            continue
                        data = await resp.json()
                        return data, model
            except Exception as e:
                log.warning(f"HF image API attempt {attempt+1}: {e}")
                if attempt < 2:
                    await asyncio.sleep(2 ** attempt)

        return [{"label": "artificial", "score": 0.5}, {"label": "real", "score": 0.5}], model

    @staticmethod
    async def _hf_api_code(code: str) -> tuple[list, str]:
        """Same as text but uses code model."""
        model = settings.code_model_name
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Content-Type": "application/json"}
                if settings.huggingface_token:
                    headers["Authorization"] = f"Bearer {settings.huggingface_token}"
                async with session.post(
                    f"{HF_INFERENCE_BASE}/{model}",
                    json={"inputs": code, "options": {"wait_for_model": True}},
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        scores = data[0] if isinstance(data[0], list) else data
                        return scores, model
        except Exception as e:
            log.warning(f"HF code API failed: {e}")
        return [{"label": "Real", "score": 0.5}, {"label": "Fake", "score": 0.5}], model

    # ─────────────────────────────────────────────────────────
    #  SCORE PARSERS
    # ─────────────────────────────────────────────────────────
    @staticmethod
    def _parse_classification_scores(raw: list, model_name: str) -> dict:
        ai_labels = {"fake", "label_1", "ai", "generated", "machine"}
        human_labels = {"real", "label_0", "human", "original"}

        ai_score = 0.5
        human_score = 0.5

        for item in raw:
            label = str(item.get("label", "")).lower()
            score = float(item.get("score", 0.5))
            if any(k in label for k in ai_labels):
                ai_score = score
            elif any(k in label for k in human_labels):
                human_score = score

        # Normalize
        total = ai_score + human_score
        if total > 0:
            ai_score = ai_score / total
            human_score = human_score / total

        return {
            "ai_score": round(ai_score * 100),
            "human_score": round(human_score * 100),
            "raw_scores": raw,
            "model": model_name,
        }

    @staticmethod
    def _parse_image_scores(raw: list, model_name: str) -> dict:
        ai_keywords = {"artificial", "ai", "generated", "fake", "synthetic", "label_1"}
        human_keywords = {"real", "natural", "human", "authentic", "label_0", "photograph"}

        ai_score = 0.5
        human_score = 0.5

        for item in raw:
            label = str(item.get("label", "")).lower()
            score = float(item.get("score", 0.5))
            if any(k in label for k in ai_keywords):
                ai_score = score
            elif any(k in label for k in human_keywords):
                human_score = score

        total = ai_score + human_score
        if total > 0:
            ai_score = ai_score / total
            human_score = human_score / total

        return {
            "ai_score": round(ai_score * 100),
            "human_score": round(human_score * 100),
            "raw_scores": raw,
            "model": model_name,
        }

    @staticmethod
    def _strip_code_comments(code: str) -> str:
        """Remove single-line comments to help model focus on logic."""
        import re
        # Remove Python # comments
        code = re.sub(r"#.*$", "", code, flags=re.MULTILINE)
        # Remove // comments
        code = re.sub(r"//.*$", "", code, flags=re.MULTILINE)
        return code
