"""
Central configuration — reads from environment variables / .env file.
Copy .env.example to .env and fill in your keys.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── API Keys ───────────────────────────────────────────────
    huggingface_token: str = ""         # hf_xxxxx  (required for HF Inference API)
    gemini_api_key: str = ""            # optional
    newsapi_key: str = ""               # optional
    anthropic_api_key: str = ""         # optional — for Claude enrichment
    sightengine_api_user: str = ""      # optional — for Sightengine premium image detection
    sightengine_api_secret: str = ""    # optional — for Sightengine premium image detection

    # ── Model Selection ────────────────────────────────────────
    # Text: fine-tuned RoBERTa on GPT-2 Output Dataset (500k docs)
    text_model_name: str = "Hello-SimpleAI/chatgpt-detector-roberta"
    text_model_fallback: str = "roberta-base-openai-detector"

    # Image: ViT fine-tuned on AI datasets
    image_model_name: str = "capcheck/ai-image-detection"
    image_model_fallback: str = "Smogy/SMOGY-Ai-images-detector"

    # Code: RoBERTa (works well for code authorship)
    code_model_name: str = "roberta-base-openai-detector"

    # ── Inference Mode ─────────────────────────────────────────
    # "local"  → download & run models locally (GPU/CPU)
    # "api"    → use Hugging Face Inference API (cloud, needs HF token)
    inference_mode: str = "local"         # "local" | "api"

    # ── Local Model Settings ───────────────────────────────────
    device: str = "auto"                # "cpu" | "cuda" | "mps" | "auto"
    max_text_length: int = 512          # tokens (RoBERTa max is 512)
    batch_size: int = 1

    # ── External Data Sources ──────────────────────────────────
    wikipedia_api: str = "https://en.wikipedia.org/api/rest_v1"
    wikipedia_action_api: str = "https://en.wikipedia.org/w/api.php"
    devto_api: str = "https://dev.to/api"
    quotable_api: str = "https://api.quotable.io"
    newsapi_base: str = "https://newsapi.org/v2"

    # ── Server ────────────────────────────────────────────────
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    log_level: str = "info"

    # ── Cache ─────────────────────────────────────────────────
    cache_ttl_seconds: int = 300        # cache human-source fetches for 5 min
    max_cache_entries: int = 100

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
