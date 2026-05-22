"""
AI Content Detection System — FastAPI Backend
=============================================
Runs three detection pipelines:
  • Text   → RoBERTa (roberta-base-openai-detector) + linguistic features
  • Image  → ViT / EfficientNet fine-tuned on CIFAKE/GenImage
  • Code   → RoBERTa + code-specific heuristics

External data sources fetched live:
  • Wikipedia REST API   (human ground-truth text)
  • DEV.to Articles API  (human dev articles)
  • NewsAPI.org          (journalist articles)
  • Quotable API         (human quotes)
"""
import os
import sys

# Ensure the 'back' directory is in the Python path so that absolute imports
# of sibling packages (routers, services, config) work correctly when run from
# the parent/root directory (e.g. on Render)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import logging
import time
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import json
import redis
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from routers import text, image, code, health, sources, metrics
from services.model_manager import ModelManager

load_dotenv()
# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")


# ── Lifespan: load models once at startup ──────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("🚀 Starting AI Detection Backend — loading models...")
    t0 = time.time()
    await ModelManager.load_all()
    log.info(f"✅ Models ready in {time.time()-t0:.1f}s")
    yield
    log.info("🛑 Shutting down — releasing model memory...")
    ModelManager.unload_all()


# ── App ────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Content Detection API",
    description="Detects AI-generated text, images, and code using ML models + live human data sources",
    version="3.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Redis Cache setup
redis_client = redis.Redis(host='localhost', port=6379, db=0)

# ── Middleware ─────────────────────────────────────────────────
import os
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# ── Routers ───────────────────────────────────────────────────
app.include_router(health.router,   prefix="/api/v1", tags=["Health"])
app.include_router(text.router,     prefix="/api/v1", tags=["Text Detection"])
app.include_router(image.router,    prefix="/api/v1", tags=["Image Detection"])
app.include_router(code.router,     prefix="/api/v1", tags=["Code Detection"])
app.include_router(sources.router,  prefix="/api/v1", tags=["Human Sources"])
app.include_router(metrics.router,  prefix="/api/v1", tags=["Metrics & Evaluation"])


# ── Global error handler ──────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    log.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


@app.post("/api/v1/detect/batch", tags=["Batch Detection"])
@limiter.limit("60/minute")
async def detect_batch(request: Request, texts: list[str]):
    if len(texts) > 50:
        return JSONResponse({"error": "Max 50 texts per batch"}, status_code=400)
    
    results = []
    for t in texts:
        cache_key = f"detect:{hash(t)}"
        try:
            cached = redis_client.get(cache_key)
            if cached:
                results.append(json.loads(cached))
                continue
        except redis.ConnectionError:
            pass # ignore redis if offline
            
        # mock processing fallback
        res = {"text": t[:10], "score": 0.8}
        try:
            redis_client.setex(cache_key, 3600, json.dumps(res))
        except redis.ConnectionError:
            pass
        results.append(res)
    return {"results": results}

@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "AI Content Detection API",
        "version": "3.0.0",
        "endpoints": {
            "text_detect":   "POST /api/v1/detect/text",
            "image_detect":  "POST /api/v1/detect/image",
            "code_detect":   "POST /api/v1/detect/code",
            "batch_detect":  "POST /api/v1/detect/batch",
            "fetch_sources": "GET  /api/v1/sources/human-text?topic=<topic>",
            "evaluate":      "POST /api/v1/evaluate",
            "health":        "GET  /api/v1/health",
            "docs":          "GET  /docs",
        },
    }
