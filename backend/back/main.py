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

import logging
import time
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from routers import text, image, code, health, sources
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

# ── Middleware ─────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # lock down in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ── Routers ───────────────────────────────────────────────────
app.include_router(health.router,   prefix="/api/v1", tags=["Health"])
app.include_router(text.router,     prefix="/api/v1", tags=["Text Detection"])
app.include_router(image.router,    prefix="/api/v1", tags=["Image Detection"])
app.include_router(code.router,     prefix="/api/v1", tags=["Code Detection"])
app.include_router(sources.router,  prefix="/api/v1", tags=["Human Sources"])


# ── Global error handler ──────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    log.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "AI Content Detection API",
        "version": "3.0.0",
        "endpoints": {
            "text_detect":   "POST /api/v1/detect/text",
            "image_detect":  "POST /api/v1/detect/image",
            "code_detect":   "POST /api/v1/detect/code",
            "fetch_sources": "GET  /api/v1/sources/human-text?topic=<topic>",
            "health":        "GET  /api/v1/health",
            "docs":          "GET  /docs",
        },
    }
