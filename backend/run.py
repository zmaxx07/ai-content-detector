#!/usr/bin/env python3
"""
run.py — Start the AI Detection Backend server
Usage: python run.py [--mode api|local] [--port 8000] [--reload]
"""

import argparse
import logging
import os
import sys

import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("run")


def main():
    parser = argparse.ArgumentParser(description="AI Content Detection API Server")
    parser.add_argument("--mode",   choices=["api", "local"], default=None, help="Inference mode")
    parser.add_argument("--port",   type=int, default=None)
    parser.add_argument("--host",   default=None)
    parser.add_argument("--reload", action="store_true", help="Hot-reload on code change (dev only)")
    args = parser.parse_args()

    # CLI args override .env
    if args.mode:
        os.environ["INFERENCE_MODE"] = args.mode
    if args.port:
        os.environ["PORT"] = str(args.port)
    if args.host:
        os.environ["HOST"] = args.host

    from app.config import get_settings
    cfg = get_settings()

    log.info("=" * 60)
    log.info("  AI Content Detection Backend  v3.0.0")
    log.info("=" * 60)
    log.info(f"  Inference mode : {cfg.inference_mode.upper()}")
    log.info(f"  Text model     : {cfg.text_model_name}")
    log.info(f"  Image model    : {cfg.image_model_name}")
    log.info(f"  HF token       : {'✅ set' if cfg.huggingface_token else '❌ missing — add to .env'}")
    log.info(f"  NewsAPI key    : {'✅ set' if cfg.newsapi_key else '⚠  not set (optional)'}")
    log.info(f"  Server         : http://{cfg.host}:{cfg.port}")
    log.info(f"  Docs           : http://{cfg.host}:{cfg.port}/docs")
    log.info("=" * 60)

    if not cfg.huggingface_token and cfg.inference_mode == "api":
        log.warning("⚠  HUGGINGFACE_TOKEN not set — HF Inference API calls will be rate-limited or fail")
        log.warning("   Get a free token at: https://huggingface.co/settings/tokens")
        log.warning("   Then add it to .env: HUGGINGFACE_TOKEN=hf_xxxx")

    uvicorn.run(
        "app.main:app",
        host=cfg.host,
        port=cfg.port,
        log_level=cfg.log_level,
        reload=args.reload,
        workers=1,    # keep 1 worker so model loads once
    )


if __name__ == "__main__":
    main()
