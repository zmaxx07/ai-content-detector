"""
Sources Router
GET /api/v1/sources/human-text?topic=<topic>
Returns live human-written text from Wikipedia, DEV.to, NewsAPI, Quotable.
"""
import time
from fastapi import APIRouter, Query
from models.schemas import SourcesResponse
from services.source_fetcher import fetch_human_samples

router = APIRouter()

@router.get("/sources/human-text", response_model=SourcesResponse, summary="Fetch live human text samples")
async def get_human_sources(topic: str = Query("technology", min_length=2, max_length=100)):
    t0 = time.time()
    samples = await fetch_human_samples(topic)
    sources_used = list({s.source for s in samples})
    return SourcesResponse(
        topic=topic,
        samples=samples,
        total_fetched=len(samples),
        sources_used=sources_used,
        fetch_time_ms=round((time.time() - t0) * 1000),
    )
