"""
HumanSourceFetcher
==================
Fetches real human-written text from multiple free APIs:

  1. Wikipedia REST API     — encyclopedic articles (peer-reviewed)
  2. Wikipedia Action API   — full article intro text
  3. DEV.to Articles API    — developer blog posts
  4. Quotable.io API        — verified human quotes
  5. NewsAPI.org            — journalist-written news (needs free key)

Results are cached per topic to avoid redundant network calls.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import aiohttp

from config import get_settings
from models.schemas import HumanSample

log = logging.getLogger("source_fetcher")
settings = get_settings()

# ── Simple in-memory cache ─────────────────────────────────────
_cache: dict[str, tuple[float, list[HumanSample]]] = {}


def _cached(topic: str) -> Optional[list[HumanSample]]:
    if topic in _cache:
        ts, data = _cache[topic]
        if time.time() - ts < settings.cache_ttl_seconds:
            return data
    return None


def _store(topic: str, data: list[HumanSample]):
    # Evict oldest entry if cache is full
    if len(_cache) >= settings.max_cache_entries:
        oldest = min(_cache, key=lambda k: _cache[k][0])
        del _cache[oldest]
    _cache[topic] = (time.time(), data)


# ── Individual source fetchers ─────────────────────────────────

async def _fetch_wikipedia_summary(session: aiohttp.ClientSession, topic: str) -> Optional[HumanSample]:
    """Wikipedia REST API — article summary (no auth needed)."""
    try:
        url = f"{settings.wikipedia_api}/page/summary/{topic.replace(' ', '_')}"
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as r:
            if r.status == 200:
                data = await r.json()
                extract = data.get("extract", "")
                if len(extract) > 100:
                    return HumanSample(
                        source="Wikipedia",
                        url=data.get("content_urls", {}).get("desktop", {}).get("page", "#"),
                        text=extract,
                        title=data.get("title"),
                        sample_type="encyclopedia",
                        icon="📖",
                    )
    except Exception as e:
        log.debug(f"Wikipedia summary failed for '{topic}': {e}")
    return None


async def _fetch_wikipedia_full(session: aiohttp.ClientSession, topic: str) -> Optional[HumanSample]:
    """Wikipedia Action API — full intro text (no auth needed)."""
    try:
        params = {
            "action": "query",
            "prop": "extracts",
            "exintro": "true",
            "explaintext": "true",
            "titles": topic,
            "format": "json",
            "origin": "*",
        }
        async with session.get(
            settings.wikipedia_action_api,
            params=params,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as r:
            if r.status == 200:
                data = await r.json()
                pages = data.get("query", {}).get("pages", {})
                page = next(iter(pages.values()), {})
                text = page.get("extract", "")
                if text and len(text) > 200:
                    return HumanSample(
                        source="Wikipedia (Full)",
                        url=f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}",
                        text=text[:2000],       # cap at 2000 chars
                        title=page.get("title"),
                        sample_type="encyclopedia",
                        icon="📖",
                    )
    except Exception as e:
        log.debug(f"Wikipedia full failed for '{topic}': {e}")
    return None


async def _fetch_devto(session: aiohttp.ClientSession, topic: str) -> list[HumanSample]:
    """DEV.to public API — real developer articles (no auth needed)."""
    results = []
    try:
        params = {"tag": topic.lower(), "per_page": 5, "top": "1"}
        async with session.get(
            f"{settings.devto_api}/articles",
            params=params,
            timeout=aiohttp.ClientTimeout(total=8),
        ) as r:
            if r.status == 200:
                articles = await r.json()
                for a in articles[:3]:
                    desc = a.get("description", "") or ""
                    body = a.get("body_markdown", "") or ""
                    content = desc if len(desc) > 80 else body[:500]
                    if len(content) > 80:
                        results.append(HumanSample(
                            source="DEV.to",
                            url=a.get("url", "#"),
                            text=content[:800],
                            title=a.get("title"),
                            author=a.get("user", {}).get("name"),
                            sample_type="blog",
                            icon="💻",
                        ))
    except Exception as e:
        log.debug(f"DEV.to fetch failed for '{topic}': {e}")
    return results


async def _fetch_quotable(session: aiohttp.ClientSession) -> Optional[HumanSample]:
    """Quotable.io — verified human quotes (no auth needed)."""
    try:
        params = {"limit": 8, "minLength": 50}
        async with session.get(
            f"{settings.quotable_api}/quotes/random",
            params=params,
            timeout=aiohttp.ClientTimeout(total=6),
        ) as r:
            if r.status == 200:
                quotes = await r.json()
                combined = " ".join(
                    f'"{q["content"]}" — {q["author"]}' for q in quotes
                )
                return HumanSample(
                    source="Quotable.io",
                    url="https://quotable.io",
                    text=combined,
                    sample_type="quotes",
                    icon="💬",
                )
    except Exception as e:
        log.debug(f"Quotable fetch failed: {e}")
    return None


async def _fetch_newsapi(session: aiohttp.ClientSession, topic: str) -> list[HumanSample]:
    """NewsAPI.org — journalist articles (free key required)."""
    if not settings.newsapi_key:
        return []
    results = []
    try:
        params = {
            "q": topic,
            "language": "en",
            "sortBy": "relevancy",
            "pageSize": 5,
            "apiKey": settings.newsapi_key,
        }
        async with session.get(
            f"{settings.newsapi_base}/everything",
            params=params,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as r:
            if r.status == 200:
                data = await r.json()
                for a in (data.get("articles") or [])[:3]:
                    desc = a.get("description") or ""
                    content = a.get("content") or ""
                    text = (desc + " " + content[:400]).strip()
                    if len(text) > 80:
                        results.append(HumanSample(
                            source="NewsAPI",
                            url=a.get("url", "#"),
                            text=text[:900],
                            title=a.get("title"),
                            author=a.get("author"),
                            sample_type="news",
                            icon="📰",
                        ))
    except Exception as e:
        log.debug(f"NewsAPI fetch failed for '{topic}': {e}")
    return results


# ── Main public function ───────────────────────────────────────

async def fetch_human_samples(topic: str) -> list[HumanSample]:
    """
    Fetch human-written text samples from all configured sources.
    Results are cached per topic for `cache_ttl_seconds`.
    """
    cached = _cached(topic)
    if cached:
        log.info(f"Cache hit for topic '{topic}' ({len(cached)} samples)")
        return cached

    log.info(f"Fetching live human samples for topic: '{topic}'")
    t0 = time.time()

    connector = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Run all fetchers concurrently
        tasks = [
            _fetch_wikipedia_summary(session, topic),
            _fetch_wikipedia_full(session, topic),
            _fetch_devto(session, topic),
            _fetch_quotable(session),
            _fetch_newsapi(session, topic),
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    samples: list[HumanSample] = []
    for r in raw_results:
        if isinstance(r, Exception):
            log.debug(f"Source exception: {r}")
        elif isinstance(r, list):
            samples.extend(r)
        elif r is not None:
            samples.append(r)

    log.info(f"Fetched {len(samples)} human samples in {(time.time()-t0)*1000:.0f}ms")
    _store(topic, samples)
    return samples


def compute_vocabulary_similarity(text_a: str, text_b: str) -> int:
    """
    Jaccard similarity of significant words between two texts.
    Returns integer 0–100.
    """
    import re
    def tokenize(t: str) -> set[str]:
        return set(w.lower() for w in re.findall(r"\b[a-z]{4,}\b", t.lower()))

    set_a = tokenize(text_a)
    set_b = tokenize(text_b)
    if not set_a or not set_b:
        return 0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return round((intersection / union) * 100) if union else 0
