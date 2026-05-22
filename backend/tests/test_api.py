"""
Tests for AI Content Detection API
Run with: pytest tests/ -v
"""

import pytest
from fastapi.testclient import TestClient

# ── Sample texts ───────────────────────────────────────────────
HUMAN_TEXT = """
I've been struggling with this bug for three days now and honestly I'm losing my mind.
The stack trace doesn't make any sense — it's like the library is just randomly dropping
connections, no warning, no error, just gone. Tried everything I could think of:
restarting the service, increasing timeouts, even rewrote the connection pool from scratch.
Still happens. Going to sleep on it and see if I spot something tomorrow. Sometimes you
just need to step away, you know? Anyway, if anyone's seen something like this with
aiohttp and Redis, please reach out. Would genuinely appreciate it.
"""

AI_TEXT = """
In today's rapidly evolving technological landscape, it is important to note that
artificial intelligence plays a pivotal role in transforming various industries.
Furthermore, the state-of-the-art machine learning algorithms are leveraging
cutting-edge approaches to solve complex problems. It is worth noting that these
advancements have far-reaching implications for businesses and individuals alike.
In conclusion, the paradigm shift brought about by AI technologies represents
a testament to human ingenuity and innovation. Delving into these developments
further reveals the remarkable potential that lies ahead.
"""

HUMAN_CODE = """
# ugh this is a mess but it works for now
# TODO: clean this up before PR review
def get_user(uid):
    u = db.query(f"SELECT * FROM users WHERE id={uid}")
    if not u:
        return None  # just return None, caller handles it
    return u[0]  # first row only
"""

AI_CODE = '''
def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieves a user from the database by their unique identifier.
    
    Args:
        user_id: The unique identifier of the user to retrieve.
        
    Returns:
        A dictionary containing the user data if found, None otherwise.
        
    Raises:
        DatabaseConnectionError: If the database connection fails.
        ValueError: If the user_id is invalid.
    """
    try:
        if not isinstance(user_id, int) or user_id <= 0:
            raise ValueError(f"Invalid user_id: {user_id}. Must be a positive integer.")
        
        query = "SELECT * FROM users WHERE id = %s"
        result = db.execute_query(query, (user_id,))
        
        if result and len(result) > 0:
            return result[0]
        else:
            return None
            
    except DatabaseConnectionError as e:
        logger.error(f"Database connection failed while retrieving user {user_id}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error retrieving user {user_id}: {e}")
        raise
'''


# ── Client fixture ─────────────────────────────────────────────
@pytest.fixture(scope="module")
def client():
    """Create test client — skips model loading for speed."""
    import os
    os.environ["INFERENCE_MODE"] = "api"
    os.environ["HUGGINGFACE_TOKEN"] = ""  # tests use mocked responses

    from back.main import app
    from unittest.mock import patch, AsyncMock

    mock_ml = AsyncMock(return_value={
        "ai_score": 72, "human_score": 28,
        "raw_scores": [{"label": "Fake", "score": 0.72}, {"label": "Real", "score": 0.28}],
        "model": "roberta-base-openai-detector",
    })
    mock_sources = AsyncMock(return_value=[])

    with patch("back.services.model_manager.ModelManager.load_all", AsyncMock()), \
         patch("back.services.model_manager.ModelManager.predict_text", mock_ml), \
         patch("back.services.model_manager.ModelManager.predict_code", mock_ml), \
         patch("back.services.source_fetcher.fetch_human_samples", mock_sources):
        with TestClient(app) as c:
            yield c


# ── Health ─────────────────────────────────────────────────────
def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "models_loaded" in data


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "endpoints" in r.json()


# ── Text detection ─────────────────────────────────────────────
def test_text_detect_returns_verdict(client):
    r = client.post("/api/v1/detect/text", json={"text": AI_TEXT, "topic": "technology"})
    assert r.status_code == 200
    data = r.json()
    assert data["verdict"] in ("AI-Generated", "Likely AI", "Likely Human", "Human-Written")
    assert 0 <= data["confidence"] <= 100
    assert "signals" in data
    assert "linguistic_features" in data


def test_text_detect_human_text(client):
    r = client.post("/api/v1/detect/text", json={"text": HUMAN_TEXT, "topic": "debugging"})
    assert r.status_code == 200
    data = r.json()
    # Human text should score lower on AI probability (after adjustment)
    assert data["breakdown"]["ai_score"] <= 85


def test_text_detect_too_short(client):
    r = client.post("/api/v1/detect/text", json={"text": "too short", "topic": "tech"})
    assert r.status_code == 422  # validation error


def test_text_detect_linguistic_features(client):
    r = client.post("/api/v1/detect/text", json={"text": AI_TEXT, "topic": "AI"})
    assert r.status_code == 200
    feats = r.json()["linguistic_features"]
    assert feats["word_count"] > 0
    assert feats["lexical_diversity"] >= 0
    assert isinstance(feats["ai_phrases_found"], list)


def test_text_detect_ai_phrases_detected(client):
    """AI text above contains known AI phrases — should be detected."""
    r = client.post("/api/v1/detect/text", json={"text": AI_TEXT, "topic": "AI"})
    assert r.status_code == 200
    feats = r.json()["linguistic_features"]
    # AI_TEXT contains phrases like "delve into", "state-of-the-art", etc.
    assert len(feats["ai_phrases_found"]) > 0


# ── Code detection ─────────────────────────────────────────────
def test_code_detect_ai_code(client):
    r = client.post("/api/v1/detect/code", json={"code": AI_CODE})
    assert r.status_code == 200
    data = r.json()
    assert data["verdict"] in ("AI-Generated", "Likely AI", "Likely Human", "Human-Written")
    assert "code_features" in data


def test_code_detect_human_code(client):
    r = client.post("/api/v1/detect/code", json={"code": HUMAN_CODE})
    assert r.status_code == 200
    data = r.json()
    feats = data["code_features"]
    # Human code has human markers (TODO)
    assert feats["human_code_markers"] >= 1


def test_code_detect_language(client):
    python_code = "# python code\ndef hello():\n    print('world')\n\nhello()"
    r = client.post("/api/v1/detect/code", json={"code": python_code})
    assert r.status_code == 200
    data = r.json()
    assert data["detected_language"] == "python"


def test_code_detect_too_short(client):
    r = client.post("/api/v1/detect/code", json={"code": "x=1"})
    assert r.status_code == 422


# ── Sources ────────────────────────────────────────────────────
def test_sources_endpoint(client):
    r = client.get("/api/v1/sources/human-text?topic=technology")
    assert r.status_code == 200
    data = r.json()
    assert data["topic"] == "technology"
    assert "samples" in data


# ── Docs ───────────────────────────────────────────────────────
def test_docs_available(client):
    r = client.get("/docs")
    assert r.status_code == 200
