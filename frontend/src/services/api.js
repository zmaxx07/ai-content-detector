// src/services/api.js
// All API calls go through this file to the FastAPI backend

const BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, options);
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || data.error || `HTTP ${res.status}`);
  return data;
}

// ── Text Detection ─────────────────────────────────────────────
export async function detectText(text, topic = 'technology', fetchRefs = true) {
  return request('/detect/text', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, topic, fetch_human_references: fetchRefs }),
  });
}

// ── Image Detection ────────────────────────────────────────────
export async function detectImage(file) {
  const fd = new FormData();
  fd.append('file', file);
  return request('/detect/image', { method: 'POST', body: fd });
}

// ── Code Detection ─────────────────────────────────────────────
export async function detectCode(code, language = null) {
  return request('/detect/code', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ code, language }),
  });
}

// ── Live Human Sources ─────────────────────────────────────────
export async function fetchHumanSources(topic) {
  return request(`/sources/human-text?topic=${encodeURIComponent(topic)}`);
}

// ── Health Check ───────────────────────────────────────────────
export async function checkHealth() {
  return request('/health');
}
