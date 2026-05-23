# AI Content Detection System — Full-Stack Project
AI-Based Detection of AI-Generated Content

Detects AI-generated **Text**, **Images**, and **Code** using:
- 🤗 **HuggingFace ML Models** (RoBERTa, SDXL-Detector) via Inference API
- 🌐 **Live Human Reference Data** from Wikipedia, DEV.to, NewsAPI, Quotable
- 📊 **15+ Linguistic Features** extracted and analyzed  
- ⚡ **FastAPI Backend** + ⚛️ **React Frontend**

---

## Project Structure

```
ai_detector_project/
├── backend/                    ← Python FastAPI server
│   ├── back/
│   │   ├── main.py             ← FastAPI app entry point
│   │   ├── config.py           ← Settings (reads .env)
│   │   ├── models/schemas.py   ← Pydantic request/response types
│   │   ├── routers/
│   │   │   ├── text.py         ← POST /api/v1/detect/text
│   │   │   ├── image.py        ← POST /api/v1/detect/image
│   │   │   ├── code.py         ← POST /api/v1/detect/code
│   │   │   ├── sources.py      ← GET  /api/v1/sources/human-text
│   │   │   └── health.py       ← GET  /api/v1/health
│   │   └── services/
│   │       ├── model_manager.py      ← ML inference (HF API or local)
│   │       ├── source_fetcher.py     ← Wikipedia/DEV.to/NewsAPI fetcher
│   │       └── linguistic_analyzer.py← 15+ feature extractor
│   ├── tests/test_api.py       ← pytest test suite
│   ├── run.py                  ← Server startup script
│   ├── requirements.txt        ← Python dependencies
│   └── .env.example            ← Environment config template
│
├── frontend/                   ← React web app
│   ├── public/index.html
│   ├── src/
│   │   ├── App.jsx             ← Root component + tab nav + history
│   │   ├── index.js            ← React entry point
│   │   ├── index.css           ← Global styles
│   │   ├── services/api.js     ← All backend API calls
│   │   └── components/
│   │       ├── TextDetector.jsx  ← Text analysis UI
│   │       ├── ImageDetector.jsx ← Image upload + analysis UI
│   │       ├── CodeDetector.jsx  ← Code analysis UI
│   │       ├── History.jsx       ← Detection history log
│   │       └── ui.jsx            ← Shared design system components
│   ├── package.json
│   └── .env                    ← REACT_APP_API_URL=http://localhost:8000/api/v1
│
├── start.sh                    ← Start both backend + frontend (Linux/Mac)
├── start.bat                   ← Start both backend + frontend (Windows)
└── README.md                   ← This file
```

---

## Quickstart — 5 Steps

### Step 1 — Get API Keys (free)

| Key | Where | Required? |
|-----|-------|-----------|
| HuggingFace token | https://huggingface.co/settings/tokens | ✅ Yes |
| NewsAPI key | https://newsapi.org/register | Optional |

---

### Step 2 — Setup Backend

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env — add your HUGGINGFACE_TOKEN
```

---

### Step 3 — Setup Frontend

```bash
cd frontend
npm install
```

---

### Step 4 — Run Everything

**Option A: One command (Linux/Mac)**
```bash
chmod +x start.sh && ./start.sh
```

**Option B: One command (Windows)**
```bat
start.bat
```

**Option C: Manually (two terminals)**

Terminal 1 — Backend:
```bash
cd backend
source venv/bin/activate
python run.py --mode api
```

Terminal 2 — Frontend:
```bash
cd frontend
npm start
```

**Option D: Docker Compose (All-in-one)**
Build and run the Backend, React Frontend, and Redis Cache simultaneously in containers:
```bash
docker-compose up --build
```

---

### Step 5 — Open in Browser

| URL | What |
|-----|------|
| http://localhost:3000 | React Frontend (Option A, B, C) |
| http://localhost:80 (or http://localhost) | React Frontend (Option D - Docker Compose) |
| http://localhost:8000/docs | FastAPI Swagger API Docs |
| http://localhost:8000/api/v1/health | Health check JSON |

---

## Detection Pipeline

```
User Input (text / image / code)
         │
         ▼
[1] FastAPI receives request
         │
         ▼
[2] ML Model via HuggingFace Inference API
    • Text/Code → roberta-base-openai-detector (500k training docs)
    • Image     → Organika/sdxl-detector (CIFAKE + GenImage dataset)
         │
         ▼
[3] Live Human Reference Fetch (concurrent)
    • Wikipedia REST API   — encyclopedic articles
    • DEV.to Articles API  — developer blog posts
    • Quotable.io API      — verified human quotes
    • NewsAPI.org          — journalist articles
         │
         ▼
[4] Linguistic Feature Extraction (text/code only)
    • 30+ AI phrase markers ("delve into", "furthermore", etc.)
    • Sentence length variance
    • Lexical diversity (Type-Token Ratio)
    • Human informal signals (contractions, slang)
    • Code: comment density, identifier length, docstrings, TODOs
         │
         ▼
[5] Score Blending
    final_ai_score = ML_score + linguistic_adjustment (−20 to +20%)
         │
         ▼
[6] Verdict + Confidence + Signals → React Frontend
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/detect/text` | Detect AI-generated text |
| POST | `/api/v1/detect/image` | Detect AI-generated image (multipart) |
| POST | `/api/v1/detect/code` | Detect AI-generated code |
| GET | `/api/v1/sources/human-text?topic=X` | Fetch live human references |
| GET | `/api/v1/health` | Backend health + model status |

---

## Models & Datasets

| Task | Model | Dataset | Size |
|------|-------|---------|------|
| Text | `roberta-base-openai-detector` | GPT-2 Output Dataset | 500k docs |
| Text (fallback) | `Hello-SimpleAI/chatgpt-detector-roberta` | AI Text Detection Pile | 130k docs |
| Image | `Organika/sdxl-detector` | CIFAKE + GenImage | 1.2M images |
| Image (fallback) | `umm-maybe/AI-image-detector` | Mixed AI/real | 60k images |
| Code | `roberta-base-openai-detector` | CodeSearchNet + AI-CodeBench | 2M snippets |

---

## External Human Data Sources

| Source | URL | Key Needed | Provides |
|--------|-----|-----------|---------|
| Wikipedia | en.wikipedia.org/api/rest_v1 | ❌ Free | Encyclopedic articles |
| DEV.to | dev.to/api/articles | ❌ Free | Developer blog posts |
| Quotable.io | api.quotable.io | ❌ Free | Human quotes |
| NewsAPI.org | newsapi.org/v2 | ✅ Free | News articles |

---

## Tech Stack

**Backend:** Python 3.12 · FastAPI · Uvicorn · Pydantic · aiohttp · Transformers · PyTorch · Pillow

**Frontend:** React 18 · JavaScript (ES2022) · CSS-in-JS · Fetch API

**ML:** HuggingFace Transformers · RoBERTa · ViT · HuggingFace Inference API

**Data:** Wikipedia REST API · DEV.to API · NewsAPI · Quotable API
