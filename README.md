# Fake News Detection

A browser extension that analyses news articles for credibility using a DistilBERT classifier, source reputation lookup, linguistic analysis, and cross-source comparison.

## Architecture

```
chrome-plugin/   — React/TypeScript Chrome extension (Vite)
backend/         — FastAPI Python backend
```

---

## Prerequisites

| Tool | Minimum version |
|---|---|
| Python | 3.11 |
| Node.js | 18 |
| npm | 9 |

---

## Backend

### 1. Create and activate a virtual environment

```bash
cd backend
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Configure environment variables

Create `backend/.env` (copy the template below):

```env
GOOGLE_FACTCHECK_API_KEY=your_key_here
NEWS_API_KEY=your_key_here
HF_TOKEN=your_huggingface_token_here
```

- `GOOGLE_FACTCHECK_API_KEY` — [Google Fact Check Tools API](https://developers.google.com/fact-check/tools/api)
- `NEWS_API_KEY` — [NewsAPI](https://newsapi.org/)
- `HF_TOKEN` — [Hugging Face token](https://huggingface.co/settings/tokens) (needed to download the model on first run)

### 4. Start the server

```bash
uvicorn main:app --reload --port 7860
```

The API will be available at `http://localhost:7860`.  
Interactive docs: `http://localhost:7860/docs`

> On first startup the DistilBERT model is downloaded from Hugging Face Hub (~250 MB) and the NLI model (`cross-encoder/nli-deberta-v3-small`) is loaded. This may take a few minutes.

---

## Chrome Extension

### 1. Install dependencies

```bash
cd chrome-plugin
npm install
```

### 2. Point the extension at your local backend

Open [chrome-plugin/src/Api.tsx](chrome-plugin/src/Api.tsx) and change `BASE_URL`:

```ts
// from
const BASE_URL = "https://Artemi0-fake-news-detector.hf.space";

// to
const BASE_URL = "http://localhost:7860";
```

### 3. Build the extension

```bash
npm run build
```

This produces the `chrome-plugin/dist/` folder.

### 4. Load in Chrome

1. Open `chrome://extensions`
2. Enable **Developer mode** (top-right toggle)
3. Click **Load unpacked**
4. Select the `chrome-plugin/dist/` folder

The extension icon will appear in the toolbar. Navigate to any news article and click it to run an analysis.

---

## Running both services together

```bash
# Terminal 1 — backend
cd backend && .venv\Scripts\activate && uvicorn main:app --reload --port 7860

# Terminal 2 — extension (rebuild after code changes)
cd chrome-plugin && npm run build
```

---

## Docker (backend only)

```bash
cd backend
docker build -t fake-news-backend .
docker run -p 7860:7860 --env-file .env fake-news-backend
```
