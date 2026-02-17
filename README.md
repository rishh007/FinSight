
# FinSight — Financial Analyst 💰📈

FinSight is a lightweight financial analytics + conversational assistant toolkit. It combines data extraction, charting, SEC/filing parsing, news ingestion, and a WebSocket-based chat UI so users can ask for charts, filings, news summaries, and concise analyst reports.

Deepwiki - https://deepwiki.com/rishh007/FinSight

---

## Quick repo layout

```
.
├── main_backend.py                  # FastAPI backend + WebSocket endpoint (serves UI + agent)
├── requirements.txt         # Python dependencies
├── state.py                 # FinanceAgentState (shared session state)
├── workflows/               # Tool / node implementations (entity extraction, data fetchers, etc.)
│   ├── extract_entities.py
│   ├── get_stock_data_and_chart.py
│   ├── get_financial_news.py
│   ├── get_sec_filing_section.py
│   └── curate_report.py
├── static/                  # Frontend files served by FastAPI (index.html, css, js)
│   └── index.html
├── charts/                  # Generated charts (served as static assets)
├── reports/                 # Generated reports / docx / pdf
├── main.ipynb               # Notebook for prototyping and experiments
└── TODO
```

---

## Features

* Conversational Web UI (WebSocket) — ask for charts, news, filings, or request reports.
* Workflow-based backend: each intent maps to a small node (extract entities → classify intent → route → tool).
* Chart generation, basic metrics extraction, SEC filing section extraction, news scraping, and report curation.
* Session-based state management for multi-turn conversations.

---

## Quickstart (dev)

1. Clone project:

```bash
git clone https://github.com/rishh007/FinSight.git
cd FinSight
```

2. Create & activate a venv:

```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

3. Install deps:

```bash
pip install -r requirements.txt
```

4. Run the server (development):

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 5500
```

5. Open the UI:

```
http://localhost:5500/
```

---

## Production / Cloud deployment notes

Recommended files to add for deployment:

* `Dockerfile` — container image (example below)
* `requirements.txt` — already present
* `Procfile` (for Heroku-like) — e.g. `web: uvicorn main:app --host 0.0.0.0 --port $PORT`
* `.env` or platform secrets — store API keys and model endpoints (DO NOT commit)
* `nginx.conf` (optional) — for reverse proxy in front of uvicorn for static caching and TLS termination

### Example minimal Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY . /app

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose the port the app listens on (match uvicorn command)
EXPOSE 5500

# Use uvicorn with a single worker for small apps (scale with process manager)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5500", "--workers", "1"]
```

### Example Procfile

```
web: uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

## Which files to consider / include when preparing for a web/cloud application

When you package or deploy FinSight for the web, prioritize and review these files:

1. **`main_backend.py`** — *Essential.* This is the FastAPI app that:

   * mounts `static/` or serves `index.html`,
   * exposes `/ws/{session_id}` WebSocket endpoint,
   * contains session management and routes the agent workflow.
   * *Action:* Harden error handling, add logging, ensure environment-driven config (port, model endpoints).

2. **`requirements.txt`** — *Essential.* All Python deps for the cloud image.

3. **`state.py`** — *Essential.* Ensure session format is serializable if you plan to persist sessions (Redis, DB).

   * *Action:* If you will scale across multiple processes, move sessions to shared store (Redis) and avoid in-memory `sessions` dict.

4. **`workflows/*.py`** — *Essential.* All tool nodes — review and ensure:

   * No blocking long-running sync calls (wrap heavy I/O in `asyncio.to_thread` or use async libs).
   * No hard-coded file paths; use configurable paths (env var for `charts/`, `reports/`).

5. **`static/index.html` + static assets** — *Essential for UI.* Ensure JS WebSocket URL is environment-aware (use relative ws path or read `window.location`).

6. **`charts/` and `reports/` directories** — *Include if you serve generated artifacts.*

   * Consider writing generated artifacts to a cloud storage (S3/GCS) for persistence and scaling.

7. **`main.ipynb & main.py`** — *Optional.* Useful for demos but not required in production.

8. **`TODO` and docs** — *Optional.* Keep repo housekeeping.

9. **Secrets/config** (not in repo) — *CRITICAL.* Provide values for:

   * LLM endpoint/config (Ollama or other),
   * any news or SEC API keys,
   * storage credentials (S3) if used.

---

## Operational suggestions before deploying

* **Session persistence & scaling**

  * Replace in-memory `sessions` dict with Redis backed session store if deploying multiple workers/replicas.
  * Use sticky sessions at load balancer or centralized store.

* **Long-running tasks**

  * Offload heavy fetches (SEC parsing / long RAG retrieval) to background workers (Celery, RQ, or FastAPI background tasks) and stream partial updates via WebSocket events.

* **Security**

  * Do not commit `.env`. Use platform secret managers or environment variables.
  * Add rate limits to WebSocket/API endpoints.
  * Validate user input for path traversal (when saving/loading files).

* **Observability**

  * Add structured logging (JSON logs), error reporting (Sentry), and metrics (Prometheus).

* **Static assets & caching**

  * Serve large charts/reports from object storage (S3) and give UI signed URLs or direct links.

---

## Contact / Contributing

Contributions welcome — open issues or PRs. For deployment help or a production-ready template (Docker + Redis + background worker example), I can produce those files next.


* generate a ready-to-run `Dockerfile`, `Procfile`, and `.env.example` for FinSight, **or**
* create a small patch that moves `sessions` to Redis and adds background task support?
