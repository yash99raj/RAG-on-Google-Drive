# highwatch-rag

RAG pipeline over Google Drive — hybrid BM25 + dense retrieval, incremental sync, cited answers via Claude or Gemma.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  INDEXING PIPELINE                                              │
│                                                                 │
│  Google Drive ──► Connector ──► Extract/Clean ──► Chunker      │
│      (PDF, DOCX,    (service        (pypdf,         (recursive  │
│       GDoc, txt)     account)        docx)           splitter)  │
│                                          │                      │
│                                          ▼                      │
│                                     Embedder                    │
│                                  (BGE / MiniLM)                 │
│                                          │                      │
│                                          ▼                      │
│                                    OpenSearch                   │
│                               (BM25 text + knn_vector)          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  QUERY PIPELINE                                                 │
│                                                                 │
│  User Query ──► POST /ask ──► Hybrid Retriever                  │
│                                    │                            │
│                         ┌──────────┴──────────┐                │
│                         ▼                     ▼                 │
│                    BM25 Search          Dense Search            │
│                    (match text)         (knn vector)            │
│                         │                     │                 │
│                         └──────────┬──────────┘                │
│                                    ▼                            │
│                              RRF Fusion                         │
│                          (top-k chunk IDs)                      │
│                                    │                            │
│                                    ▼                            │
│                            LLM (Claude / Gemma)                 │
│                                    │                            │
│                                    ▼                            │
│                         Answer + Cited Sources                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Features

- **Incremental sync** — uses Drive Changes API with page tokens; only re-indexes files whose content changed
- **Hybrid BM25 + dense retrieval** — keyword and semantic search fused via Reciprocal Rank Fusion (RRF) for best recall
- **Content-hash dedup** — sha256 of normalized text stored in OpenSearch; unchanged documents skipped entirely
- **Dual LLM backend** — Anthropic Claude (async) and Google Gemma via `google.genai`; swap with one env var
- **Cited answers** — every response includes source file name, chunk ID, score, and a 200-char snippet per source

---

## Setup

### Prerequisites

- Python 3.11+
- Docker + Docker Compose
- GCP service account JSON with `https://www.googleapis.com/auth/drive.readonly` scope
- Drive folder shared with the service account email

### Installation

```bash
git clone <repo-url>
cd RAG-on-Google-Drive

# Configure environment
cp .env.example .env
# Edit .env — fill in ANTHROPIC_API_KEY or GOOGLE_API_KEY, GDRIVE_FOLDER_ID
# Place your GCP service account file at the path set in GDRIVE_CREDENTIALS_PATH

# Start OpenSearch
docker compose up -d opensearch

# Install Python dependencies
pip install -e .

# Start API server
uvicorn src.api.main:create_app --factory --port 8000
```

On startup the server auto-creates the `highwatch_chunks_v1` and `highwatch_sync_state_v1` OpenSearch indices if they don't exist.

---

## API Usage

### Sync Google Drive

Full sync — indexes all supported files in the configured folder:

```bash
curl -X POST http://localhost:8000/sync-drive \
  -H "Content-Type: application/json" \
  -d '{"force_full": true}'
```

Incremental sync — processes only Drive changes since last run:

```bash
curl -X POST http://localhost:8000/sync-drive \
  -H "Content-Type: application/json" \
  -d '{}'
```

Response:

```json
{
  "started_at": "2026-05-07T04:33:18Z",
  "finished_at": "2026-05-07T04:33:49Z",
  "files_seen": 1,
  "files_indexed": 1,
  "files_skipped": 0,
  "chunks_indexed": 124,
  "errors": []
}
```

### Ask a Question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "what is the summary of the company"}'
```

With optional filters and custom top_k:

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "refund policy", "top_k": 5, "filters": {"metadata.source": "gdrive"}}'
```

Response:

```json
{
  "answer": "The company is MD ACCOUNTANTS & AUDITORS INC. ...",
  "sources": [
    {
      "doc_id": "ae0ae7a24e6db09fb245fb850e65a9109632bab9",
      "file_name": "Policy_43.pdf",
      "web_view_link": "",
      "chunk_id": "d2d19ac5c2d711a5",
      "score": 1.0,
      "snippet": "written instructions, notes, memoranda..."
    }
  ],
  "retrieval": {
    "bm25_hits": 6,
    "dense_hits": 6,
    "fused_hits": 12,
    "latency_ms": 5938.0
  }
}
```

### Health Checks

```bash
curl http://localhost:8000/healthz   # {"status": "ok"}
curl http://localhost:8000/readyz    # {"status": "ready", "opensearch": "up"}
```

---

## Eval

Run the golden-set evaluation against a live server:

```bash
python eval/run_eval.py
```

Output:

```
question                                       | has_answer | has_sources | keyword_hit | latency_ms
-------------------------------------------------------------------------------------------------------
What is the refund policy?                     |        yes |         yes |         yes |      4821.3
Who should I contact for IT support?           |        yes |         yes |          no |      3102.7
...
retrieval_rate:   100%  (5/5)
keyword_accuracy:  80%  (4/5)
avg_latency_ms:   4211.4
```

---

## Environment Variables

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `OPENSEARCH_HOST` | str | `localhost` | OpenSearch hostname |
| `OPENSEARCH_PORT` | int | `9200` | OpenSearch port |
| `ANTHROPIC_API_KEY` | str | `""` | API key for Claude (required if `LLM_PROVIDER=anthropic`) |
| `GDRIVE_CREDENTIALS_PATH` | str | `credentials.json` | Path to GCP service account JSON |
| `GDRIVE_FOLDER_ID` | str | `""` | Google Drive folder ID to index |
| `EMBEDDING_MODEL` | str | `BAAI/bge-small-en-v1.5` | HuggingFace model name for embeddings |
| `LLM_MODEL` | str | `claude-sonnet-4-5` | Model name passed to the LLM provider |
| `LLM_PROVIDER` | str | `anthropic` | `anthropic` or `google` |
| `GOOGLE_API_KEY` | str | `""` | API key for Google Gemma (required if `LLM_PROVIDER=google`) |
| `CHUNK_SIZE` | int | `800` | Max characters per chunk |
| `CHUNK_OVERLAP` | int | `120` | Overlap characters carried between chunks |
| `TOP_K` | int | `8` | Number of chunks returned by retriever |

---

## Folder Structure

```
RAG-on-Google-Drive/
├── src/
│   ├── api/            # FastAPI app, /sync-drive and /ask routes
│   ├── connectors/     # Google Drive connector (service account auth)
│   ├── core/           # Settings (pydantic-settings), structured logging
│   ├── embedding/      # Sentence Transformers embedder with sha256 cache
│   ├── models/         # Pydantic models: Document, Chunk, API request/response
│   ├── processing/     # PDF/DOCX/GDoc text extraction, cleaning, chunking
│   ├── rag/            # LLM backends, prompt builder, RAG pipeline
│   ├── search/         # OpenSearch client, index management, hybrid retriever
│   └── sync/           # Sync orchestrator, page token + content-hash state store
├── eval/
│   ├── golden_set.json # 5-question golden set
│   └── run_eval.py     # Eval harness (httpx, no FastAPI dependency)
├── tests/
├── docker-compose.yml  # OpenSearch + API services
├── Dockerfile
├── pyproject.toml
└── .env.example
```

---

## Design Decisions

- **Hybrid retrieval over pure dense search** — BM25 catches exact keyword matches (product names, error codes, proper nouns) that dense embeddings miss; RRF fusion avoids score-scale mismatch between the two signals without requiring tuned interpolation weights.

- **Incremental sync via Drive Changes API** — full re-index of a large folder on every run is expensive and slow; page tokens let the system process only deltas, making frequent background sync practical without burning embedding API quota.

- **Content-hash dedup** — Drive `modifiedTime` is updated by Google even for metadata-only changes (e.g. sharing settings); comparing sha256 of normalized text prevents unnecessary re-embedding and re-indexing when the actual content hasn't changed.

- **LLM provider abstraction** — a thin `LLM` ABC with `AnthropicLLM` and `GoogleAILLM` implementations means the retrieval and prompt logic is untouched when switching providers; new providers (OpenAI, Cohere, local Ollama) require adding one class and one factory branch.

---

## Engineering Challenges

1. **hatchling build failure** — `pip install -e .` failed because `pyproject.toml` had no `[tool.hatch.build.targets.wheel]` stanza. Hatchling's auto-discovery can't resolve a flat `src/` layout without explicit direction. Fixed by adding `packages = ["src"]` to the wheel target.

2. **`lru_cache` on `Settings`** — startup raised `TypeError: unhashable type: 'Settings'` because Pydantic v2 `BaseSettings` objects are mutable and cannot be LRU cache keys. Removed the `Settings` parameter from `get_client()` and called `get_settings()` inside the cached function instead.

3. **tenacity `before_sleep_log` type mismatch** — `before_sleep_log` received the string `"warning"` instead of the integer constant `logging.WARNING`, crashing at import time. Fixed by importing stdlib `logging` and passing `logging.WARNING`.

4. **Wrong service account project** — Drive API returned `403 Forbidden` on every call despite correct scopes. The `service_account.json` belonged to a different GCP project with a different email identity; the folder was shared with a different SA. Fixed by placing the correct JSON and re-sharing the folder with its email.

5. **`google.generativeai` SDK deprecated mid-build** — `ImportError` on `GenerativeModel` after a pip update. Google split the package; the new SDK is `google-genai` with a completely different `Client` API. Migrated `GoogleAILLM` to `from google import genai`, wrapped the synchronous call in `asyncio.to_thread()`.

6. **OpenSearch 400: knn nested inside `bool.must`** — every `/ask` call returned 400 because `dense_search` wrapped the `knn` clause inside a `bool.must`. OpenSearch's knn plugin requires `knn` to be the root `query` value. Restructured to `{"query": {"knn": {"vector": {...}}}}` flat.

7. **OpenSearch 400: empty-dict term filter from Swagger** — Swagger UI auto-fills `filters` with `{"additionalProp1": {}}`, producing an invalid `{"term": {"additionalProp1": {}}}` clause. Added a `valid_filters` guard that strips `None`, `""`, `{}`, and `[]` values before building the query.

8. **`force_full=true` not bypassing content-hash dedup** — sync returned `files_skipped=1, chunks_indexed=0` even with an empty index. The `force_full` flag only controlled which file list path ran (full vs incremental); the per-file `if cached_hash == content_hash: skip` check ran unconditionally. Fixed the condition to `if not request.force_full and cached_hash == content_hash`.

9. **`service_account.json` missing from container image** — the Dockerfile copied `src/` and `eval/` but not the credentials file; `GoogleDriveConnector` failed immediately inside the container with a file-not-found error. Baking secrets into images is also bad practice — the JSON would be embedded in the image layer and recoverable via `docker history`. Fixed by volume-mounting `./service_account.json:/app/service_account.json:ro` in `docker-compose.yml`, keeping the image generic and the secret machine-local.

10. **API crashed on container start before OpenSearch was ready** — `depends_on: - opensearch` list form only waits for the container process to start, not for OpenSearch to accept connections (~15s warmup). The FastAPI lifespan's `ensure_index()` fired immediately into a refused connection and crashed. Fixed by using `condition: service_healthy` with the existing OpenSearch `healthcheck` — Compose now blocks API startup until `/9200` returns 200.
