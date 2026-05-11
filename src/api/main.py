from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.core.config import get_settings
from src.core.logging import configure_logging
from src.api.routes.ask import router as ask_router
from src.api.routes.sync import router as sync_router
from src.search.index import ensure_index
from src.search.opensearch_client import get_client
from src.sync.state import ensure_state_index


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = get_settings()
    client = get_client()
    await ensure_index(client, settings)
    await ensure_state_index(client, settings.opensearch_index_prefix)
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="highwatch-rag", lifespan=lifespan)

    app.include_router(sync_router)
    app.include_router(ask_router)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        settings = get_settings()
        url = f"http://{settings.opensearch_host}:{settings.opensearch_port}/"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(url, timeout=2.0)
                resp.raise_for_status()
            return {"status": "ready", "opensearch": "up"}
        except Exception as exc:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "opensearch": "down", "error": str(exc)},
            )

    return app
