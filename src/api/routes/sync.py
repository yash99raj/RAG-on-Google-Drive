import structlog
from fastapi import APIRouter

from src.connectors.google_drive import GoogleDriveConnector
from src.core.config import Settings, get_settings
from src.embedding.st_embedder import SentenceTransformerEmbedder
from src.models.api import SyncRequest, SyncResponse
from src.search.opensearch_client import get_client
from src.sync.orchestrator import SyncOrchestrator
from src.sync.state import OpenSearchStateStore

router = APIRouter()
logger = structlog.get_logger(__name__)

_embedder: SentenceTransformerEmbedder | None = None


def _get_embedder() -> SentenceTransformerEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformerEmbedder()
    return _embedder


def _get_orchestrator(settings: Settings) -> SyncOrchestrator:
    client = get_client()
    state = OpenSearchStateStore(client, settings.opensearch_index_prefix)
    return SyncOrchestrator(
        connector=GoogleDriveConnector(),
        state=state,
        embedder=_get_embedder(),
        client=client,
        settings=settings,
    )


@router.post("/sync-drive", response_model=SyncResponse)
async def sync_drive(request: SyncRequest) -> SyncResponse:
    settings = get_settings()
    orchestrator = _get_orchestrator(settings)
    logger.info("sync started", folder_id=request.folder_id, force_full=request.force_full)
    response = await orchestrator.run(request)
    logger.info(
        "sync finished",
        files_seen=response.files_seen,
        files_indexed=response.files_indexed,
        files_skipped=response.files_skipped,
        chunks_indexed=response.chunks_indexed,
        errors=len(response.errors),
    )
    return response
