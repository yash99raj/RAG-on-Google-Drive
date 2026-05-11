from fastapi import APIRouter

from src.core.config import get_settings
from src.embedding.st_embedder import SentenceTransformerEmbedder
from src.models.api import AskRequest, AskResponse
from src.rag.llm import get_llm
from src.rag.pipeline import RAGPipeline
from src.search.opensearch_client import get_client
from src.search.retriever import HybridRetriever

router = APIRouter()

_embedder: SentenceTransformerEmbedder | None = None


def _get_embedder() -> SentenceTransformerEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformerEmbedder()
    return _embedder


def _get_pipeline() -> RAGPipeline:
    settings = get_settings()
    client = get_client()
    retriever = HybridRetriever(client, _get_embedder(), settings)
    llm = get_llm(settings)
    return RAGPipeline(retriever, llm, settings)


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    pipeline = _get_pipeline()
    return await pipeline.run(request)
