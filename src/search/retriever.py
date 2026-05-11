import asyncio
import time
from typing import Optional

from opensearchpy._async.client import AsyncOpenSearch

from src.core.config import Settings
from src.embedding.base import Embedder
from src.models.api import RetrievalDebug
from src.models.chunk import EmbeddedChunk
from src.search.index import _index_name


def reciprocal_rank_fusion(
    rankings: list[list[str]], k: int = 60
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    def __init__(
        self, client: AsyncOpenSearch, embedder: Embedder, settings: Settings
    ) -> None:
        self._client = client
        self._embedder = embedder
        self._settings = settings
        self._index = _index_name(settings)

    async def bm25_search(
        self, query: str, top_k: int, filters: Optional[dict] = None
    ) -> list[tuple[str, float]]:
        valid_filters = {k: v for k, v in (filters or {}).items() if v not in (None, "", {}, [])}
        if valid_filters:
            body = {
                "size": top_k,
                "query": {
                    "bool": {
                        "must": [{"match": {"text": query}}],
                        "filter": [{"term": {f: v}} for f, v in valid_filters.items()]
                    }
                }
            }
        else:
            body = {"size": top_k, "query": {"match": {"text": query}}}
        resp = await self._client.search(index=self._index, body=body)
        return [(h["_source"]["chunk_id"], h["_score"]) for h in resp["hits"]["hits"]]

    async def dense_search(
        self, vector: list[float], top_k: int, filters: Optional[dict] = None
    ) -> list[tuple[str, float]]:
        body = {
            "size": top_k,
            "query": {
                "knn": {
                    "vector": {
                        "vector": vector,
                        "k": top_k
                    }
                }
            }
        }
        resp = await self._client.search(index=self._index, body=body)
        return [(h["_source"]["chunk_id"], h["_score"]) for h in resp["hits"]["hits"]]

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict] = None,
    ) -> tuple[list[str], RetrievalDebug]:
        k = top_k or self._settings.top_k
        candidates = k * 2

        t0 = time.monotonic()
        vector = await self._embedder.embed_query(query)
        bm25_results, dense_results = await asyncio.gather(
            self.bm25_search(query, candidates, filters),
            self.dense_search(vector, candidates, filters),
        )
        latency_ms = (time.monotonic() - t0) * 1000

        bm25_ids = [cid for cid, _ in bm25_results]
        dense_ids = [cid for cid, _ in dense_results]

        fused = reciprocal_rank_fusion([bm25_ids, dense_ids], k=self._settings.rrf_k)
        top_ids = [cid for cid, _ in fused[:k]]

        debug = RetrievalDebug(
            bm25_hits=len(bm25_results),
            dense_hits=len(dense_results),
            fused_hits=len(fused),
            latency_ms=round(latency_ms, 2),
        )
        return top_ids, debug

    async def fetch_chunks(self, chunk_ids: list[str]) -> list[EmbeddedChunk]:
        if not chunk_ids:
            return []

        resp = await self._client.mget(
            body={"ids": chunk_ids},
            index=self._index,
        )
        id_to_chunk: dict[str, EmbeddedChunk] = {}
        for doc in resp["docs"]:
            if doc.get("found"):
                id_to_chunk[doc["_id"]] = EmbeddedChunk(**doc["_source"])

        return [id_to_chunk[cid] for cid in chunk_ids if cid in id_to_chunk]
