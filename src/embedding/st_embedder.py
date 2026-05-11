import asyncio
import hashlib
from typing import Optional

import numpy as np
import structlog

from src.core.config import get_settings
from src.embedding.base import Embedder

logger = structlog.get_logger(__name__)

_PASSAGE_PREFIX = "Represent this sentence for searching relevant passages: "
_QUERY_PREFIX = "Represent this query for searching relevant passages: "
_BATCH_SIZE = 32


class SentenceTransformerEmbedder(Embedder):
    def __init__(self) -> None:
        self._model = None
        self._cache: dict[str, list[float]] = {}

    async def _get_model(self):
        if self._model is None:
            settings = get_settings()
            from sentence_transformers import SentenceTransformer

            self._model = await asyncio.to_thread(
                SentenceTransformer, settings.embedding_model
            )
        return self._model

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        model = await self._get_model()
        results: list[Optional[list[float]]] = [None] * len(texts)
        to_encode_indices: list[int] = []
        to_encode_texts: list[str] = []

        for i, text in enumerate(texts):
            key = hashlib.sha256(text.encode()).hexdigest()
            if key in self._cache:
                results[i] = self._cache[key]
            else:
                to_encode_indices.append(i)
                to_encode_texts.append(_PASSAGE_PREFIX + text)

        hits = len(texts) - len(to_encode_indices)
        logger.debug(
            "embedding batch",
            total=len(texts),
            cache_hits=hits,
            to_encode=len(to_encode_indices),
            hit_rate=round(hits / len(texts), 2) if texts else 0,
        )

        for batch_start in range(0, len(to_encode_texts), _BATCH_SIZE):
            batch = to_encode_texts[batch_start : batch_start + _BATCH_SIZE]
            batch_indices = to_encode_indices[batch_start : batch_start + _BATCH_SIZE]

            vecs: np.ndarray = await asyncio.to_thread(
                model.encode,
                batch,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

            for idx, vec, orig_text in zip(
                batch_indices, vecs, texts[batch_start : batch_start + _BATCH_SIZE]
            ):
                vec_list = vec.tolist()
                key = hashlib.sha256(orig_text.encode()).hexdigest()
                self._cache[key] = vec_list
                results[idx] = vec_list

        return results  # type: ignore[return-value]

    async def embed_query(self, text: str) -> list[float]:
        model = await self._get_model()
        prefixed = _QUERY_PREFIX + text
        vec: np.ndarray = await asyncio.to_thread(
            model.encode,
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.tolist()
