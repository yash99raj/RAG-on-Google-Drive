from opensearchpy._async.client import AsyncOpenSearch
from opensearchpy.helpers import async_bulk

from src.core.config import Settings
from src.models.chunk import EmbeddedChunk


def _index_name(settings: Settings) -> str:
    return f"{settings.opensearch_index_prefix}_chunks_v1"


async def ensure_index(client: AsyncOpenSearch, settings: Settings) -> None:
    name = _index_name(settings)
    if await client.indices.exists(index=name):
        return
    await client.indices.create(
        index=name,
        body={
            "settings": {
                "index.knn": True,
                "number_of_shards": 1,
                "number_of_replicas": 0,
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "doc_id": {"type": "keyword"},
                    "text": {"type": "text", "similarity": "BM25"},
                    "order": {"type": "integer"},
                    "char_start": {"type": "integer"},
                    "char_end": {"type": "integer"},
                    "page": {"type": "integer"},
                    "content_hash": {"type": "keyword"},
                    "vector": {
                        "type": "knn_vector",
                        "dimension": settings.embedding_dim,
                        "method": {
                            "name": "hnsw",
                            "engine": "lucene",
                            "space_type": "cosinesimil",
                        },
                    },
                    "metadata": {
                        "type": "object",
                        "properties": {
                            "file_name": {"type": "keyword"},
                            "mime_type": {"type": "keyword"},
                            "modified_time": {"type": "date"},
                            "web_view_link": {"type": "keyword"},
                            "source": {"type": "keyword"},
                        },
                    },
                }
            },
        },
    )


async def bulk_upsert(
    client: AsyncOpenSearch, settings: Settings, embedded_chunks: list[EmbeddedChunk]
) -> int:
    index = _index_name(settings)
    actions = [
        {
            "_op_type": "index",
            "_index": index,
            "_id": chunk.chunk_id,
            **chunk.model_dump(),
        }
        for chunk in embedded_chunks
    ]
    succeeded, _ = await async_bulk(client, actions)
    return succeeded


async def delete_by_doc_id(
    client: AsyncOpenSearch, index_name: str, doc_id: str
) -> int:
    resp = await client.delete_by_query(
        index=index_name,
        body={"query": {"term": {"doc_id": doc_id}}},
    )
    return resp.get("deleted", 0)
