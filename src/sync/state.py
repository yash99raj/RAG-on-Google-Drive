from abc import ABC, abstractmethod
from typing import Optional

from opensearchpy._async.client import AsyncOpenSearch


class SyncStateStore(ABC):
    @abstractmethod
    async def get_page_token(self) -> Optional[str]: ...

    @abstractmethod
    async def set_page_token(self, token: str) -> None: ...

    @abstractmethod
    async def get_content_hash(self, doc_id: str) -> Optional[str]: ...

    @abstractmethod
    async def set_content_hash(self, doc_id: str, hash: str) -> None: ...

    @abstractmethod
    async def delete_doc(self, doc_id: str) -> None: ...


class OpenSearchStateStore(SyncStateStore):
    def __init__(self, client: AsyncOpenSearch, prefix: str) -> None:
        self._client = client
        self._index = f"{prefix}_sync_state_v1"

    async def get_page_token(self) -> Optional[str]:
        return await self._get("page_token")

    async def set_page_token(self, token: str) -> None:
        await self._put("page_token", "page_token", token)

    async def get_content_hash(self, doc_id: str) -> Optional[str]:
        return await self._get(doc_id)

    async def set_content_hash(self, doc_id: str, hash: str) -> None:
        await self._put("content_hash", doc_id, hash)

    async def delete_doc(self, doc_id: str) -> None:
        try:
            await self._client.delete(index=self._index, id=doc_id)
        except Exception:
            pass

    async def _get(self, doc_id: str) -> Optional[str]:
        try:
            resp = await self._client.get(index=self._index, id=doc_id)
            return resp["_source"]["value"]
        except Exception:
            return None

    async def _put(self, kind: str, doc_id: str, value: str) -> None:
        await self._client.index(
            index=self._index,
            id=doc_id,
            body={"kind": kind, "value": value},
        )


async def ensure_state_index(client: AsyncOpenSearch, prefix: str) -> None:
    index = f"{prefix}_sync_state_v1"
    if await client.indices.exists(index=index):
        return
    await client.indices.create(
        index=index,
        body={
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {
                "properties": {
                    "kind": {"type": "keyword"},
                    "value": {"type": "keyword"},
                }
            },
        },
    )
