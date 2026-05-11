import hashlib
from datetime import datetime, timezone

import structlog
from opensearchpy._async.client import AsyncOpenSearch

from src.connectors.base import DocumentConnector
from src.core.config import Settings
from src.embedding.base import Embedder
from src.models.api import SyncRequest, SyncResponse
from src.models.chunk import ChunkMetadata, EmbeddedChunk
from src.models.document import Document
from src.processing.chunk import chunk_text
from src.processing.clean import clean_text
from src.processing.extract import extract_text
from src.search.index import _index_name, bulk_upsert, delete_by_doc_id
from src.sync.state import OpenSearchStateStore

logger = structlog.get_logger(__name__)

_GDOC_MIME = "application/vnd.google-apps.document"


def _now() -> datetime:
    return datetime.now(timezone.utc)


class SyncOrchestrator:
    def __init__(
        self,
        connector: DocumentConnector,
        state: OpenSearchStateStore,
        embedder: Embedder,
        client: AsyncOpenSearch,
        settings: Settings,
    ) -> None:
        self._connector = connector
        self._state = state
        self._embedder = embedder
        self._client = client
        self._settings = settings
        self._index = _index_name(settings)

    async def run(self, request: SyncRequest) -> SyncResponse:
        started_at = _now()
        files_seen = files_indexed = files_skipped = chunks_indexed = 0
        errors: list[str] = []

        existing_token = await self._state.get_page_token()
        if not request.force_full and existing_token:
            changes, new_token = await self._connector.list_changes(existing_token)
            await self._state.set_page_token(new_token)
            events = changes
        else:
            token = await self._connector.get_start_page_token()
            await self._state.set_page_token(token)
            events = [e async for e in self._connector.list_files(request.folder_id)]

        for event in events:
            files_seen += 1
            doc_id = hashlib.sha1(event.file_id.encode()).hexdigest()

            if event.deleted:
                try:
                    await delete_by_doc_id(self._client, self._index, doc_id)
                    await self._state.delete_doc(doc_id)
                except Exception as exc:
                    logger.error("delete failed", doc_id=doc_id, error=str(exc))
                    errors.append(str(exc))
                continue

            try:
                if event.mime_type == _GDOC_MIME:
                    raw = await self._connector.export_gdoc(event.file_id)
                else:
                    raw = await self._connector.download(event.file_id)

                text = clean_text(extract_text(raw, event.mime_type, event.name))
                content_hash = hashlib.sha256(text.encode()).hexdigest()

                cached_hash = await self._state.get_content_hash(doc_id)
                if not request.force_full and cached_hash == content_hash:
                    files_skipped += 1
                    continue

                doc = Document(
                    doc_id=doc_id,
                    gdrive_file_id=event.file_id,
                    name=event.name,
                    mime_type=event.mime_type,
                    modified_time=event.modified_time,
                    web_view_link="",
                    content_hash=content_hash,
                )
                meta = ChunkMetadata(
                    file_name=doc.name,
                    mime_type=doc.mime_type,
                    modified_time=doc.modified_time,
                    web_view_link=doc.web_view_link,
                )
                chunks = chunk_text(
                    text,
                    doc_id,
                    meta,
                    self._settings.chunk_size,
                    self._settings.chunk_overlap,
                )
                vectors = await self._embedder.embed_texts([c.text for c in chunks])
                embedded = [
                    EmbeddedChunk(**c.model_dump(), vector=v)
                    for c, v in zip(chunks, vectors)
                ]

                await delete_by_doc_id(self._client, self._index, doc_id)
                n = await bulk_upsert(self._client, self._settings, embedded)
                await self._state.set_content_hash(doc_id, content_hash)

                files_indexed += 1
                chunks_indexed += n
                logger.info(
                    "indexed file",
                    name=event.name,
                    doc_id=doc_id,
                    chunks=n,
                )

            except Exception as exc:
                logger.error("index failed", name=event.name, error=str(exc))
                errors.append(str(exc))

        return SyncResponse(
            started_at=started_at,
            finished_at=_now(),
            files_seen=files_seen,
            files_indexed=files_indexed,
            files_skipped=files_skipped,
            chunks_indexed=chunks_indexed,
            errors=errors,
        )
