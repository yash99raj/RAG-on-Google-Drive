import asyncio
import io
import logging
from datetime import datetime
from typing import AsyncIterator, Optional

import structlog
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from tenacity import before_sleep_log, retry, stop_after_attempt, wait_exponential

from src.connectors.base import ChangeEvent, DocumentConnector
from src.core.config import get_settings

logger = structlog.get_logger(__name__)
_stdlib_logger = logging.getLogger(__name__)

SUPPORTED_MIME_TYPES: set[str] = {
    "application/pdf",
    "application/vnd.google-apps.document",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "text/markdown",
}

_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

_MIME_FILTER = " or ".join(
    f"mimeType='{m}'" for m in SUPPORTED_MIME_TYPES
)


def _retry():
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
        before_sleep=before_sleep_log(_stdlib_logger, logging.WARNING),
    )


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.rstrip("Z")).replace(
        tzinfo=None
    ) if value else datetime.utcnow()


class GoogleDriveConnector(DocumentConnector):
    def __init__(self) -> None:
        settings = get_settings()
        creds = service_account.Credentials.from_service_account_file(
            settings.gdrive_credentials_path,
            scopes=_SCOPES,
        )
        self._service = build("drive", "v3", credentials=creds, cache_discovery=False)

    async def list_files(
        self, folder_id: Optional[str] = None
    ) -> AsyncIterator[ChangeEvent]:
        query = f"({_MIME_FILTER}) and trashed=false"
        if folder_id:
            query = f"'{folder_id}' in parents and {query}"

        @_retry()
        def _page(token: Optional[str]):
            kwargs = dict(
                q=query,
                fields="nextPageToken,files(id,name,mimeType,modifiedTime)",
                pageSize=100,
            )
            if token:
                kwargs["pageToken"] = token
            return self._service.files().list(**kwargs).execute()

        page_token: Optional[str] = None
        while True:
            result = await asyncio.to_thread(_page, page_token)
            for f in result.get("files", []):
                yield ChangeEvent(
                    file_id=f["id"],
                    name=f["name"],
                    mime_type=f["mimeType"],
                    modified_time=_parse_dt(f.get("modifiedTime", "")),
                )
            page_token = result.get("nextPageToken")
            if not page_token:
                break

    async def list_changes(
        self, page_token: str
    ) -> tuple[list[ChangeEvent], str]:
        @_retry()
        def _fetch():
            return (
                self._service.changes()
                .list(
                    pageToken=page_token,
                    fields="nextPageToken,newStartPageToken,changes(fileId,removed,file(name,mimeType,modifiedTime,trashed))",
                )
                .execute()
            )

        result = await asyncio.to_thread(_fetch)
        events: list[ChangeEvent] = []
        for change in result.get("changes", []):
            deleted = change.get("removed", False)
            f = change.get("file") or {}
            if not deleted:
                deleted = f.get("trashed", False)
            mime = f.get("mimeType", "")
            if not deleted and mime not in SUPPORTED_MIME_TYPES:
                continue
            events.append(
                ChangeEvent(
                    file_id=change["fileId"],
                    name=f.get("name", ""),
                    mime_type=mime,
                    modified_time=_parse_dt(f.get("modifiedTime", "")),
                    deleted=deleted,
                )
            )
        next_token: str = result.get("nextPageToken") or result.get(
            "newStartPageToken", page_token
        )
        return events, next_token

    async def get_start_page_token(self) -> str:
        @_retry()
        def _fetch():
            return self._service.changes().getStartPageToken().execute()

        result = await asyncio.to_thread(_fetch)
        return result["startPageToken"]

    async def download(self, file_id: str) -> bytes:
        @_retry()
        def _fetch():
            request = self._service.files().get_media(fileId=file_id)
            buf = io.BytesIO()
            downloader = MediaIoBaseDownload(buf, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            return buf.getvalue()

        return await asyncio.to_thread(_fetch)

    async def export_gdoc(self, file_id: str) -> bytes:
        @_retry()
        def _fetch():
            request = self._service.files().export(
                fileId=file_id, mimeType="text/plain"
            )
            buf = io.BytesIO()
            downloader = MediaIoBaseDownload(buf, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            return buf.getvalue()

        return await asyncio.to_thread(_fetch)
