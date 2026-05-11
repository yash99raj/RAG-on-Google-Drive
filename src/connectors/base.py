from abc import ABC, abstractmethod
from datetime import datetime
from typing import AsyncIterator, Optional

from pydantic import BaseModel


class ChangeEvent(BaseModel):
    file_id: str
    name: str
    mime_type: str
    modified_time: datetime
    deleted: bool = False


class DocumentConnector(ABC):
    @abstractmethod
    async def list_files(
        self, folder_id: Optional[str]
    ) -> AsyncIterator[ChangeEvent]: ...

    @abstractmethod
    async def list_changes(
        self, page_token: str
    ) -> tuple[list[ChangeEvent], str]: ...

    @abstractmethod
    async def get_start_page_token(self) -> str: ...

    @abstractmethod
    async def download(self, file_id: str) -> bytes: ...

    @abstractmethod
    async def export_gdoc(self, file_id: str) -> bytes: ...
