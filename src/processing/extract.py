import io

import structlog
from pypdf import PdfReader
from docx import Document as DocxDocument

logger = structlog.get_logger(__name__)


def extract_text(file_bytes: bytes, mime_type: str, file_name: str) -> str:
    if mime_type == "application/pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif mime_type == "application/vnd.google-apps.document":
        text = file_bytes.decode("utf-8")
    elif mime_type in ("text/plain", "text/markdown"):
        text = file_bytes.decode("utf-8")
    elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = DocxDocument(io.BytesIO(file_bytes))
        text = "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError(f"Unsupported mime type: {mime_type}")

    logger.info("extracted text", file_name=file_name, chars=len(text))
    return text
