import hashlib

from src.models.chunk import Chunk, ChunkMetadata

_SEPARATORS = ["\n\n", "\n", " ", ""]


def _split(text: str, chunk_size: int) -> list[str]:
    # Iterative: apply each separator in turn to any chunk still over the limit.
    # Avoids the recursion-depth explosion of the naive recursive approach.
    chunks = [text]
    for sep in _SEPARATORS:
        next_chunks: list[str] = []
        for chunk in chunks:
            if len(chunk) <= chunk_size:
                next_chunks.append(chunk)
                continue
            if sep == "":
                next_chunks.extend(
                    chunk[i : i + chunk_size] for i in range(0, len(chunk), chunk_size)
                )
                continue
            parts = chunk.split(sep)
            current = ""
            for part in parts:
                candidate = (current + sep + part) if current else part
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current:
                        next_chunks.append(current)
                    current = part
            if current:
                next_chunks.append(current)
        chunks = next_chunks
        if all(len(c) <= chunk_size for c in chunks):
            break
    return chunks


def chunk_text(
    text: str,
    doc_id: str,
    metadata: ChunkMetadata,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    raw_chunks = _split(text, chunk_size)

    chunks: list[Chunk] = []
    char_pos = 0
    overlap_carry = ""

    for order, raw in enumerate(raw_chunks):
        chunk_text_str = (overlap_carry + raw) if overlap_carry else raw
        if len(chunk_text_str) > chunk_size:
            chunk_text_str = chunk_text_str[-chunk_size:]

        char_start = max(0, char_pos - len(overlap_carry))
        char_end = char_start + len(chunk_text_str)

        chunk_id = hashlib.sha256(f"{doc_id}{order}".encode()).hexdigest()[:16]

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=chunk_text_str,
                order=order,
                char_start=char_start,
                char_end=char_end,
                page=None,
                metadata=metadata,
            )
        )

        char_pos += len(raw)
        overlap_carry = chunk_text_str[-chunk_overlap:] if chunk_overlap else ""

    return chunks
