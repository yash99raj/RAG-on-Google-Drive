from src.models.chunk import EmbeddedChunk


def build_system_prompt() -> str:
    return (
        "You are a helpful assistant. Answer using ONLY the provided context. "
        "If the answer is not in the context, say so. "
        "Cite source document name for each claim."
    )


def build_user_prompt(query: str, chunks: list[EmbeddedChunk]) -> str:
    context_parts = [
        f"[Source: {chunk.metadata.file_name} | chunk {chunk.order}]\n{chunk.text}"
        for chunk in chunks
    ]
    context = "\n\n".join(context_parts)
    return f"{context}\n\nQuestion: {query}\nAnswer:"
