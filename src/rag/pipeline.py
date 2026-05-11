from src.core.config import Settings
from src.models.api import AskRequest, AskResponse, Source
from src.rag.llm import LLM
from src.rag.prompt import build_system_prompt, build_user_prompt
from src.search.retriever import HybridRetriever


class RAGPipeline:
    def __init__(
        self, retriever: HybridRetriever, llm: LLM, settings: Settings
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._settings = settings

    async def run(self, request: AskRequest) -> AskResponse:
        top_k = request.top_k or self._settings.top_k

        chunk_ids, debug = await self._retriever.search(
            query=request.query,
            top_k=top_k,
            filters=request.filters,
        )
        chunks = await self._retriever.fetch_chunks(chunk_ids)

        system = build_system_prompt()
        user = build_user_prompt(request.query, chunks)
        answer = await self._llm.complete(system, user)

        id_to_score = dict(zip(chunk_ids, [1.0 / (i + 1) for i in range(len(chunk_ids))]))

        sources = [
            Source(
                doc_id=chunk.doc_id,
                file_name=chunk.metadata.file_name,
                web_view_link=chunk.metadata.web_view_link,
                chunk_id=chunk.chunk_id,
                score=id_to_score.get(chunk.chunk_id, 0.0),
                snippet=chunk.text[:200],
            )
            for chunk in chunks
        ]

        return AskResponse(answer=answer, sources=sources, retrieval=debug)
