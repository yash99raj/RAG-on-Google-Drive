import asyncio
from abc import ABC, abstractmethod

from src.core.config import Settings, get_settings


class LLM(ABC):
    @abstractmethod
    async def complete(self, system: str, user: str) -> str: ...


class AnthropicLLM(LLM):
    def __init__(self, settings: Settings) -> None:
        import anthropic
        self._client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        self._model = settings.llm_model

    async def complete(self, system: str, user: str) -> str:
        msg = await self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return msg.content[0].text


class GoogleAILLM(LLM):
    def __init__(self, settings: Settings) -> None:
        from google import genai
        self._client = genai.Client(api_key=settings.google_api_key)
        # strip any erroneous prefix (e.g. "gemini/" or "models/")
        model = settings.llm_model
        for prefix in ("models/", "gemini/"):
            if model.startswith(prefix):
                model = model[len(prefix):]
                break
        self._model = model

    async def complete(self, system: str, user: str) -> str:
        from google.genai import types
        prompt = f"{system}\n\n{user}"
        response = await asyncio.to_thread(
            self._client.models.generate_content,
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=1024),
        )
        return response.text


def get_llm(settings: Settings | None = None) -> LLM:
    if settings is None:
        settings = get_settings()
    if settings.llm_provider == "google":
        return GoogleAILLM(settings)
    return AnthropicLLM(settings)
