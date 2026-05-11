import functools

from opensearchpy._async.client import AsyncOpenSearch

from src.core.config import get_settings


@functools.lru_cache
def get_client() -> AsyncOpenSearch:
    settings = get_settings()
    return AsyncOpenSearch(
        hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
        http_auth=None,
        use_ssl=False,
        verify_certs=False,
    )
