from rag.generation import LLMClient
from config import settings


def create_llm_client() -> LLMClient:
    provider = settings.AI_PROVIDER.lower()

    if provider == "vllm":
        return LLMClient(
            base_url=settings.VLLM_BASE_URL,
            model=settings.VLLM_MODEL_NAME,
            api_key=settings.VLLM_API_KEY
        )
    elif provider == "deepseek":
        if not settings.DEEPSEEK_API_KEY:
            raise ValueError(
                "DEEPSEEK_API_KEY не установлен. "
                "Добавьте его в .env файл или переменные окружения."
            )
        return LLMClient(
            base_url=settings.DEEPSEEK_BASE_URL,
            model=settings.DEEPSEEK_MODEL_NAME,
            api_key=settings.DEEPSEEK_API_KEY
        )
    elif provider == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY не установлен. "
                "Добавьте его в .env файл или переменные окружения."
            )
        return LLMClient(
            base_url=settings.OPENAI_BASE_URL,
            model=settings.OPENAI_MODEL_NAME,
            api_key=settings.OPENAI_API_KEY
        )
    elif provider == "gemini":
        if not settings.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY не установлен. "
                "Добавьте его в .env файл или переменные окружения."
            )
        return LLMClient(
            base_url=settings.GEMINI_BASE_URL,
            model=settings.GEMINI_MODEL_NAME,
            api_key=settings.GEMINI_API_KEY
        )
    else:
        raise ValueError(
            f"Неизвестный провайдер: {provider}. "
            f"Поддерживаемые провайдеры: vllm, deepseek, gemini"
        )
