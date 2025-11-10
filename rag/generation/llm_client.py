from openai import OpenAI


class LLMClient:
    """Универсальный клиент для работы с LLM через OpenAI-совместимый API"""

    def __init__(self, base_url: str, model: str, api_key: str = "test-key"):
        """
        Args:
            base_url: базовый URL API (например, http://127.0.0.1:8000/v1/ или https://api.deepseek.com/v1/)
            model: название модели
            api_key: API ключ
        """
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 256,
        temperature: float = 0.7
    ) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = resp.choices[0].message.content

            if not content:
                print(f"WARNING: LLM returned empty content")
                print(f"Response: {resp}")

            return content or ""
        except Exception as e:
            print(f"ERROR in LLM chat: {e}")
            print(f"Model: {self.model}")
            print(f"Base URL: {self.client.base_url}")
            raise


# Обратная совместимость
VLLMClient = LLMClient
