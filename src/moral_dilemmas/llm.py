import os
from dataclasses import dataclass, field
from typing import Any

from ollama import Client


@dataclass
class LLM:
    """Send prompts to an Ollama-compatible model."""

    model: str
    system_prompt: str
    host: str | None = None
    temperature: float = 0.1
    timeout: float = 120.0
    client: Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.host = self.host or os.getenv("OLLAMA_HOST")

        if self.host is None:
            raise ValueError("Set OLLAMA_HOST, or pass host explicitly.")

        headers = self._auth_headers()
        self.client = Client(host=self.host, headers=headers, timeout=self.timeout)

    def generate(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        """Send a prompt to the model and return the generated text."""

        generation_options = {
            "temperature": self.temperature if temperature is None else temperature,
        }

        if options is not None:
            generation_options.update(options)

        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                system=self.system_prompt,
                stream=False,
                options=generation_options,
            )
        except ConnectionError as error:
            raise ConnectionError(
                f"Could not connect to Ollama at {self.host}. "
                "Check the exact HTW Ollama endpoint, including protocol and port."
            ) from error

        return self._response_text(response)

    def connected(self) -> bool:
        """Return whether Ollama is reachable."""
        try:
            self.client.list()
        except Exception:
            return False
        return True

    @staticmethod
    def _response_text(response: Any) -> str:
        if hasattr(response, "response"):
            return response.response
        return response["response"]

    @staticmethod
    def _auth_headers() -> dict[str, str] | None:
        api_key = os.getenv("OLLAMA_API_KEY")
        if api_key is None:
            return None
        return {"Authorization": f"Bearer {api_key}"}
