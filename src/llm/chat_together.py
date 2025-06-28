import os
import requests
import json
from typing import Optional, Type
from pydantic import BaseModel, ValidationError


class ChatTogether:
    def __init__(self, model: str, api_key: str = None, temperature: float = 0.7, max_tokens: int = 512):
        self.model = model
        self.api_key = api_key or os.getenv("TOGETHER_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.schema: Optional[Type[BaseModel]] = None  # For structured output

        if not self.api_key:
            raise ValueError("Together API key not found in environment variables.")

        self.url = "https://api.together.xyz/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def with_structured_output(self, schema: Type[BaseModel]):
        self.schema = schema
        return self

    def invoke(self, prompt: str) -> str | BaseModel:
        messages = [{"role": "user", "content": prompt}]

        # Inject schema formatting instruction if set
        if self.schema:
            messages.insert(0, {
                "role": "system",
                "content": (
                    f"You must respond ONLY in JSON format matching this schema:\n"
                    f"{self.schema.schema_json(indent=2)}"
                )
            })

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

        response = requests.post(self.url, headers=self.headers, json=payload)
        response.raise_for_status()

        raw = response.json()["choices"][0]["message"]["content"]

        if self.schema:
            try:
                # Clean up any code block wrapping (e.g., ```json ... ```)
                json_start = raw.find("{")
                json_end = raw.rfind("}") + 1
                json_str = raw[json_start:json_end]
                return self.schema.parse_raw(json_str)
            except (json.JSONDecodeError, ValidationError) as e:
                raise ValueError(f"Failed to parse structured output: {e}\nRaw output: {raw}")

        return raw

    def __call__(self, prompt: str) -> str | BaseModel:
        return self.invoke(prompt)
