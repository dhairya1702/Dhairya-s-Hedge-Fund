import requests
import re
from typing import Optional, Type
from pydantic import BaseModel
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration


class ChatOllama(BaseChatModel):
    model: str = "llama3"
    base_url: str = "http://localhost:11434"
    temperature: float = 0.3
    max_tokens: int = 1024
    schema: Optional[Type[BaseModel]] = None
    forced_instruction: Optional[str] = None

    def _format_messages(self, messages: list[BaseMessage]) -> str:
        formatted = "\n".join(
            f"{'User' if m.type == 'human' else 'Assistant'}: {m.content}"
            for m in messages
        )
        if self.schema and self.forced_instruction:
            formatted += self.forced_instruction
        return formatted

    def _extract_json(self, text: str) -> str:
        match = re.search(r"<json>(.*?)</json>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text.strip()

    def _generate(self, messages: list[BaseMessage], stop: Optional[list[str]] = None) -> ChatResult:
        prompt = self._format_messages(messages)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
        }

        #print("📤 [DEBUG] Sending payload to Ollama:", payload)

        response = requests.post(f"{self.base_url}/api/generate", json=payload)
        response.raise_for_status()
        result = response.json()["response"]

        #print("\n🔍 [DEBUG] Ollama Raw Output:", repr(result))

        cleaned_result = self._extract_json(result)
        print("\n🧼 [DEBUG] Cleaned JSON Result:", cleaned_result)

        if self.schema:
            try:
                parsed = self.schema.model_validate_json(cleaned_result)
                print("✅ [DEBUG] Successfully parsed Pydantic model:", parsed)
                print("✅ [DEBUG] Parsed type:", type(parsed))
                # 🧪 Manual wrap to simulate actual LangChain flow
                fake_result = ChatResult(generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content=cleaned_result,
                            additional_kwargs={"parsed": parsed}
                        )
                    )
                ])

                # 🔍 Try to access what call_llm would access
                gen = fake_result.generations[0]
                print("\n🧪 gen.message.additional_kwargs:", gen.message.additional_kwargs)

                if "parsed" in gen.message.additional_kwargs:
                    print("\n✅ Extracted parsed model:", gen.message.additional_kwargs["parsed"])
                else:
                    print("❌ No parsed model found")

                # ✅ Return this simulated result
                return fake_result
                return parsed  # ✅ Return the parsed object directly
            except Exception as e:
                print("❌ [ERROR] Failed to parse structured output:", e)
                print("⚠️ [DEBUG] Raw cleaned result was:", cleaned_result)

        # Default fallback (used only if no schema)
        print("⚠️ [DEBUG] Falling back to raw text response wrapping")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=cleaned_result))])

    def with_structured_output(self, schema: Type[BaseModel], **kwargs):
        self.schema = schema
        self.forced_instruction = (
            "\n\nReturn ONLY the following structured JSON output wrapped in <json> tags."
            "\nDO NOT include any commentary or markdown or explanation."
            "\nExample:\n<json>\n{...your JSON here...}\n</json>"
        )
        return self

    @property
    def _llm_type(self) -> str:
        return "ollama"

    @property
    def _identifying_params(self):
        return {
            "model": self.model,
            "base_url": self.base_url
        }
