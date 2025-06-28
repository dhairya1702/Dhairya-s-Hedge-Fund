"""Helper functions for LLM"""
from langchain_core.messages import AIMessage

import json
from typing import TypeVar, Type, Optional, Any
from pydantic import BaseModel
from utils.progress import progress

T = TypeVar('T', bound=BaseModel)

def call_llm(
    prompt: Any,
    model_name: str,
    model_provider: str,
    pydantic_model: Type[T],
    agent_name: Optional[str] = None,
    max_retries: int = 3,
    default_factory=None
) -> T:
    from llm.models import get_model, get_model_info
    model_info = get_model_info(model_name)
    llm = get_model(model_name, model_provider)

    if not (model_info and not model_info.has_json_mode()):
        llm = llm.with_structured_output(
            pydantic_model,
            method="json_mode",
        )

    for attempt in range(max_retries):
        try:
            print(f"🔥 Attempt {attempt+1}: Calling {model_name} from {model_provider}")
            result = llm.invoke(prompt)

            print("\nLLM.py 🔍 [DEBUG] Raw LLM result:", result)
            print("\nLLM.py 🔍 [DEBUG] Type of result:", type(result))

            if isinstance(result, pydantic_model):
                print("✅ Result is already a structured Pydantic model")
                return result

            # ✅ CASE: LangChain returned ChatResult
            if hasattr(result, "generations"):
                gen = result.generations[0]
                message = gen.message if hasattr(gen, "message") else gen  # Fallback
            elif isinstance(result, AIMessage):
                message = result
            else:
                raise ValueError("Unknown response format from LLM")

            # ✅ Try to pull parsed object if it's in additional_kwargs
            if "parsed" in message.additional_kwargs:
                parsed = message.additional_kwargs["parsed"]
                print("✅ [DEBUG] Found structured model in .additional_kwargs:", parsed)
                return parsed

            # ✅ CASE 1: Already structured output (like LLaMA returns Pydantic directly)
            if isinstance(result, pydantic_model):
                print("✅ Result is already a structured Pydantic model")
                return result

            # ✅ CASE 2: Non-JSON-mode model like Deepseek (needs manual extraction)
            if model_info and not model_info.has_json_mode():
                parsed_result = extract_json_from_deepseek_response(result.content)
                print("📦 Extracted Deepseek JSON:", parsed_result)
                if parsed_result:
                    return pydantic_model(**parsed_result)

            # ✅ CASE 3: Standard Langchain response (ChatResult object)
            if hasattr(result, "generations"):
                raw_content = result.generations[0].text if hasattr(result.generations[0], "text") else result.generations[0].message.content
                print("🧪 Parsed raw content from ChatResult:", raw_content)
                return pydantic_model(**json.loads(raw_content))

            # ❌ None matched — fallback error
            raise ValueError("Unknown response format from LLM")

        except Exception as e:
            print("💥 Exception caught:", e)
            if agent_name:
                progress.update_status(agent_name, None, f"Error - retry {attempt + 1}/{max_retries}")
            if attempt == max_retries - 1:
                print(f"❌ Error in LLM call after {max_retries} attempts: {e}")
                return default_factory() if default_factory else create_default_response(pydantic_model)

    return create_default_response(pydantic_model)





def create_default_response(model_class: Type[T]) -> T:
    """Creates a safe default response based on the model's fields."""
    default_values = {}
    for field_name, field in model_class.model_fields.items():
        if field.annotation == str:
            default_values[field_name] = "Error in analysis, using default"
        elif field.annotation == float:
            default_values[field_name] = 0.0
        elif field.annotation == int:
            default_values[field_name] = 0
        elif hasattr(field.annotation, "__origin__") and field.annotation.__origin__ == dict:
            default_values[field_name] = {}
        else:
            # For other types (like Literal), try to use the first allowed value
            if hasattr(field.annotation, "__args__"):
                default_values[field_name] = field.annotation.__args__[0]
            else:
                default_values[field_name] = None

    return model_class(**default_values)

def extract_json_from_deepseek_response(content: str) -> Optional[dict]:
    """Extracts JSON from Deepseek's markdown-formatted response."""
    try:
        json_start = content.find("```json")
        if json_start != -1:
            json_text = content[json_start + 7:]  # Skip past ```json
            json_end = json_text.find("```")
            if json_end != -1:
                json_text = json_text[:json_end].strip()
                return json.loads(json_text)
    except Exception as e:
        print(f"Error extracting JSON from Deepseek response: {e}")
    return None