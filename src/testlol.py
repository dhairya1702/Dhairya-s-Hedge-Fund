from pydantic import BaseModel
from llm.chat_ollama import ChatOllama  # Adjust import as needed

# Define a dummy schema matching what your model outputs
class SignalSchema(BaseModel):
    signal: str
    confidence: float
    reasoning: str

# Set up your wrapper with structured output
llm = ChatOllama(model="llama3").with_structured_output(SignalSchema)

# Simulate a chat history
messages = [
    # You can also import HumanMessage and AIMessage, this is fine for now
    {"type": "human", "content": "Give a bearish signal with reasoning and confidence in JSON <json> tags"},
]

# Quick LangChain-compatible message conversion
from langchain_core.messages import HumanMessage
msg_chain = [HumanMessage(content=m["content"]) for m in messages]

# Call invoke directly
result = llm.invoke(msg_chain)

print("✅ invoke() returned:", result)
print("✅ Type:", type(result))

# Check if .generations and .additional_kwargs["parsed"] works
if hasattr(result, "generations"):
    msg = result.generations[0].message
    print("🧠 Raw content:", msg.content)
    print("🧠 Parsed model (if any):", msg.additional_kwargs.get("parsed"))
