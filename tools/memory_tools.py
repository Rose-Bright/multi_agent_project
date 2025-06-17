from langchain_core.tools import tool
from utils.memory import MemoryManager

memory_manager = MemoryManager()

@tool
def retrieve_memory(
    user_id: str = "123",
    date: str = None,
    subject: str = None,
    role: str = None,
    agent_name: str = None,
    k: int = 5
) -> list:
    """
    Retrieve relevant long-term memories for the user.
    Returns a list of dicts with 'text' and 'metadata'.
    """
    all_memories = memory_manager.get_all_long_term_memory(user_id)
    filtered = all_memories

    if agent_name:
        filtered = [m for m in filtered if m["metadata"].get("agent") == agent_name]
    if role:
        filtered = [m for m in filtered if m["metadata"].get("role") == role]
    if date:
        filtered = [m for m in filtered if m["metadata"].get("timestamp", "").startswith(date)]
    if subject:
        filtered = [m for m in filtered if subject.lower() in m["text"].lower()]

    # Return up to k results, each with text and metadata
    return filtered[:k]
