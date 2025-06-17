import os
from dotenv import load_dotenv
import uuid
import datetime
from langchain.embeddings import init_embeddings
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain_chroma import Chroma

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

vectorstore = Chroma(persist_directory="./chroma_db")

class MemoryManager:
    def __init__(self):
        self.short_term = InMemorySaver()

        # With semantic search
        embeddings = init_embeddings(
            "openai:text-embedding-3-small",
            api_key=api_key)
        self.vectorstore = Chroma(
            embedding_function=embeddings,
            persist_directory="./chroma_db"
        )
        self.long_term = InMemoryStore(index={"embed": embeddings, "dims": 1536})

    def get_short_term(self):
        return self.short_term

    def get_long_term(self):
        return self.long_term

    def save_chat_history(self, thread_id, new_messages):
        existing = self.get_latest_chat_history(thread_id)
        for msg in new_messages:
            if msg not in existing:
                existing.append(msg)

        # Save to short-term (checkpointing)
        checkpoint_id = str(uuid.uuid4())
        now = datetime.datetime.now(datetime.UTC).isoformat()
        checkpoint_ns = f"{thread_id}-ns"
        checkpoint = {
            "v": 1,
            "id": checkpoint_id,
            "ts": now,
            "channel_values": {"chat_history": existing},
            "channel_versions": {"chat_history": checkpoint_id},
            "versions_seen": {},
            "pending_sends": [],
        }
        metadata = {
            "source": "input",
            "step": 0,
            "writes": {"chat_history": existing},
            "parents": {},
        }
        self.short_term.put(
            {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_ns": checkpoint_ns
                }
            },
            checkpoint,
            metadata,
            {"chat_history": checkpoint_id}
        )

    def get_latest_chat_history(self, thread_id):
        checkpoint_ns = f"{thread_id}-ns"
        checkpoints = list(self.short_term.list({"configurable": {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}}))
        if checkpoints:
            return checkpoints[0].checkpoint.get("channel_values", {}).get("chat_history", [])
        return []

    def write_long_term_memory(self, user_id, text: str, role: str = None, agent_name: str = None):
        """Save text to Chroma vectorstore with user_id, agent/role, and timestamp as metadata."""
        metadata = {
            "user_id": user_id,
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat()
        }
        if role:
            metadata["role"] = role
        if agent_name:
            metadata["agent"] = agent_name
        self.vectorstore.add_texts([text], metadatas=[metadata])

    def search_long_term_memory(self, user_id, query: str, k=5):
        """Semantic search in Chroma vectorstore for a user's memories."""
        results = self.vectorstore.similarity_search(query, k=k, filter={"user_id": user_id})
        return [{"text": doc.page_content, "metadata": doc.metadata} for doc in results]

    def get_all_long_term_memory(self, user_id):
        """Return all long-term memory entries for a user from Chroma."""
        # Chroma does not have a direct 'list all' API, so you can search with a blank query
        results = self.vectorstore.similarity_search("", k=100, filter={"user_id": user_id})
        return [{"text": doc.page_content, "metadata": doc.metadata} for doc in results]
