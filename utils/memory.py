from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

class ShortTermMemoryManager:
    def __init__(self):
        self.saver = InMemorySaver()

    def get_saver(self):
        return self.saver

class LongTermMemoryManager:
    def __init__(self):
        self.store = InMemoryStore()

    def get_store(self):
        return self.store

    def put_user_data(self, user_id: str, data: dict):
        self.store.put(("users",), user_id, data)

    def get_user_data(self, user_id: str):
        info = self.store.get(("users",), user_id)
        return info.value if info else None
