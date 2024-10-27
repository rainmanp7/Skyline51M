# knowledge_base.py
from collections import deque
import asyncio
import logging

class SimpleKnowledgeBase:
    def __init__(self, max_recent_items=100):
        self.data = {}
        self.recent_updates = deque(maxlen=max_recent_items)
        self.lock = asyncio.Lock()

    async def update(self, key, value):
        async with self.lock:
            if key in self.data:
                self.data[key].extend(value)
                self.data[key] = list(set(self.data[key]))
            else:
                self.data[key] = value
            self.recent_updates.append((key, value))
