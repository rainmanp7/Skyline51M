
# Beginning of knowledge_base.py
from collections import deque

class SimpleKnowledgeBase:
    def __init__(self, max_recent_items=100):
        self.data = {}
        self.recent_updates = deque(maxlen=max_recent_items)

    def update(self, key, value):
        with knowledge_lock:
            if key in self.data:
                self.data[key].extend(value)
                self.data[key] = list(set(self.data[key]))
            else:
                self.data[key] = value
            self.recent_updates.append((key, value))
# End of knowledge_base.py
