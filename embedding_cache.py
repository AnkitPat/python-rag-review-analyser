import hashlib
from typing import Dict

class EmbeddingCache:
    def __init__(self):
        self.cache: Dict[str, list] = {}

    def get_hash(self, text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()

    def has(self, text: str) -> bool:
        return self.get_hash(text) in self.cache

    def get(self, text: str):
        return self.cache.get(self.get_hash(text))

    def set(self, text: str, embedding):
        self.cache[self.get_hash(text)] = embedding
