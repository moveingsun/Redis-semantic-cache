from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Callable, Iterable


Embedding = Iterable[float]


@dataclass(frozen=True)
class CacheHit:
    prompt: str
    response: str
    similarity: float


class RedisSemanticCache:
    """A minimal semantic cache layer that uses Redis as storage."""

    def __init__(
        self,
        redis_client,
        embedder: Callable[[str], Embedding],
        *,
        key_prefix: str = "semantic-cache",
        namespace: str = "default",
        ttl_seconds: int | None = None,
    ) -> None:
        self._redis = redis_client
        self._embedder = embedder
        self._key_prefix = key_prefix
        self._namespace = namespace
        self._ttl_seconds = ttl_seconds

    def put(self, prompt: str, response: str) -> None:
        vector = self._normalize(self._embedder(prompt))
        entry_id = self._entry_id(prompt)
        entry_key = self._entry_key(entry_id)

        self._redis.hset(
            entry_key,
            mapping={
                "prompt": prompt,
                "response": response,
                "embedding": json.dumps(vector),
            },
        )
        self._redis.sadd(self._index_key(), entry_id)

        if self._ttl_seconds is not None:
            self._redis.expire(entry_key, self._ttl_seconds)

    def get(self, prompt: str, *, min_similarity: float = 0.85) -> CacheHit | None:
        query_embedding = self._normalize(self._embedder(prompt))
        best_hit: CacheHit | None = None

        for raw_entry_id in self._redis.smembers(self._index_key()):
            entry_id = self._decode(raw_entry_id)
            data = self._redis.hgetall(self._entry_key(entry_id))
            if not data:
                continue

            stored_prompt = self._decode(data.get(b"prompt") or data.get("prompt"))
            stored_response = self._decode(data.get(b"response") or data.get("response"))
            embedding_raw = self._decode(data.get(b"embedding") or data.get("embedding"))
            if stored_prompt is None or stored_response is None or embedding_raw is None:
                continue

            stored_embedding = self._normalize(json.loads(embedding_raw))
            similarity = self._cosine_similarity(query_embedding, stored_embedding)
            if similarity < min_similarity:
                continue

            if best_hit is None or similarity > best_hit.similarity:
                best_hit = CacheHit(
                    prompt=stored_prompt,
                    response=stored_response,
                    similarity=similarity,
                )

        return best_hit

    def delete(self, prompt: str) -> None:
        entry_id = self._entry_id(prompt)
        self._redis.srem(self._index_key(), entry_id)
        self._redis.delete(self._entry_key(entry_id))

    def clear(self) -> None:
        entry_ids = [self._decode(raw) for raw in self._redis.smembers(self._index_key())]
        for entry_id in entry_ids:
            if entry_id is not None:
                self._redis.delete(self._entry_key(entry_id))
        self._redis.delete(self._index_key())

    def _entry_id(self, prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def _entry_key(self, entry_id: str) -> str:
        return f"{self._key_prefix}:{self._namespace}:entry:{entry_id}"

    def _index_key(self) -> str:
        return f"{self._key_prefix}:{self._namespace}:entries"

    @staticmethod
    def _normalize(embedding: Embedding) -> list[float]:
        values = [float(v) for v in embedding]
        if not values:
            return []
        magnitude = math.sqrt(sum(v * v for v in values))
        if magnitude == 0:
            return [0.0 for _ in values]
        return [v / magnitude for v in values]

    @staticmethod
    def _cosine_similarity(left: Embedding, right: Embedding) -> float:
        left_values = list(left)
        right_values = list(right)
        if not left_values or not right_values:
            return 0.0

        dimensions = min(len(left_values), len(right_values))
        return sum(left_values[i] * right_values[i] for i in range(dimensions))

    @staticmethod
    def _decode(value) -> str | None:
        if value is None:
            return None
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return str(value)
