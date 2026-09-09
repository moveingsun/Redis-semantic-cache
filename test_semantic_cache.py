import unittest

from semantic_cache import RedisSemanticCache


class FakeRedis:
    def __init__(self):
        self._hashes = {}
        self._sets = {}
        self._expirations = {}

    def hset(self, key, mapping):
        self._hashes[key] = {
            k.encode("utf-8"): v.encode("utf-8") if isinstance(v, str) else str(v).encode("utf-8")
            for k, v in mapping.items()
        }

    def hgetall(self, key):
        return dict(self._hashes.get(key, {}))

    def sadd(self, key, value):
        self._sets.setdefault(key, set()).add(value.encode("utf-8") if isinstance(value, str) else value)

    def smembers(self, key):
        return set(self._sets.get(key, set()))

    def srem(self, key, value):
        encoded = value.encode("utf-8") if isinstance(value, str) else value
        self._sets.setdefault(key, set()).discard(encoded)

    def delete(self, key):
        self._hashes.pop(key, None)
        self._sets.pop(key, None)
        self._expirations.pop(key, None)

    def expire(self, key, ttl_seconds):
        self._expirations[key] = ttl_seconds


def simple_embed(text: str) -> list[float]:
    lower = text.lower()
    if "weather" in lower:
        return [1.0, 0.0, 0.0]
    if "password" in lower:
        return [0.0, 1.0, 0.0]
    if "database" in lower:
        return [0.0, 0.0, 1.0]
    return [1.0, 1.0, 1.0]


class RedisSemanticCacheTests(unittest.TestCase):
    def setUp(self):
        self.redis = FakeRedis()
        self.cache = RedisSemanticCache(redis_client=self.redis, embedder=simple_embed)

    def test_put_and_get_hit(self):
        self.cache.put("reset password", "Use account settings.")
        hit = self.cache.get("password reset steps", min_similarity=0.1)
        self.assertIsNotNone(hit)
        assert hit is not None
        self.assertEqual(hit.response, "Use account settings.")

    def test_get_miss_when_similarity_too_low(self):
        self.cache.put("weather forecast", "Check weather app.")
        hit = self.cache.get("database migration", min_similarity=0.99999)
        self.assertIsNone(hit)

    def test_delete_removes_entry(self):
        self.cache.put("order status", "Check your orders page.")
        self.cache.delete("order status")
        self.assertIsNone(self.cache.get("order status", min_similarity=0.1))

    def test_ttl_is_applied(self):
        cache = RedisSemanticCache(redis_client=self.redis, embedder=simple_embed, ttl_seconds=30)
        cache.put("hello", "world")
        self.assertEqual(len(self.redis._expirations), 1)
        self.assertIn(30, self.redis._expirations.values())


if __name__ == "__main__":
    unittest.main()
