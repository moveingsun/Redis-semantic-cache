# Redis-semantic-cache

Minimal semantic cache layer backed by Redis.

## What it does

- Stores prompt/response pairs in Redis.
- Stores an embedding vector for each prompt.
- On lookup, compares a new prompt embedding against cached entries using cosine similarity.
- Returns the best cached response when the similarity is above a threshold.

## Usage

```python
from semantic_cache import RedisSemanticCache
import redis


def embed(text: str) -> list[float]:
    # Replace with your real embedding function/model.
    return [float(len(text)), float(sum(ord(c) for c in text) % 1000)]


client = redis.Redis(host="localhost", port=6379, db=0)
cache = RedisSemanticCache(redis_client=client, embedder=embed)

cache.put("How do I reset my password?", "Open settings > account > reset password.")

hit = cache.get("I forgot my password")
if hit:
    print(hit.response, hit.similarity)
```