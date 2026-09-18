import redis.asyncio as aioredis

from aichat.serve.redis_settings import redis_settings

_pool = None


def get_pool():
    global _pool
    if _pool is None and redis_settings.redis_host:
        _pool = aioredis.ConnectionPool(
            host=redis_settings.redis_host, password=redis_settings.redis_password
        )
    return _pool


def discard_pool():
    """Discard the cached pool without awaiting disconnect. The old pool's
    connections will be cleaned up by the garbage collector.  Used in tests
    to avoid reusing connections that were created on a now-closed event loop."""
    global _pool
    _pool = None
