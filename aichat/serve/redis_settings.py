from pydantic_settings import BaseSettings


class RedisSettings(BaseSettings):
    redis_password: str | None = None
    redis_host: str = "redis-master"
    redis_db: int = 0


redis_settings = RedisSettings()
