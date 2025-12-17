from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    database_url: str = "postgresql://user:password@localhost/dbname"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None
    openai_api_key: str = ""
    neon_database_url: str = ""
    debug: bool = False

    class Config:
        env_file = ".env"

settings = Settings()