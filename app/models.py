from typing import Any
from typing import Optional

from pydantic import BaseModel
from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    api_key: str = Field(alias="API_KEY")


class EmbeddingChunk(BaseModel):
    text: str
    url: str
    embedding: Optional[list[float]] = None
