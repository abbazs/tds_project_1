"""
Configuration settings using Pydantic Settings
"""

from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000

    # AIpipe settings
    aipipe_url: str = "https://aipipe.org/api/v1/generate"
    model_name: str = "qwen2.5"

    # Generation parameters
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    max_new_tokens: int = 1000

    # Cache settings
    enable_cache: bool = True
    cache_max_size: int = 1000
    cache_ttl: int = 3600  # 1 hour

    # API settings
    api_timeout: int = 30
    max_file_size: int = 10 * 1024 * 1024  # 10MB

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_prefix = "TDS_"
        case_sensitive = False

    def get_generation_params(self) -> dict:
        """Get parameters for AI model generation"""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "max_new_tokens": self.max_new_tokens,
        }
