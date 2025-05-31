from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Model Configuration
    model_name: str = Field(
        default="Qwen/Qwen2.5-0.5B-Instruct",
        description="Local Qwen model to use"
    )
    use_fp16: bool = Field(
        default=True,
        description="Use FP16 precision for model inference"
    )

    # Generation Parameters
    max_new_tokens: int = Field(default=500, description="Maximum tokens to generate")  # Increased
    temperature: float = Field(default=0.7, description="Generation temperature")  # Adjusted
    top_p: float = Field(default=0.95, description="Top-p sampling")  # Adjusted
    repetition_penalty: float = Field(default=1.2, description="Repetition penalty")  # Adjusted

    # Application Configuration
    app_host: str = Field(default="0.0.0.0", description="Application host")
    app_port: int = Field(default=8000, description="Application port")
    debug: bool = Field(default=False, description="Debug mode")

    # Caching and Optimization
    enable_response_cache: bool = Field(
        default=True, description="Enable response caching"
    )
    cache_size: int = Field(default=500, description="Maximum cache entries")

settings = Settings()