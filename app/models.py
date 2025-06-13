"""
Pydantic models for API requests and responses
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import validator


class Link(BaseModel):
    """Link model for reference URLs"""

    url: str = Field(..., description="URL of the reference link")
    text: str = Field(..., description="Display text for the link")

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://fastapi.tiangolo.com/deployment/",
                "text": "FastAPI Deployment Guide",
            }
        }


class QuestionRequest(BaseModel):
    """Request model for student questions"""

    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Student's question about the TDS course",
    )
    context: str | None = Field(
        None, max_length=1000, description="Additional context for the question"
    )
    student_id: str | None = Field(
        None,
        description="Optional student identifier for personalized responses",
    )

    @validator("question")
    def question_must_not_be_empty(self, v):
        if not v.strip():
            raise ValueError("Question cannot be empty or just whitespace")
        return v.strip()

    class Config:
        json_schema_extra = {
            "example": {
                "question": "How do I deploy my FastAPI application to Railway?",
                "context": "I've built a FastAPI app for my TDS project and need to deploy it",
                "student_id": "CS21B001",
            }
        }


class AnswerResponse(BaseModel):
    """Response model for answers"""

    answer: str = Field(..., description="The generated answer to the question")
    links: list[Link] = Field(
        default=[], description="Relevant reference links"
    )
    source: str = Field(
        ..., description="Source of the answer (rule_based, ai_model, fallback)"
    )
    model_name: str = Field(
        ..., description="Name of the model used to generate the answer"
    )
    processing_time: float = Field(
        ..., description="Time taken to process the request in seconds"
    )
    metadata: dict[str, Any] = Field(
        default={}, description="Additional metadata about the response"
    )
    timestamp: datetime = Field(
        default_factory=datetime.now,
        description="When the response was generated",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "To deploy FastAPI to Railway, you need to create a Dockerfile and railway.toml file...",
                "links": [
                    {
                        "url": "https://docs.railway.app/deploy/deployments",
                        "text": "Railway Deployment Guide",
                    }
                ],
                "source": "ai_model",
                "model_name": "qwen2.5",
                "processing_time": 1.23,
                "metadata": {"confidence": 0.95, "topic": "deployment"},
                "timestamp": "2024-01-15T10:30:00",
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check"""

    status: str = Field(..., description="Overall system status")
    version: str = Field(..., description="Application version")
    uptime: float = Field(..., description="System uptime in seconds")
    capabilities: list[str] = Field(
        ..., description="List of system capabilities"
    )
    diagnostics: dict[str, Any] = Field(
        ..., description="Detailed diagnostic information"
    )
    warnings: list[str] = Field(default=[], description="Any system warnings")
    model_info: dict[str, Any] = Field(
        ..., description="Information about the AI model"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "uptime": 3600.5,
                "capabilities": [
                    "AI question answering",
                    "Rule-based matching",
                    "Response caching",
                    "File processing",
                ],
                "diagnostics": {
                    "memory_usage": "45.2MB",
                    "cache_hit_rate": 0.78,
                    "total_requests": 150,
                },
                "warnings": [],
                "model_info": {
                    "name": "qwen2.5",
                    "temperature": 0.7,
                    "max_tokens": 1000,
                },
            }
        }
