from typing import List, Optional
from pydantic import BaseModel, Field

class FileAttachment(BaseModel):
    """Model for file attachments in requests"""
    filename: str = Field(..., description="Name of the attached file")
    content: str = Field(..., description="Base64 encoded file content")
    content_type: str = Field(..., description="MIME type of the file")

    class Config:
        json_schema_extra = {
            "example": {
                "filename": "error_log.txt",
                "content": "SGVsbG8gV29ybGQ=",  # "Hello World" in base64
                "content_type": "text/plain",
            }
        }

class QuestionRequest(BaseModel):
    """Model for incoming question requests"""
    question: str = Field(
        ..., min_length=1, max_length=800, description="The student's question"
    )
    attachments: Optional[List[FileAttachment]] = Field(
        default=None, description="Optional file attachments"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "question": "How do I deploy a FastAPI application to Railway?",
                "attachments": [],
            }
        }

class LinkResponse(BaseModel):
    """Model for relevant links in responses"""
    url: str = Field(..., description="URL of the relevant resource")
    text: str = Field(..., description="Description or excerpt of the linked content")

    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://discourse.onlinedegree.iitm.ac.in/t/fastapi-deployment/456",
                "text": "FastAPI deployment guide and best practices",
            }
        }

class AnswerResponse(BaseModel):
    """Model for API responses to questions"""
    answer: str = Field(..., description="The generated answer to the question")
    links: List[LinkResponse] = Field(
        default_factory=list, description="Relevant links and resources"
    )
    source: str = Field(
        default="local_model", description="Response source: local_model, cache, rule_based, or fallback"
    )
    model_used: Optional[str] = Field(
        default=None, description="Model that generated the response"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "To deploy FastAPI to Railway: 1) Create requirements.txt 2) Use uvicorn main:app --host 0.0.0.0 --port $PORT 3) Set environment variables 4) Deploy from GitHub",
                "links": [
                    {
                        "url": "https://discourse.onlinedegree.iitm.ac.in/t/fastapi-deployment/456",
                        "text": "FastAPI deployment guide",
                    }
                ],
                "source": "local_model",
                "model_used": "Qwen/Qwen2.5-0.5B-Instruct",
            }
        }

class ServiceCapabilities(BaseModel):
    """Model for service capabilities"""
    local_model_responses: bool = Field(..., description="Whether local model responses are available")
    rule_based_responses: bool = Field(
        ..., description="Whether rule-based responses are available"
    )
    response_caching: bool = Field(
        ..., description="Whether response caching is enabled"
    )
    error_recovery: bool = Field(
        default=True, description="Whether error recovery is available"
    )

class HealthResponse(BaseModel):
    """Model for health check responses"""
    status: str = Field(
        ..., description="Overall health status: healthy, degraded, unhealthy, error"
    )
    timestamp: str = Field(..., description="ISO timestamp of health check")
    service: str = Field(default="TDS Virtual TA", description="Service name")
    version: str = Field(default="1.0.0", description="Service version")
    issues: List[str] = Field(
        default_factory=list, description="Critical issues affecting service"
    )
    warnings: List[str] = Field(
        default_factory=list, description="Non-critical warnings"
    )
    capabilities: ServiceCapabilities = Field(..., description="Service capabilities")
    ai_provider: Optional[str] = Field(
        default="Local Qwen Model", description="AI provider being used"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2025-05-31T23:30:45.123456",
                "service": "TDS Virtual TA",
                "version": "1.0.0",
                "issues": [],
                "warnings": [],
                "capabilities": {
                    "local_model_responses": True,
                    "rule_based_responses": True,
                    "response_caching": True,
                    "error_recovery": True,
                },
                "ai_provider": "Local Qwen Model",
            }
        }