import time
import hashlib
from datetime import datetime
from collections import deque
from typing import Dict, Optional, List
from pydantic import BaseModel

class LinkResponse(BaseModel):
    url: str
    text: str

class ResponseCache:
    """LRU cache for API responses"""
    def __init__(self, max_size: int = 500):
        self.cache: Dict[str, str] = {}
        self.max_size = max_size
        self.access_order: deque[str] = deque()

    def _get_key(self, question: str) -> str:
        """Generate cache key from question"""
        return hashlib.md5(question.lower().strip().encode()).hexdigest()[:16]

    def get(self, question: str) -> Optional[str]:
        """Get cached response for question"""
        key = self._get_key(question)
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def set(self, question: str, response: str):
        """Cache response for question"""
        key = self._get_key(question)
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = self.access_order.popleft()
            del self.cache[oldest_key]
        self.cache[key] = response
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

def get_service_health(settings, response_cache: Optional[ResponseCache] = None) -> Dict:
    """Get comprehensive service health status with error handling"""
    try:
        issues = []
        warnings = []

        # Check model availability
        try:
            import torch
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(settings.model_name)
        except Exception as e:
            issues.append(f"Failed to load model {settings.model_name}: {str(e)}")

        # Determine overall health
        if issues:
            health_status = "unhealthy"
        else:
            health_status = "healthy"

        return {
            "status": health_status,
            "issues": issues,
            "warnings": warnings,
            "capabilities": {
                "local_model_responses": not issues,
                "rule_based_responses": True,
                "response_caching": bool(response_cache and settings.enable_response_cache),
                "error_recovery": True,
            },
        }

    except Exception as e:
        return {
            "status": "error",
            "issues": [f"Health check failed: {str(e)}"],
            "warnings": [],
            "capabilities": {
                "local_model_responses": False,
                "rule_based_responses": True,
                "response_caching": False,
                "error_recovery": True,
            },
        }

def generate_rule_based_response(question: str) -> Dict:
    """Generate rule-based response using knowledge base"""
    question_lower = question.lower()
    knowledge_base = {
        "ai_model": {
            "keywords": ["model", "gpt", "ai", "openai", "which model", "what model"],
            "answer": "For TDS assignments, you **must use `gpt-3.5-turbo-0125`** (not gpt-4o-mini or other models). This is specified in the project requirements to ensure consistent results across all submissions.",
            "links": [
                {
                    "url": "https://discourse.onlinedegree.iitm.ac.in/t/ai-model-requirements/123",
                    "text": "Official AI model requirements for TDS assignments",
                }
            ],
        },
        "fastapi": {
            "keywords": [
                "fastapi",
                "api",
                "deploy",
                "deployment",
                "uvicorn",
                "railway",
            ],
            "answer": "For FastAPI deployment on Railway/Render: 1) Create `requirements.txt` with dependencies 2) Use `uvicorn main:app --host 0.0.0.0 --port $PORT` 3) Set environment variables in platform dashboard 4) Add CORS middleware 5) Include health check endpoint",
            "links": [
                {
                    "url": "https://discourse.onlinedegree.iitm.ac.in/t/fastapi-deployment/456",
                    "text": "FastAPI deployment guide and best practices",
                }
            ],
        },
        "pydantic": {
            "keywords": ["pydantic", "validation", "field", "basemodel", "error"],
            "answer": "Common Pydantic solutions: 1) Use `Field(...)` for required fields 2) Check data types match your model 3) Use `Optional[Type]` for optional fields 4) Verify JSON structure matches model definition 5) Import `BaseModel` and `Field` correctly",
            "links": [
                {
                    "url": "https://discourse.onlinedegree.iitm.ac.in/t/pydantic-errors/789",
                    "text": "Common Pydantic validation errors and solutions",
                }
            ],
        },
    }

    best_match = None
    best_score = 0

    for category, data in knowledge_base.items():
        score = 0
        for keyword in data["keywords"]:
            if keyword in question_lower:
                score += len(keyword)
        if score > best_score:
            best_score = score
            best_match = data

    if best_match:
        return {
            "answer": best_match["answer"],
            "links": [LinkResponse(**link) for link in best_match.get("links", [])],
            "source": "rule_based",
        }

    default_answers = [
        "I'm here to help with your TDS course questions! For specific technical issues, please provide more details about your problem.",
        "That's an interesting question! For the most accurate help, could you share more context about what you're trying to achieve?",
        "I can help with Python, FastAPI, Pydantic, Machine Learning, and other TDS topics. Please be more specific about your question.",
    ]

    return {
        "answer": random.choice(default_answers),
        "links": [
            LinkResponse(
                url="https://discourse.onlinedegree.iitm.ac.in/c/tools-data-science",
                text="TDS Discourse forum for detailed discussions",
            )
        ],
        "source": "fallback",
    }