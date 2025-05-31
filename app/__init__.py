from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional, Dict
import base64
import json
import asyncio
from datetime import datetime
import httpx
import time
from collections import deque
import hashlib
import random

# Pydantic Settings Configuration
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Hugging Face Configuration
    huggingface_api_key: Optional[str] = Field(default=None, description="Hugging Face API key")
    
    # Model Selection (choose based on your needs)
    primary_model: str = Field(
        default="Qwen/Qwen2.5-7B-Instruct",
        description="Primary model to use"
    )
    fallback_model: str = Field(
        default="microsoft/DialoGPT-medium", 
        description="Fallback model if primary fails"
    )
    
    # Available models for different use cases:
    # - "Qwen/Qwen2.5-7B-Instruct" (excellent quality)
    # - "microsoft/DialoGPT-medium" (conversational)
    # - "meta-llama/Llama-2-7b-chat-hf" (good general purpose)
    # - "mistralai/Mistral-7B-Instruct-v0.1" (fast and good)
    # - "google/flan-t5-large" (instruction following)
    
    # Rate Limiting (Conservative for free tier)
    hf_tier: str = Field(default="free", description="HF tier: free, pro, or enterprise")
    requests_per_minute: int = Field(default=12, description="Requests per minute")
    requests_per_hour: int = Field(default=800, description="Requests per hour")
    cooldown_seconds: float = Field(default=3.0, description="Cooldown between requests")
    
    # Generation Parameters
    max_new_tokens: int = Field(default=200, description="Maximum tokens to generate")
    temperature: float = Field(default=0.3, description="Generation temperature")
    top_p: float = Field(default=0.9, description="Top-p sampling")
    repetition_penalty: float = Field(default=1.1, description="Repetition penalty")
    
    # Application Configuration
    app_host: str = Field(default="0.0.0.0", description="Application host")
    app_port: int = Field(default=8000, description="Application port")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Caching and Optimization
    enable_response_cache: bool = Field(default=True, description="Enable response caching")
    cache_size: int = Field(default=500, description="Maximum cache entries")
    api_timeout: int = Field(default=30, description="API timeout in seconds")

# Initialize settings
settings = Settings()

# Rate Limiter
class HFRateLimiter:
    def __init__(self):
        self.minute_requests = deque()
        self.hour_requests = deque()
        self.last_request = 0
    
    async def can_make_request(self) -> tuple[bool, str]:
        now = time.time()
        
        # Clean old requests
        while self.minute_requests and now - self.minute_requests[0] > 60:
            self.minute_requests.popleft()
        while self.hour_requests and now - self.hour_requests[0] > 3600:
            self.hour_requests.popleft()
        
        # Check limits
        if len(self.minute_requests) >= settings.requests_per_minute:
            wait_time = 60 - (now - self.minute_requests[0])
            return False, f"minute_limit_exceeded_wait_{wait_time:.0f}s"
        
        if len(self.hour_requests) >= settings.requests_per_hour:
            return False, "hour_limit_exceeded"
        
        # Check cooldown
        if now - self.last_request < settings.cooldown_seconds:
            wait_time = settings.cooldown_seconds - (now - self.last_request)
            await asyncio.sleep(wait_time)
        
        # Record request
        current_time = time.time()
        self.minute_requests.append(current_time)
        self.hour_requests.append(current_time)
        self.last_request = current_time
        
        return True, "ok"
    
    def get_stats(self) -> Dict:
        now = time.time()
        # Clean for accurate stats
        while self.minute_requests and now - self.minute_requests[0] > 60:
            self.minute_requests.popleft()
        while self.hour_requests and now - self.hour_requests[0] > 3600:
            self.hour_requests.popleft()
        
        return {
            "requests_this_minute": len(self.minute_requests),
            "requests_this_hour": len(self.hour_requests),
            "minute_limit": settings.requests_per_minute,
            "hour_limit": settings.requests_per_hour,
            "minute_remaining": settings.requests_per_minute - len(self.minute_requests),
            "hour_remaining": settings.requests_per_hour - len(self.hour_requests)
        }

# Response Cache
class ResponseCache:
    def __init__(self, max_size: int = 500):
        self.cache = {}
        self.max_size = max_size
        self.access_order = deque()
    
    def _get_key(self, question: str) -> str:
        return hashlib.md5(question.lower().strip().encode()).hexdigest()[:16]
    
    def get(self, question: str) -> Optional[str]:
        key = self._get_key(question)
        if key in self.cache:
            # Update access order (LRU)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def set(self, question: str, response: str):
        key = self._get_key(question)
        
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = self.access_order.popleft()
            del self.cache[oldest_key]
        
        self.cache[key] = response
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)

# Global instances
rate_limiter = HFRateLimiter()
response_cache = ResponseCache(settings.cache_size) if settings.enable_response_cache else None

# Pydantic models
class FileAttachment(BaseModel):
    filename: str
    content: str  # base64 encoded
    content_type: str

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=800, description="The student's question")
    attachments: Optional[List[FileAttachment]] = Field(default=None, description="Optional file attachments")

class LinkResponse(BaseModel):
    url: str
    text: str

class AnswerResponse(BaseModel):
    answer: str
    links: List[LinkResponse] = Field(default_factory=list)
    source: str = Field(default="ai", description="Response source: ai, cache, or fallback")
    model_used: Optional[str] = Field(default=None, description="Model that generated the response")

# FastAPI app
app = FastAPI(
    title="TDS Virtual TA (Hugging Face API)",
    description="Virtual Teaching Assistant using Hugging Face Inference API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Course knowledge base
KNOWLEDGE_BASE = {
    "ai_model": {
        "keywords": ["model", "gpt", "ai", "openai", "which model", "what model"],
        "answer": "For TDS assignments, you **must use `gpt-3.5-turbo-0125`** (not gpt-4o-mini or other models). This is specified in the project requirements to ensure consistent results across all submissions.",
        "links": [
            {
                "url": "https://discourse.onlinedegree.iitm.ac.in/t/ai-model-requirements/123",
                "text": "Official AI model requirements for TDS assignments"
            }
        ]
    },
    "fastapi": {
        "keywords": ["fastapi", "api", "deploy", "deployment", "uvicorn", "railway"],
        "answer": "For FastAPI deployment on Railway/Render: 1) Create `requirements.txt` with dependencies 2) Use `uvicorn main:app --host 0.0.0.0 --port $PORT` 3) Set environment variables in platform dashboard 4) Add CORS middleware 5) Include health check endpoint",
        "links": [
            {
                "url": "https://discourse.onlinedegree.iitm.ac.in/t/fastapi-deployment/456",
                "text": "FastAPI deployment guide and best practices"
            }
        ]
    },
    "pydantic": {
        "keywords": ["pydantic", "validation", "field", "basemodel", "error"],
        "answer": "Common Pydantic solutions: 1) Use `Field(...)` for required fields 2) Check data types match your model 3) Use `Optional[Type]` for optional fields 4) Verify JSON structure matches model definition 5) Import `BaseModel` and `Field` correctly",
        "links": [
            {
                "url": "https://discourse.onlinedegree.iitm.ac.in/t/pydantic-errors/789",
                "text": "Common Pydantic validation errors and solutions"
            }
        ]
    }
}

async def call_huggingface_api(question: str, model_name: str = None) -> Optional[Dict]:
    """Call Hugging Face Inference API with rate limiting"""
    
    # Check if API key is configured
    if not settings.huggingface_api_key:
        print("❌ No Hugging Face API key configured")
        return None
    
    # Check rate limits
    can_proceed, status = await rate_limiter.can_make_request()
    if not can_proceed:
        print(f"🚦 Rate limit: {status}")
        return None
    
    model_to_use = model_name or settings.primary_model
    
    try:
        headers = {
            "Authorization": f"Bearer {settings.huggingface_api_key}",
            "Content-Type": "application/json"
        }
        
        # Create optimized prompt based on model type
        if "qwen" in model_to_use.lower():
            # Qwen chat format
            prompt = f"""<|im_start|>system
You are a helpful Teaching Assistant for the "Tools in Data Science" course at IIT Madras. Provide concise, practical answers to student questions about Python, FastAPI, Pydantic, Machine Learning, and related topics.

Key course requirements:
- Use gpt-3.5-turbo-0125 for AI assignments (not gpt-4o-mini)
- Include MIT LICENSE in GitHub repositories
- Follow Python best practices<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant"""
        
        elif "llama" in model_to_use.lower():
            # Llama chat format
            prompt = f"""<s>[INST] You are a helpful Teaching Assistant for a Data Science course. 

Student question: {question}

Provide a helpful, concise answer (2-3 sentences): [/INST]"""
        
        else:
            # Generic format for other models
            prompt = f"""You are a helpful Teaching Assistant for a Data Science course.

Student question: {question}

Provide a helpful, concise answer:"""
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": settings.max_new_tokens,
                "temperature": settings.temperature,
                "top_p": settings.top_p,
                "repetition_penalty": settings.repetition_penalty,
                "do_sample": True,
                "return_full_text": False
            },
            "options": {
                "wait_for_model": True,
                "use_cache": False
            }
        }
        
        url = f"https://api-inference.huggingface.co/models/{model_to_use}"
        
        async with httpx.AsyncClient(timeout=settings.api_timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get("generated_text", "").strip()
                    if generated_text:
                        print(f"✅ Got response from {model_to_use}")
                        return {
                            "answer": generated_text,
                            "model": model_to_use,
                            "source": "ai"
                        }
            
            elif response.status_code == 429:
                print(f"⚠️ API rate limited for {model_to_use}")
            elif response.status_code == 503:
                print(f"⚠️ Model {model_to_use} loading, try again later")
            else:
                print(f"❌ API error {response.status_code} for {model_to_use}")
                
    except httpx.TimeoutException:
        print(f"⏰ Timeout calling {model_to_use}")
    except Exception as e:
        print(f"❌ Error calling {model_to_use}: {e}")
    
    return None

def generate_rule_based_response(question: str) -> Dict:
    """Generate rule-based response using knowledge base"""
    question_lower = question.lower()
    
    # Find best match
    best_match = None
    best_score = 0
    
    for category, data in KNOWLEDGE_BASE.items():
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
            "source": "rule_based"
        }
    
    # Default response
    default_answers = [
        "I'm here to help with your TDS course questions! For specific technical issues, please provide more details about your problem.",
        "That's an interesting question! For the most accurate help, could you share more context about what you're trying to achieve?",
        "I can help with Python, FastAPI, Pydantic, Machine Learning, and other TDS topics. Please be more specific about your question.",
    ]
    
    return {
        "answer": random.choice(default_answers),
        "links": [LinkResponse(
            url="https://discourse.onlinedegree.iitm.ac.in/c/tools-data-science",
            text="TDS Discourse forum for detailed discussions"
        )],
        "source": "fallback"
    }

async def get_smart_response(question: str) -> Dict:
    """Get response using smart fallback strategy"""
    
    # Check cache first
    if response_cache:
        cached_response = response_cache.get(question)
        if cached_response:
            print("📋 Using cached response")
            return {
                "answer": cached_response,
                "links": [],
                "source": "cache"
            }
    
    # Try AI only if API key is configured
    if settings.huggingface_api_key:
        # Try primary model
        ai_response = await call_huggingface_api(question, settings.primary_model)
        if ai_response:
            # Cache successful response
            if response_cache:
                response_cache.set(question, ai_response["answer"])
            return ai_response
        
        # Try fallback model
        print(f"🔄 Trying fallback model: {settings.fallback_model}")
        ai_response = await call_huggingface_api(question, settings.fallback_model)
        if ai_response:
            if response_cache:
                response_cache.set(question, ai_response["answer"])
            return ai_response
    else:
        print("⚠️ No API key configured, using rule-based responses only")
    
    # Use rule-based response
    print("🔄 Using rule-based response")
    return generate_rule_based_response(question)

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    print("🚀 TDS Virtual TA with Hugging Face API starting...")
    
    # Check configuration
    if settings.huggingface_api_key:
        print(f"🤖 Primary model: {settings.primary_model}")
        print(f"🔄 Fallback model: {settings.fallback_model}")
        print(f"🚦 Rate limits: {settings.requests_per_minute}/min, {settings.requests_per_hour}/hour")
    else:
        print("⚠️  WARNING: No Hugging Face API key configured!")
        print("🔄 Running in rule-based mode only")
    
    print(f"📋 Caching: {'enabled' if settings.enable_response_cache else 'disabled'}")
    print("✅ Service started successfully!")

@app.get("/")
async def root():
    """Root endpoint"""
    health = get_service_health()
    
    return {
        "message": "TDS Virtual TA with Hugging Face API",
        "status": health["status"],
        "version": "1.0.0",
        "ai_provider": "Hugging Face Inference API" if settings.huggingface_api_key else "Rule-based only",
        "primary_model": settings.primary_model if settings.huggingface_api_key else None,
        "fallback_model": settings.fallback_model if settings.huggingface_api_key else None,
        "capabilities": health["capabilities"],
        "issues": health["issues"] if health["issues"] else None,
        "warnings": health["warnings"] if health["warnings"] else None,
        "rate_limiting": {
            "tier": settings.hf_tier,
            "requests_per_minute": settings.requests_per_minute,
            "requests_per_hour": settings.requests_per_hour
        } if settings.huggingface_api_key else None,
        "features": {
            "response_caching": settings.enable_response_cache,
            "smart_fallback": True,
            "rule_based_backup": True
        },
        "endpoints": {
            "answer": "/api/",
            "health": "/health",
            "rate_status": "/rate-status",
            "docs": "/docs"
        }
    }

@app.post("/api/", response_model=AnswerResponse)
async def answer_question(request: QuestionRequest):
    """Main API endpoint to answer student questions"""
    try:
        # Get smart response (tries AI, falls back to rules)
        response_data = await get_smart_response(request.question)
        
        return AnswerResponse(
            answer=response_data["answer"],
            links=response_data.get("links", []),
            source=response_data.get("source", "unknown"),
            model_used=response_data.get("model")
        )
        
    except Exception as e:
        # Emergency fallback
        emergency_response = generate_rule_based_response(request.question)
        return AnswerResponse(
            answer=f"{emergency_response['answer']}\n\n*Note: Experienced a technical issue but provided a helpful response based on course knowledge.*",
            links=emergency_response.get("links", []),
            source="emergency_fallback"
        )

@app.get("/rate-status")
async def get_rate_status():
    """Get current rate limiting status"""
    stats = rate_limiter.get_stats()
    
    return {
        "rate_limiting": stats,
        "tier": settings.hf_tier,
        "recommendations": {
            "can_make_request": stats["minute_remaining"] > 0 and stats["hour_remaining"] > 0,
            "approaching_minute_limit": stats["minute_remaining"] < 3,
            "approaching_hour_limit": stats["hour_remaining"] < 50,
            "suggested_wait_minutes": max(0, (60 - stats["minute_remaining"])) if stats["minute_remaining"] == 0 else 0
        },
        "cache_stats": {
            "enabled": settings.enable_response_cache,
            "size": len(response_cache.cache) if response_cache else 0,
            "max_size": settings.cache_size
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health = get_service_health()
    stats = rate_limiter.get_stats()
    
    return {
        "status": health["status"],
        "timestamp": datetime.now().isoformat(),
        "issues": health["issues"],
        "warnings": health["warnings"],
        "ai_provider": "Hugging Face Inference API" if settings.huggingface_api_key else "Rule-based only",
        "configuration": {
            "api_key_configured": bool(settings.huggingface_api_key),
            "primary_model": settings.primary_model if settings.huggingface_api_key else None,
            "fallback_model": settings.fallback_model if settings.huggingface_api_key else None
        },
        "rate_limiting": {
            "minute_remaining": stats["minute_remaining"],
            "hour_remaining": stats["hour_remaining"],
            "healthy": stats["minute_remaining"] > 0 and stats["hour_remaining"] > 0
        } if settings.huggingface_api_key else {
            "status": "not_applicable",
            "reason": "no_api_key"
        },
        "caching": {
            "enabled": settings.enable_response_cache,
            "entries": len(response_cache.cache) if response_cache else 0
        },
        "capabilities": health["capabilities"],
        "fallback_systems": {
            "rule_based": True,
            "emergency_response": True
        },
        "setup_instructions": [
            "Set HUGGINGFACE_API_KEY environment variable",
            "Get API key from: https://huggingface.co/settings/tokens",
            "Restart the service after setting the key"
        ] if not settings.huggingface_api_key else None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug
    )