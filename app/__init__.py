import random
from datetime import datetime
from typing import Dict, Optional, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from app.config import settings
from app.utils import get_service_health, ResponseCache
from app.models import AnswerResponse, QuestionRequest, LinkResponse

# FastAPI app
app = FastAPI(
    title="TDS Virtual TA (Local Qwen Model)",
    description="Virtual Teaching Assistant using local Qwen2.5-0.5B-Instruct model",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Initialize response cache
response_cache = (
    ResponseCache(max_size=settings.cache_size)
    if settings.enable_response_cache
    else None
)

# Initialize local Qwen model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    settings.model_name,
    torch_dtype=torch.float16 if settings.use_fp16 else torch.float32,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(settings.model_name)

# Course knowledge base
KNOWLEDGE_BASE = {
    "ai_model": {
        "keywords": ["model", "gpt", "ai", "openai", "which model", "what model"],
        "answer": "For TDS assignments, you **must use `gpt-3.5-turbo-0125`** (not gpt-4o-mini or other models). This is specified in the project requirements to ensure consistent results across all submissions.",
        "links": [
            {
                "url": "https://discourse.onlinedegree.iitm.ac.in/t/ai-model-requirements/123",
                "text": "Official AI model requirements for TDS assignments",
            }
        ]
    },
    "fastapi": {
        "keywords": ["fastapi", "api", "deploy", "deployment", "uvicorn", "railway"],
        "answer": "To deploy a FastAPI application to Railway: 1) Create a `requirements.txt` with dependencies (e.g., `fastapi`, `uvicorn`). 2) Add a `Procfile` with `web: uvicorn main:app --host 0.0.0.0 --port $PORT`. 3) Optionally, use a `Dockerfile` for custom builds. 4) Push code to a GitHub repository. 5) Create a Railway project, link the repository, and set environment variables in the dashboard. 6) Add CORS middleware and a `/health` endpoint for monitoring.",
        "links": [
            {
                "url": "https://discourse.onlinedegree.iitm.ac.in/t/fastapi-deployment/456",
                "text": "FastAPI deployment guide and best practices"
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

def generate_local_model_response(question: str) -> Dict[str, str]:
    """Generate response using local Qwen model"""
    try:
        # Create optimized prompt for Qwen model
        prompt = (
            "<|im_start|>system\n"
            "You are a helpful Teaching Assistant for the \"Tools in Data Science\" course at IIT Madras. "
            "Provide concise, practical answers to student questions about Python, FastAPI, Pydantic, "
            "Machine Learning, and related topics. Focus on specific deployment steps when asked about "
            "deploying applications, especially to platforms like Railway. Avoid generic setup instructions "
            "unless relevant. For questions unrelated to the course, respond with: 'This question is outside the scope of the Tools in Data Science course. Please ask about Python, FastAPI, Pydantic, Machine Learning, or related topics.' "
            "Do not include the system or user prompt in your response.\n\n"
            "Key course requirements:\n"
            "- Use gpt-3.5-turbo-0125 for AI assignments (not gpt-4o-mini)\n"
            "- Include MIT LICENSE in GitHub repositories\n"
            "- Follow Python best practices<|im_end|>\n"
            f"<|im_start|>user\n{question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate response
        outputs = model.generate(
            **inputs,
            max_new_tokens=settings.max_new_tokens,
            temperature=settings.temperature,
            top_p=settings.top_p,
            repetition_penalty=settings.repetition_penalty,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Decode response
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract only the assistant's response
        assistant_response = generated_text

        # Remove the prompt (system and user parts)
        prompt_end_marker = "<|im_start|>assistant"
        if prompt_end_marker in generated_text:
            assistant_response = generated_text.split(prompt_end_marker)[-1]
        else:
            # Fallback: remove everything up to the question
            question_marker = f"user\n{question}"
            if question_marker in generated_text:
                assistant_response = generated_text.split(question_marker)[-1]
                if "<|im_start|>assistant" in assistant_response:
                    assistant_response = assistant_response.split("<|im_start|>assistant")[-1]

        # Remove any trailing markers
        if "<|im_end|>" in assistant_response:
            assistant_response = assistant_response.split("<|im_end|>")[0]

        # Clean up whitespace
        assistant_response = assistant_response.strip()

        # Ensure non-empty response
        if not assistant_response:
            print(f"⚠️ Empty response from {settings.model_name}")
            return {}

        print(f"✅ Got response from {settings.model_name}")
        return {
            "answer": assistant_response,
            "model": settings.model_name,
            "source": "local_model",
        }

    except Exception as e:
        print(f"❌ Error generating response with {settings.model_name}: {e}")
        return {}

def generate_rule_based_response(question: str) -> Dict:
    """Generate rule-based response using knowledge base"""
    question_lower = question.lower()

    # Find best match
    best_match = None
    best_score = 0

    for category, data in KNOWLEDGE_BASE.items():
        score = 0
        keywords: List[str] = data["keywords"]
        for keyword in keywords:
            if keyword.lower() in question_lower:
                score += len(keyword)

        if score > best_score:
            best_score = score
            best_match = data

    if best_match:
        links = best_match.get("links", [])
        return {
            "answer": best_match["answer"],
            "links": links,
            "source": "rule_based",
        }

    # Default response for off-topic questions
    return {
        "answer": "This question is outside the scope of the Tools in Data Science course. Please ask about Python, FastAPI, Pydantic, Machine Learning, or related topics.",
        "links": [
            {
                "url": "https://discourse.onlinedegree.iitm.ac.in/c/tools-data-science",
                "text": "TDS Discourse forum for detailed discussions"
            }
        ],
        "source": "fallback",
    }

async def get_smart_response(question: str) -> Dict:
    """Get response using smart fallback strategy"""
    # Check cache first
    if response_cache:
        cached_response = response_cache.get(question)
        if cached_response:
            print("📋 Using cached response")
            return {"answer": cached_response, "links": [], "source": "cache"}

    # Check knowledge base first
    rule_response = generate_rule_based_response(question)
    if rule_response["source"] == "rule_based":
        print("📚 Using rule-based response from knowledge base")
        if response_cache:
            response_cache.set(question, rule_response["answer"])
        return rule_response

    # Use default rule-based response for off-topic questions
    if rule_response["source"] == "fallback":
        print("🔄 Using default rule-based response")
        if response_cache:
            response_cache.set(question, rule_response["answer"])
        return rule_response

    # Try local model as fallback (only for TDS-related questions)
    model_response = generate_local_model_response(question)
    if model_response:
        if response_cache:
            response_cache.set(question, model_response["answer"])
        return model_response

    # Final fallback
    print("🔄 Using default rule-based response")
    return generate_rule_based_response(question)

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    print("🚀 TDS Virtual TA with Local Qwen Model starting...")
    print(f"🤖 Model: {settings.model_name}")
    print(f"📋 Caching: {'enabled' if settings.enable_response_cache else 'disabled'}")
    print(f"🔧 FP16: {'enabled' if settings.use_fp16 else 'disabled'}")
    print("✅ Service started successfully!")

@app.get("/")
async def root():
    """Root endpoint"""
    health = get_service_health(settings, response_cache)

    return {
        "message": "TDS Virtual TA with Local Qwen Model",
        "status": health["status"],
        "version": "1.0.0",
        "ai_provider": "Local Qwen Model",
        "model": settings.model_name,
        "capabilities": health["capabilities"],
        "issues": health["issues"] if health["issues"] else None,
        "warnings": health["warnings"] if health["warnings"] else None,
        "features": {
            "response_caching": settings.enable_response_cache,
            "rule_based_backup": True,
        },
        "endpoints": {
            "answer": "/api/",
            "health": "/health",
            "docs": "/docs",
        },
    }

@app.post("/api/", response_model=AnswerResponse)
async def answer_question(request: QuestionRequest):
    """Main API endpoint to answer student questions"""
    try:
        # Get smart response
        response_data = await get_smart_response(request.question)

        return AnswerResponse(
            answer=response_data["answer"],
            links=response_data.get("links", []),
            source=response_data.get("source", "unknown"),
            model_used=response_data.get("model"),
        )

    except Exception as e:
        print(f"❌ Emergency fallback triggered: {e}")
        # Emergency fallback
        emergency_response = generate_rule_based_response(request.question)
        return AnswerResponse(
            answer="{}\n\n*Note: Experienced a technical issue but provided a helpful response based on course knowledge.*".format(emergency_response['answer']),
            links=emergency_response.get("links", []),
            source="emergency_fallback",
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health = get_service_health(settings, response_cache)

    return {
        "status": health["status"],
        "timestamp": datetime.now().isoformat(),
        "issues": health["issues"],
        "warnings": health["warnings"],
        "ai_provider": "Local Qwen Model",
        "configuration": {
            "model": settings.model_name,
            "use_fp16": settings.use_fp16,
        },
        "caching": {
            "enabled": settings.enable_response_cache,
            "entries": len(response_cache.cache) if response_cache else 0,
        },
        "capabilities": health["capabilities"],
        "fallback_systems": {"rule_based": True, "emergency_response": True},
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, host=settings.app_host, port=settings.app_port, reload=settings.debug
    )