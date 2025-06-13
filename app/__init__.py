"""
TDS Virtual TA - AI Teaching Assistant for Tools in Data Science
Main FastAPI Application
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi import File
from fastapi import HTTPException
from fastapi import UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import Settings
from .models import AnswerResponse
from .models import HealthResponse
from .models import Link
from .models import QuestionRequest
from .utils import AIpipeClient
from .utils import HealthChecker
from .utils import ResponseCache
from .utils import RuleBasedEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize settings and components
settings = Settings()
cache = (
    ResponseCache(max_size=settings.cache_max_size)
    if settings.enable_cache
    else None
)
rule_engine = RuleBasedEngine()
health_checker = HealthChecker()
aipipe_client = AIpipeClient(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    logger.info("🚀 Starting TDS Virtual TA...")
    logger.info("📚 Loading knowledge base...")
    rule_engine.load_knowledge_base()
    logger.info("✅ TDS Virtual TA initialized successfully!")

    yield  # Application runs here

    # Shutdown logic
    logger.info("🛑 Shutting down TDS Virtual TA...")
    await aipipe_client.client.aclose()
    logger.info("✅ Cleanup completed!")


# Create FastAPI app
app = FastAPI(
    title="TDS Virtual TA",
    description="AI Teaching Assistant for Tools in Data Science course at IIT Madras",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Add CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with basic info and version"""
    return {
        "name": "TDS Virtual TA",
        "version": "1.0.0",
        "description": "AI Teaching Assistant for Tools in Data Science course at IIT Madras",
        "status": "running 🚀",
        "endpoints": {
            "health": "/health",
            "api": "/api/",
            "docs": "/docs",
            "redoc": "/redoc",
        },
        "course": "Tools in Data Science - IIT Madras",
        "features": [
            "Smart Q&A with AI models",
            "Rule-based knowledge matching",
            "Response caching",
            "File attachment support",
        ],
    }


@app.post("/api/", response_model=AnswerResponse)
async def ask_question(
    request: QuestionRequest, files: list[UploadFile] | None = File(None)
):
    """
    Main API endpoint for student questions
    Implements smart response logic with caching and fallbacks
    """
    start_time = time.time()
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    logger.info(f"📝 Received question: {question[:100]}...")

    try:
        # Step 1: Check cache first
        if cache and settings.enable_cache:
            cached_response = cache.get(question)
            if cached_response:
                logger.info("💾 Returning cached response")
                return cached_response

        # Step 2: Try rule-based matching
        rule_response = rule_engine.match_question(question)
        if rule_response:
            logger.info("📚 Using rule-based response")
            response = AnswerResponse(
                answer=rule_response["answer"],
                links=[Link(**link) for link in rule_response.get("links", [])],
                source="rule_based",
                model_name="TDS Knowledge Base",
                processing_time=time.time() - start_time,
                metadata={
                    "matched_topic": rule_response.get("topic"),
                    "confidence": rule_response.get("confidence", 0.95),
                },
            )
        else:
            # Step 3: Generate response using AIpipe
            logger.info("🤖 Generating AI response via AIpipe...")
            ai_response = await aipipe_client.generate_response(question, files)

            response = AnswerResponse(
                answer=ai_response["answer"],
                links=[Link(**link) for link in ai_response.get("links", [])],
                source="ai_model",
                model_name=ai_response.get("model_name", settings.model_name),
                processing_time=time.time() - start_time,
                metadata=ai_response.get("metadata", {}),
            )

        # Cache the response
        if cache and settings.enable_cache:
            cache.put(question, response)

        logger.info(f"✅ Response generated in {response.processing_time:.2f}s")
        return response

    except Exception as e:
        logger.error(f"❌ Error processing question: {str(e)}")

        # Fallback to default message
        return AnswerResponse(
            answer=(
                "I apologize, but I'm experiencing technical difficulties right now. "
                "Here are some general resources for the TDS course:\n\n"
                "• Check the course materials on the LMS\n"
                "• Review the lecture slides and notebooks\n"
                "• Post your question on the course forum\n"
                "• Attend office hours for personalized help\n\n"
                "Please try asking your question again in a few moments."
            ),
            links=[
                Link(
                    url="https://courses.iitm.ac.in",
                    text="IIT Madras Course Portal",
                ),
                Link(
                    url="https://github.com/sanand0/aipipe",
                    text="AIpipe Documentation",
                ),
            ],
            source="fallback",
            model_name="TDS Virtual TA",
            processing_time=time.time() - start_time,
            metadata={"error": str(e), "fallback": True},
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with diagnostics"""
    logger.info("🏥 Health check requested")

    health_data = health_checker.get_health_status()

    # Check AIpipe connectivity
    try:
        aipipe_status = await aipipe_client.check_health()
        health_data["aipipe_status"] = aipipe_status
    except Exception as e:
        health_data["aipipe_status"] = {"status": "error", "error": str(e)}

    # Add cache statistics
    if cache:
        health_data["cache_stats"] = cache.get_stats()

    # Determine overall status
    status = "healthy"
    if health_data["aipipe_status"].get("status") != "ok":
        status = "degraded"

    return HealthResponse(
        status=status,
        version="1.0.0",
        uptime=health_data["uptime"],
        capabilities=health_data["capabilities"],
        diagnostics=health_data,
        warnings=health_data.get("warnings", []),
        model_info={
            "name": settings.model_name,
            "temperature": settings.temperature,
            "max_tokens": settings.max_new_tokens,
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level="info",
    )
