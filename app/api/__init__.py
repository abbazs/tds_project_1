import logging
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic import Field

from app.concise_answer import OpenAIConciseAnswer
from app.embedder import OpenAIEmbedder
from app.image_context import OpenAIImageAnalyzer
from app.models import QuestionResponse
from app.models import Settings

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("tds_project_1")

EMB = Path("pweighted.npz")
logger.info(f"Loading embeddings from {EMB}")
if not EMB.exists():
    raise FileNotFoundError(
        f"Embeddings file not found: {EMB}. Please run the embedding script first."
    )
_data = np.load(EMB, allow_pickle=True)
_corpus_emb = _data["embeddings"]  # shape: (N, D)
_texts = _data["texts"]  # shape: (N,)
_urls = _data["urls"]

# pre‐compute norms for faster cosine
_corpus_norm = np.linalg.norm(_corpus_emb, axis=1)


def top_k_matches(query_emb: np.ndarray, k: int = 10):
    # cosine sim = (A⋅B) / (||A|| * ||B||)
    q_norm = np.linalg.norm(query_emb)
    sims = (_corpus_emb @ query_emb) / (_corpus_norm * q_norm)
    idxs = np.argsort(sims)[::-1][:k]
    return [{"url": _urls[i], "text": _texts[i]} for i in idxs]


app = FastAPI(title="TDS May 2025 Project 1 Q&A API", version="1.0.0")

settings = Settings()  # type: ignore
# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        examples=["Should I use gpt-4o-mini which AI proxy supports, or gpt3.5 turbo?"],
    )
    image: Optional[str] = Field(
        None,
        description="Base64 encoded image",
        examples=[
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        ],
    )


async def process_question(request: QuestionRequest) -> QuestionResponse:
    """Process student question and return structured response."""

    if request.image:
        aii = OpenAIImageAnalyzer(api_key=settings.api_key)
        context = await aii.analyze_image(request.image)
        question = f"{request.question} {context}"
    else:
        question = request.question
    aie = OpenAIEmbedder(api_key=settings.api_key)
    embeddings = await aie.embed_text(question.strip())
    if not embeddings:
        return QuestionResponse(
            answer="No relevant information found.",
            links=[],
        )
    # get top 10 similar passages
    matches = top_k_matches(embeddings, 10)
    aic = OpenAIConciseAnswer(api_key=settings.api_key)
    resp = await aic.answer(
        question=question,
        matches=matches,
    )
    return resp


@app.post("/api/", response_model=QuestionResponse)
async def answer_question(request: QuestionRequest):
    """
    Handle student questions with optional image attachments.

    Example request:
    ```json
    {
        "question": "Should I use gpt-4o-mini which AI proxy supports, or gpt3.5 turbo?",
        "image": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
    }
    ```
    """
    try:
        logger.info(f"Received question: {request}")
        return await process_question(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
