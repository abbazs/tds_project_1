import base64
from typing import Optional

import uvicorn
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic import Field
from app.image_confext import OpenAIImageAnalyzer
from app.embedder import OpenAIEmbedder
from app.models import Settings
app = FastAPI(title="TDS May 2025 Project 1 Q&A API", version="1.0.0")

settings=Settings() # type: ignore
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


class LinkResponse(BaseModel):
    url: str
    text: str


class QuestionResponse(BaseModel):
    answer: str
    links: list[LinkResponse]


async def process_question(
    question: str, image_data: Optional[bytes] = None
) -> QuestionResponse:
    """Process student question and return structured response."""

    if image_data:
        aii=OpenAIImageAnalyzer(api_key=settings.api_key)
        context = aii.analyze_image(image_data.decode('utf-8'))

    # Example processing logic - replace with your actual implementation
    if "gpt" in question.lower() and "proxy" in question.lower():
        return QuestionResponse(
            answer="You must use `gpt-3.5-turbo-0125`, even if the AI Proxy only supports `gpt-4o-mini`. Use the OpenAI API directly for this question.",
            links=[
                LinkResponse(
                    url="https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939/4",
                    text="Use the model that's mentioned in the question.",
                ),
                LinkResponse(
                    url="https://discourse.onlinedegree.iitm.ac.in/t/ga5-question-8-clarification/155939/3",
                    text="My understanding is that you just have to use a tokenizer, similar to what Prof. Anand used, to get the number of tokens and multiply that by the given rate.",
                ),
            ],
        )

    # Default response for other questions
    return QuestionResponse(
        answer="I'll help you with your question. Please provide more specific details if needed.",
        links=[],
    )


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

    image_data = None
    if request.image:
        try:
            image_data = base64.b64decode(request.image)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 image data")

    try:
        return process_question(request.question, image_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
