import asyncio
import time
from collections import deque
from typing import Any

from openai import AsyncOpenAI
from rich.console import Console

from app.models import LinkResponse
from app.models import QuestionResponse

console = Console()


class OpenAIConciseAnswer:
    def __init__(self, api_key: str, model="gpt-4o-mini", rpm_limit: int = 500):
        """Initialize OpenAIConciseAnswer with OpenAI API key and rate limit."""
        self.rmp = rpm_limit
        self.client = AsyncOpenAI(api_key=api_key)
        self.semaphore = asyncio.Semaphore(5)
        self.request_times = deque(maxlen=rpm_limit)
        self.model = model

    async def _rate_limit(self) -> None:
        """Rate limiting for OpenAI paid tier."""
        now = time.time()

        # Remove requests older than 60 seconds
        while self.request_times and now - self.request_times[0] > 60:
            self.request_times.popleft()

        # Check if we're at the limit within the current minute
        if len(self.request_times) >= self.rmp:
            wait_time = 60 - (now - self.request_times[0]) + 0.1
            console.print(f"\n[yellow]⏳ Rate limit: waiting {wait_time:.1f}s[/yellow]")
            await asyncio.sleep(wait_time)
            # Clean up again after waiting
            while self.request_times and time.time() - self.request_times[0] > 60:
                self.request_times.popleft()

        self.request_times.append(now)

    async def answer(self, question: str, matches: list[Any]) -> QuestionResponse:
        """Provide concise answer based on matching passages."""
        async with self.semaphore:
            for attempt in range(3):
                try:
                    await self._rate_limit()

                    # Build context from matchings
                    context = "\n".join(
                        [
                            f"Source {i+1}: {match["text"].strip()}"
                            for i, match in enumerate(matches)
                        ]
                    )

                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "system",
                                "content": """You are a helpful assistant that extracts concrete, definitive answers from provided context.

                                
PRIORITY ORDER for information:
1. Content prefixed with [EXPERT] is of highest priority
2. Explicit statements about how things actually work
3. Confirmed outcomes/results
4. Official clarifications and explanations
5. Factual statements about what happens/happened

IGNORE:
- Questions asking "how would it look?" or "will it be?"
- Speculation or assumptions
- Hypothetical scenarios ("if someone gets...")

ANSWER RULES:
- Use the EXACT format/numbers mentioned in definitive statements
- Don't convert between formats
- Focus on what IS stated to happen, not what MIGHT happen

Only say 'Information not found in provided sources.' if the context is completely unrelated to the question.
Keep responses under 100 words using only confirmed facts.""",
                            },
                            {
                                "role": "user",
                                "content": f"Question: {question}\n\nContext:\n{context}\n\nAnswer:",
                            },
                        ],
                        max_tokens=150,
                        temperature=0.1,
                    )

                    answer_text = response.choices[0].message.content.strip()

                    # Return new QuestionResponse with answer
                    return QuestionResponse(
                        answer=answer_text,
                        links=matches,
                    )

                except Exception as e:
                    error_msg = str(e).lower()

                    if "rate_limit" in error_msg or "429" in error_msg:
                        wait_time = 10 + (attempt * 5)
                        console.print(
                            f"[yellow]⚠️ Rate limited (attempt {attempt + 1}/3), waiting {wait_time}s[/yellow]"
                        )
                        await asyncio.sleep(wait_time)
                    elif "400" in error_msg:
                        return None
                    elif attempt < 2:
                        await asyncio.sleep(5 + (attempt * 2))
                    else:
                        console.print(
                            f"[red]❌ AI analysis failed: {str(e)[:80]}[/red]"
                        )
                        return None

            return None
