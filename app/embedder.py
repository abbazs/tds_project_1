import asyncio
import time
from collections import deque

from openai import AsyncOpenAI

from app.utils import print_error
from app.utils import print_warning


class OpenAIEmbedder:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.request_times = deque(maxlen=60)

    async def _rate_limit(self) -> None:
        """Rate limiting for OpenAI paid tier (3000 RPM = 50 RPS)"""
        now = time.time()

        if len(self.request_times) >= 49:  # Stay under 50 RPS
            wait_time = max(0, 1 - (now - self.request_times[0]))
            if wait_time > 0:
                print_warning("\t\t\t\tWaiting for rate limit reset...")
                await asyncio.sleep(wait_time)

        self.request_times.append(now)

    async def embed_text(self, text: str) -> list[float]:
        """Embed text using OpenAI with rate limiting and retries"""
        for attempt in range(3):
            try:
                await self._rate_limit()
                response = await self.client.embeddings.create(
                    model=self.model, input=text
                )
                return response.data[0].embedding

            except Exception as e:
                error_msg = str(e).lower()
                if "429" in error_msg or "rate_limit" in error_msg:
                    wait_time = 15 + (attempt * 10)  # Shorter waits for paid tier
                    print_warning(
                        f"Rate limited (attempt {attempt + 1}/3), waiting {wait_time}s"
                    )
                    await asyncio.sleep(wait_time)
                elif attempt < 2:
                    await asyncio.sleep(2 + (attempt * 2))  # Faster retries
                else:
                    print_error(f"Embedding failed: {str(e)[:80]}")
                    return []
        return []
