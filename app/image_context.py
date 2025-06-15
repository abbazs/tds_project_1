import asyncio
import time
from collections import deque

from openai import AsyncOpenAI
from rich.console import Console

console = Console()


class OpenAIImageAnalyzer:
    def __init__(self, api_key: str, model="gpt-4o-mini", rpm_limit: int = 500):
        """Initialize OpenAIImageAnalyzer with OpenAI API key and rate limit."""
        self.rmp = rpm_limit
        self.client = AsyncOpenAI(api_key=api_key)
        self.semaphore = asyncio.Semaphore(30)
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

    async def analyze_image(self, image_b64: str) -> str | None:
        """Analyze image with OpenAI gpt-4o-mini (cost-effective vision model)."""
        async with self.semaphore:
            for attempt in range(3):
                try:
                    await self._rate_limit()

                    # Prepare image URL format for OpenAI
                    image_url = (
                        image_b64
                        if image_b64.startswith("data:image/")
                        else f"data:image/jpeg;base64,{image_b64.split(',')[-1]}"
                    )

                    response = await self.client.chat.completions.create(
                        model=self.model,  # Most cost-effective vision model
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": "Describe this image in 2-3 sentences: main subject, context/setting, and any important text or technical details.",
                                    },
                                    {
                                        "type": "image_url",
                                        "image_url": {"url": image_url},
                                    },
                                ],
                            }
                        ],
                        max_tokens=150,  # Keep tokens low for cost efficiency
                        temperature=0.1,  # Consistent responses
                    )

                    return response.choices[0].message.content.strip()

                except Exception as e:
                    error_msg = str(e).lower()

                    if "rate_limit" in error_msg or "429" in error_msg:
                        wait_time = 10 + (attempt * 5)
                        console.print(
                            f"[yellow]⚠️ Rate limited (attempt {attempt + 1}/3), waiting {wait_time}s[/yellow]"
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    elif "400" in error_msg or "invalid" in error_msg:
                        return None
                    else:
                        if attempt < 2:
                            await asyncio.sleep(5 + (attempt * 2))
                            continue
                        console.print(
                            f"[red]❌ AI analysis failed: {str(e)[:80]}[/red]"
                        )
                        return None
            return None
