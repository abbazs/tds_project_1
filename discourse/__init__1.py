#!/usr/bin/env python3.12
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "httpx>=0.27.0",
#     "pydantic>=2.8.0",
#     "rich-click>=1.8.0",
#     "rich>=13.0.0",
#     "markdownify>=0.11.6",
# ]
# ///
"""Ultra-efficient Discourse topic scraper with async processing."""

import asyncio
import base64
import json
import re
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx
import rich_click as click
from markdownify import markdownify as md
from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError
from rich.console import Console
from rich.progress import BarColumn
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TaskProgressColumn
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn

console = Console()


class CookieConfig(BaseModel):
    t_token: str = Field(alias="_t")
    forum_session: str = Field(alias="_forum_session")


class Post(BaseModel):
    username: str
    created_at: datetime
    content: str
    url: str
    image: str | None = None


class Topic(BaseModel):
    id: int
    title: str
    created_at: datetime
    last_posted_at: datetime | None = None
    posts_count: int
    views: int
    category_id: int
    url: str = ""
    posts: list[Post] = Field(default_factory=list)


class ScrapingConfig(BaseModel):
    cookies: CookieConfig
    start_date: str = "2025-01-01"
    end_date: str = "2025-04-15"
    output_dir: str = "discourse_data"
    category_id: int = 34


class DiscourseClient:
    def __init__(self, base_url: str, cookie_config: CookieConfig):
        self.base_url = base_url.rstrip("/")
        self.session = httpx.AsyncClient(
            cookies={
                "_t": cookie_config.t_token,
                "_forum_session": cookie_config.forum_session,
            },
            timeout=30.0,
            limits=httpx.Limits(max_connections=5, max_keepalive_connections=2),
        )
        self.rate_limit_delay = 0.5

    async def _request_with_retry(self, url: str, **kwargs) -> dict[str, Any]:
        """Request with exponential backoff."""
        for attempt in range(3):
            try:
                await asyncio.sleep(self.rate_limit_delay)
                response = await self.session.get(url, **kwargs)

                if response.status_code == 429:
                    self.rate_limit_delay *= 2
                    wait_time = min(self.rate_limit_delay, 10.0)
                    console.print(
                        f"[yellow]Rate limited, waiting {wait_time:.1f}s[/yellow]"
                    )
                    await asyncio.sleep(wait_time)
                    continue

                if response.status_code == 403:
                    raise click.ClickException(
                        "Authentication failed - check cookies"
                    )

                response.raise_for_status()
                self.rate_limit_delay = max(0.5, self.rate_limit_delay * 0.9)
                return response.json()

            except httpx.HTTPStatusError as e:
                if attempt == 2:
                    raise click.ClickException(
                        f"HTTP {e.response.status_code}: {e.response.text[:200]}"
                    )
                await asyncio.sleep(2**attempt)

        raise click.ClickException("Max retries exceeded")

    async def get_category_topics(
        self, category_id: int, page: int = 0
    ) -> list[Topic]:
        """Get topics from category with modern endpoints."""
        # Try multiple endpoint patterns for compatibility
        endpoints = [
            (
                f"{self.base_url}/c/{category_id}/l/latest.json",
                {"page": page} if page > 0 else {},
            ),
            (
                f"{self.base_url}/latest.json",
                (
                    {"category": category_id, "page": page}
                    if page > 0
                    else {"category": category_id}
                ),
            ),
        ]

        for url, params in endpoints:
            try:
                data = await self._request_with_retry(url, params=params)
                break
            except click.ClickException:
                continue
        else:
            raise click.ClickException("Failed to access category topics")

        topics = []
        for topic_data in data.get("topic_list", {}).get("topics", []):
            topic_id = topic_data["id"]
            topics.append(
                Topic(
                    id=topic_id,
                    title=topic_data["title"],
                    created_at=datetime.fromisoformat(
                        topic_data["created_at"].replace("Z", "+00:00")
                    ),
                    last_posted_at=(
                        datetime.fromisoformat(
                            topic_data["last_posted_at"].replace("Z", "+00:00")
                        )
                        if topic_data.get("last_posted_at")
                        else None
                    ),
                    posts_count=topic_data["posts_count"],
                    views=topic_data["views"],
                    category_id=topic_data["category_id"],
                    url=f"{self.base_url}/t/{topic_id}",
                )
            )
        return topics

    async def get_topic_posts(self, topic_id: int) -> list[Post]:
        """Get all posts for a topic."""
        url = f"{self.base_url}/t/{topic_id}/posts.json"
        data = await self._request_with_retry(url)

        posts = []
        for post_data in data.get("post_stream", {}).get("posts", []):
            html_content = post_data["cooked"]

            # Extract first image from HTML content
            image_b64 = await self._extract_first_image_as_base64(html_content)

            # Convert HTML to Markdown
            markdown_content = md(html_content, heading_style="ATX").strip()

            posts.append(
                Post(
                    username=post_data["username"],
                    created_at=datetime.fromisoformat(
                        post_data["created_at"].replace("Z", "+00:00")
                    ),
                    content=markdown_content,
                    url=f"{self.base_url}/t/{topic_id}/{post_data['post_number']}",
                    image=image_b64,
                )
            )
        return posts

    async def _extract_first_image_as_base64(self, html: str) -> str | None:
        """Extract first image from HTML and return as base64."""
        image_urls = re.findall(
            r'<img[^>]+src=["\']([^"\']+)["\'][^>]*>', html, re.IGNORECASE
        )
        if not image_urls:
            return None

        return await self.get_image_as_base64(image_urls[0])

    async def get_image_as_base64(self, image_url: str) -> str | None:
        """Download image and return as base64 data URL."""
        try:
            if not image_url.startswith("http"):
                image_url = urljoin(self.base_url, image_url)

            response = await self.session.get(image_url)
            response.raise_for_status()

            # Get content type
            content_type = response.headers.get("content-type", "image/png")

            # Encode to base64
            b64_data = base64.b64encode(response.content).decode("utf-8")
            return f"data:{content_type};base64,{b64_data}"

        except Exception as e:
            console.print(f"[red]Failed to download {image_url}: {e}[/red]")
            return None

    async def close(self):
        await self.session.aclose()


async def process_topic(
    client: DiscourseClient,
    topic: Topic,
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
    progress: Progress,
    topic_task: int,
) -> bool:
    """Process single topic: get posts and save simplified data."""
    if not (start_date <= topic.created_at <= end_date):
        progress.update(topic_task, advance=1)
        return False

    posts = await client.get_topic_posts(topic.id)

    topic_dir = output_dir / f"topic_{topic.id}"
    topic_dir.mkdir(exist_ok=True)

    # Save simplified topic data with only required fields
    topic_data = {
        "topic_id": topic.id,
        "topic_title": topic.title,
        "topic_url": topic.url,
        "posts": [post.model_dump(mode="json") for post in posts],
    }

    (topic_dir / "data.json").write_text(
        json.dumps(topic_data, indent=2, default=str)
    )
    progress.update(topic_task, advance=1)
    return True


@click.command(help="Extract Discourse topics with images between dates.")
@click.argument("json_file", type=click.Path(exists=True, readable=True))
def main(json_file: str):
    """Extract Discourse topics with images.

    JSON config format:
    {
        "cookies": {"_t": "token", "_forum_session": "session"},
        "start_date": "2025-01-01",
        "end_date": "2025-04-15",
        "output_dir": "discourse_data",
        "category_id": 34
    }
    """
    try:
        with Path(json_file).open() as f:
            config = ScrapingConfig.model_validate(json.load(f))
    except (json.JSONDecodeError, ValidationError) as e:
        raise click.ClickException(f"Invalid JSON: {e}")

    asyncio.run(scrape_discourse(config))


async def scrape_discourse(config: ScrapingConfig):
    """Main scraping logic."""
    try:
        start_dt = datetime.fromisoformat(config.start_date).replace(
            tzinfo=timezone.utc
        )
        end_dt = datetime.fromisoformat(config.end_date).replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        raise click.ClickException("Invalid date format. Use YYYY-MM-DD")

    base_url = "https://discourse.onlinedegree.iitm.ac.in"
    t_token = config.cookies.t_token
    cookie_masked = (
        f"{t_token[:4]}...{t_token[-4:]}" if len(t_token) > 8 else "****"
    )

    console.print("\n[bold blue]Scraping Plan:[/bold blue]")
    console.print(f"[cyan]URL:[/cyan] {base_url}")
    console.print(f"[cyan]Cookie:[/cyan] {cookie_masked}")
    console.print(f"[cyan]Category:[/cyan] {config.category_id}")
    console.print(
        f"[cyan]Date Range:[/cyan] {config.start_date} to {config.end_date}"
    )
    console.print(f"[cyan]Output:[/cyan] {config.output_dir}\n")

    if not click.confirm("Proceed?"):
        return

    output_path = Path(config.output_dir)
    output_path.mkdir(exist_ok=True)

    client = DiscourseClient(base_url, config.cookies)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:

            page_task = progress.add_task(
                "[cyan]Fetching topics...", total=None
            )

            # Fetch all topics in batches
            all_topics = []
            page = 0
            batch_size = 5

            while True:
                page_tasks = [
                    client.get_category_topics(config.category_id, p)
                    for p in range(page, page + batch_size)
                ]
                page_results = await asyncio.gather(
                    *page_tasks, return_exceptions=True
                )

                empty_pages = 0
                for i, result in enumerate(page_results):
                    if isinstance(result, Exception) or not result:
                        empty_pages += 1
                        continue

                    all_topics.extend(result)
                    progress.update(
                        page_task,
                        description=f"[cyan]Fetched {len(all_topics)} topics (page {page + i})",
                    )

                    if result and result[-1].created_at < start_dt:
                        empty_pages = batch_size
                        break

                page += batch_size
                if empty_pages >= batch_size:
                    break

            progress.update(
                page_task,
                completed=True,
                description=f"[green]Found {len(all_topics)} topics",
            )

            # Filter by date
            filtered_topics = [
                t for t in all_topics if start_dt <= t.created_at <= end_dt
            ]
            console.print(
                f"[green]{len(filtered_topics)} topics in date range[/green]"
            )

            if not filtered_topics:
                console.print("[yellow]No topics in date range[/yellow]")
                return

            topic_task = progress.add_task(
                "[blue]Processing topics", total=len(filtered_topics)
            )

            # Process topics concurrently
            semaphore = asyncio.Semaphore(3)

            async def process_with_semaphore(topic):
                async with semaphore:
                    return await process_topic(
                        client,
                        topic,
                        start_dt,
                        end_dt,
                        output_path,
                        progress,
                        topic_task,
                    )

            results = await asyncio.gather(
                *[process_with_semaphore(topic) for topic in filtered_topics]
            )
            console.print(f"[green]Processed {sum(results)} topics[/green]")

    finally:
        await client.close()


if __name__ == "__main__":
    main()
