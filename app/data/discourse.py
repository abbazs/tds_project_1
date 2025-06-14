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
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict
from rich.console import Console
from rich.progress import BarColumn
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TaskProgressColumn
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn

from app.image_confext import OpenAIImageAnalyzer
from app.models import Settings

console = Console()





class CookieConfig(BaseModel):
    t_token: str = Field(alias="_t")
    forum_session: str = Field(alias="_forum_session")


class ImageData(BaseModel):
    url: str
    base64_data: str | None = None
    context: str | None = None


class Post(BaseModel):
    username: str
    created_at: datetime
    content: str
    url: str
    images: list[ImageData] = Field(default_factory=list)


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
    api_key: str | None = None


def load_config(
    json_file: str, require_api_key: bool = False
) -> ScrapingConfig:
    """Load and validate configuration."""
    try:
        with Path(json_file).open() as f:
            json_data = json.load(f)

        # Get API key from settings only if required for AI commands
        if require_api_key and (
            "api_key" not in json_data or not json_data["api_key"]
        ):
            settings = Settings() # type: ignore
            json_data["api_key"] = settings.api_key

        config = ScrapingConfig.model_validate(json_data)

        # Validate API key is present for AI commands
        if require_api_key and not config.api_key:
            raise click.ClickException(
                "GENAI_API_KEY required for AI analysis. Set in JSON config or environment."
            )

        return config
    except Exception as e:
        console.print_exception()
        raise click.ClickException(f"Config error: {e}")


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

            # Extract all images with context from HTML content
            images = await self._extract_images_with_context(html_content)

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
                    images=images,
                )
            )
        return posts

    async def _extract_images_with_context(self, html: str) -> list[ImageData]:
        """Extract all images from HTML with surrounding context, excluding avatars."""
        # Find all img tags with their surrounding context
        img_pattern = r'<p[^>]*>([^<]*)<img[^>]+src=["\']([^"\']+)["\'][^>]*>([^<]*)</p>|<img[^>]+src=["\']([^"\']+)["\'][^>]*>'
        matches = re.findall(img_pattern, html, re.IGNORECASE | re.DOTALL)

        images = []
        for match in matches:
            # Extract URL and context from regex groups
            if match[1]:  # Image within paragraph
                url = match[1]
                context = (match[0] + match[2]).strip()
            else:  # Standalone image
                url = match[3]
                context = None
            # Add this at class level in DiscourseClient.__init__:
            self.skip_image_pattern = re.compile(
                r"(avatars\.discourse-cdn\.com|emoji\.discourse-cdn\.com|user_avatar|"
                r"gravatar\.com|letter_avatar_proxy|/images/emoji/|\.svg$)",
                re.IGNORECASE,
            )

            # Then in the method:
            if self.skip_image_pattern.search(url):
                continue
            # Get base64 data
            base64_data = await self.get_image_as_base64(url)

            images.append(
                ImageData(
                    url=url,
                    base64_data=base64_data,
                    context=context if context else None,
                )
            )

        return images

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


class AIImage(OpenAIImageAnalyzer):
    def __init__(self, api_key: str) -> None:
        """Initialize AIImage with Google Generative AI client."""
        super().__init__(api_key)

    async def download_and_analyze_images(
        self, output_dir: Path, progress: Progress, task_id: int
    ):
        """Download images and analyze with AI for all topics."""
        topic_files = list(output_dir.glob("topic_*.json"))

        total_images = 0
        processed_images = 0

        # Count total images first
        for topic_file in topic_files:
            with topic_file.open() as f:
                topic_data = json.load(f)

            for post in topic_data["posts"]:
                total_images += len(post.get("images", []))

        progress.update(task_id, total=total_images)
        console.print(
            f"[blue]🔍 Found {total_images} images across {len(topic_files)} topics[/blue]\n"
        )

        async with httpx.AsyncClient(timeout=30.0) as session:
            for topic_file in topic_files:
                with topic_file.open() as f:
                    topic_data = json.load(f)

                topic_title = topic_data.get("topic_title", "Unknown Topic")
                topic_id = topic_data.get("topic_id", "Unknown")

                console.print(
                    f"[bold cyan]📄 Processing Topic {topic_id}: {topic_title[:80]}[/bold cyan]"
                )

                modified = False

                for post in topic_data["posts"]:
                    for image in post.get("images", []):
                        if image.get("base64_data") and image.get("context"):
                            processed_images += 1
                            progress.update(task_id, advance=1)
                            continue

                        # Log current image being processed
                        image_url_short = image["url"].split("/")[-1][:40]
                        console.print(f"  🖼️  Downloading: {image_url_short}")

                        # Download image
                        try:
                            response = await session.get(image["url"])
                            response.raise_for_status()
                            content_type = response.headers.get(
                                "content-type", "image/png"
                            )
                            b64_data = base64.b64encode(
                                response.content
                            ).decode("utf-8")
                            image["base64_data"] = (
                                f"data:{content_type};base64,{b64_data}"
                            )
                            console.print("  ✅  Downloaded successfully")
                        except Exception as e:
                            console.print(
                                f"  ❌  Download failed: {str(e)[:50]}"
                            )
                            processed_images += 1
                            progress.update(task_id, advance=1)
                            continue

                        # Log AI analysis start
                        console.print("  🤖  AI analyzing...")

                        # Analyze with AI
                        context = await self.analyze_image(image["base64_data"])
                        image["context"] = context
                        modified = True

                        if context:
                            context_preview = (
                                context[:60] + "..."
                                if len(context) > 60
                                else context
                            )
                            console.print(f"  ✨  AI Result: {context_preview}")
                        else:
                            console.print("  ⚠️   AI analysis failed")

                        processed_images += 1
                        progress.update(task_id, advance=1)

                # Save updated data
                if modified:
                    with topic_file.open("w") as f:
                        json.dump(topic_data, f, indent=2, default=str)
                    console.print(f"  💾  Saved updates to {topic_file.name}")

                console.print()  # Add blank line between topics


async def process_topic(
    client: DiscourseClient,
    topic: Topic,
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
    progress: Progress,
    topic_task: int,
) -> bool:
    """Process single topic: get posts and save as single JSON file."""
    if not (start_date <= topic.created_at <= end_date):
        progress.update(topic_task, advance=1)
        return False

    posts = await client.get_topic_posts(topic.id)

    # Save as single JSON file instead of folder
    topic_data = {
        "topic_id": topic.id,
        "topic_title": topic.title,
        "topic_url": topic.url,
        "posts": [post.model_dump(mode="json") for post in posts],
    }

    (output_dir / f"topic_{topic.id}.json").write_text(
        json.dumps(topic_data, indent=2, default=str)
    )
    progress.update(topic_task, advance=1)
    return True


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
    output_path.mkdir(parents=True, exist_ok=True)

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


@click.group(
    name="discourse", help="Discourse scraper with separated AI analysis"
)
def cli(): # type: ignore
    """Discourse scraper with separated AI analysis."""
    click.echo("Welcome to the Discourse Scraper CLI!")


@cli.command()
@click.argument("json_file", type=click.Path(exists=True, readable=True))
def scrape(json_file: str): # type: ignore
    """Scrape discourse topics (without AI analysis)."""
    config = load_config(json_file, require_api_key=False)
    asyncio.run(scrape_discourse(config))


@cli.command()
@click.argument("json_file", type=click.Path(exists=True, readable=True))
def image_context(json_file: str): # type: ignore
    """Analyze images with AI and save context back to JSON files."""
    config = load_config(json_file, require_api_key=True)

    output_path = Path(config.output_dir)
    if not output_path.exists():
        raise click.ClickException(
            f"Output directory {config.output_dir} does not exist. Run 'scrape' first."
        )

    analyzer = AIImage(config.api_key) # type: ignore

    async def run_analysis(): # type: ignore
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task(
                "[cyan]🤖 AI Image Analysis...", total=None
            )
            await analyzer.download_and_analyze_images(
                output_path, progress, task_id
            )
            progress.update(
                task_id, description="[green]✅ AI analysis complete!"
            )

    asyncio.run(run_analysis()) # type: ignore


if __name__ == "__main__":
    cli()
