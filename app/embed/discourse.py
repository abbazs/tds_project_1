import asyncio
import json
from pathlib import Path
from typing import Dict
from typing import List

import rich_click as click
from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError
from rich.console import Console
from rich.progress import BarColumn
from rich.progress import MofNCompleteColumn
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TaskProgressColumn
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn

from app.embed.split import URLAwareTextSplitter
from app.embed.utils import concatenate_embeddings
from app.embed.utils import save_embeddings
from app.embedder import OpenAIEmbedder
from app.models import EmbeddingChunk
from app.models import Settings
from app.utils import error_exit
from app.utils import print_error

console = Console()


class PostData(BaseModel):
    username: str
    created_at: str
    content: str
    url: str
    images: List[Dict[str, str]] = Field(default_factory=list)


class TopicData(BaseModel):
    topic_id: int
    topic_title: str
    topic_url: str
    posts: List[PostData]


async def process_json_file(
    file_path: Path,
    embedder: OpenAIEmbedder,
    splitter: URLAwareTextSplitter,
    progress: Progress,
) -> List[EmbeddingChunk]:
    """Process single JSON file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        topic_data = TopicData(**data)
    except Exception as e:
        print_error(f"Error processing {file_path}: {e}")
        return []

    post_task = progress.add_task(
        f" [yellow]Processing file {file_path.stem}", total=len(topic_data.posts)
    )
    chunks = []
    for post in topic_data.posts:
        if not post.content.strip():
            if progress and post_task:
                progress.update(post_task, advance=1)
            continue
        # Add image context inline
        content = post.content
        for img in post.images:
            if (img_url := img.get("url")) and (context := img.get("context")):
                content = content.replace(img_url, f"{img_url}\n[Image: {context}]")

        text_chunks = splitter.split_text(content)
        total_chunks = len(text_chunks)
        if total_chunks > 1:
            chunk_task = progress.add_task(
                f"  [dim]Embedding {total_chunks} chunks", total=total_chunks
            )
        for _i, chunk_text in enumerate(text_chunks):
            embedding = await embedder.embed_text(chunk_text)

            if embedding:
                chunks.append(
                    EmbeddingChunk(
                        text=chunk_text,
                        url=post.url,
                        embedding=embedding,
                    )
                )
            if total_chunks > 1:
                progress.update(chunk_task, advance=1)
        if progress and post_task:
            progress.update(post_task, advance=1)
    return chunks


@click.group(name="discourse", help="Embed Discourse data")
def cli() -> None:
    pass


@cli.command()
@click.option(
    "--input-dir",
    "-i",
    type=click.Path(exists=True, path_type=Path),
    default="data/discourse",
    help="Directory with JSON files",
)
@click.option(
    "--output-file",
    "-o",
    type=click.Path(path_type=Path),
    default="embeddings/discourse.npz",
    help="Output embeddings file",
)
@click.option("--chunk-size", default=1500, help="Max chunk size")
@click.option("--chunk-overlap", default=200, help="Chunk overlap")
def embed(
    input_dir: Path, output_file: Path, chunk_size: int, chunk_overlap: int
) -> None:
    """Embed Discourse JSON files using Gemini"""

    try:
        settings = Settings()
    except ValidationError as e:
        error_exit(f"Config error: {e}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    async def run():
        console.print("[bold blue]🔍 Analyzing input directory...[/bold blue]")

        json_files = list(input_dir.glob("*.json"))
        if not json_files:
            error_exit(f"No JSON files in {input_dir}")

        # Calculate stats
        total_size = sum(f.stat().st_size for f in json_files) / (1024 * 1024)
        total_posts = 0

        for file_path in json_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    total_posts += len(data.get("posts", []))
            except Exception:
                continue

        # Show summary
        console.print("\n[bold yellow]📋 Embedding Summary[/bold yellow]")
        console.print(f"📁 Input directory: {input_dir}")
        console.print(f"📄 JSON files: {len(json_files)}")
        console.print(f"💬 Total posts: {total_posts}")
        console.print(f"📏 Total size: {total_size:.2f} MB")
        console.print(f"⚙️  Chunk size: {chunk_size} chars")
        console.print(f"🔗 Chunk overlap: {chunk_overlap} chars")
        console.print(f"💾 Output: {output_file}")

        if not click.confirm("\n🚀 Continue with embedding?", default=True):
            console.print("[yellow]❌ Cancelled by user[/yellow]")
            return

        console.print("\n[bold blue]🚀 Starting embedding...[/bold blue]")

        embedder = OpenAIEmbedder(settings.api_key)
        splitter = URLAwareTextSplitter(chunk_size, chunk_overlap)

        npz_files = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=60),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            expand=True,
        ) as progress:
            task = progress.add_task("[cyan]Processing...", total=len(json_files))

            for file_path in json_files:
                progress.update(task, description=f"[cyan]Processing {file_path.name}")
                indi_file = output_file.parent.joinpath(file_path.stem + ".npz")
                if indi_file.exists():
                    console.print(
                        f"[yellow]Skipping existing file: {indi_file}[/yellow]"
                    )
                    npz_files.append(indi_file)
                    progress.update(task, advance=1)
                    continue
                chunks = await process_json_file(
                    file_path, embedder, splitter, progress
                )
                save_embeddings(chunks=chunks, output_path=indi_file)
                npz_files.append(indi_file)
                progress.update(task, advance=1)

        concatenate_embeddings(npz_files, output_file)
        size_mb = output_file.stat().st_size / (1024 * 1024)
        console.print(f"[blue]💾 Saved: {output_file} ({size_mb:.2f} MB)[/blue]")

    asyncio.run(run())


if __name__ == "__main__":
    cli()
