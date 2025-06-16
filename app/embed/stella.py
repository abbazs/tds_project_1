# app/embed/discourse_stella.py
import asyncio
import json
from pathlib import Path
from typing import Dict
from typing import List

import rich_click as click
from pydantic import BaseModel
from pydantic import Field
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
from app.models import EmbeddingChunk
from app.stella import StellaEmbedder
from app.utils import error_exit
from app.utils import print_error
from app.utils import print_success

console = Console()

# Simple authority weights - that's it!
AUTHORITY_USERS = {
    "s.anand": "FACULTY",
    "carlton": "INSTRUCTOR",
    "iamprasna": "INSTRUCTOR",
    "jivraj": "TEACHING ASSISTANT",
    "21f3002441": "TEACHING ASSISTANT",
    "hritikroshan_hrm": "TEACHING ASSISTANT",
    "saransh_saini": "TEACHING ASSISTANT",
}


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


def ensure_stella_model(model_path: Path) -> StellaEmbedder:
    """Ensure Stella model is available locally, download if needed"""
    if model_path.exists():
        console.print(f"[green]✓ Loading existing model from {model_path}[/green]")
        return StellaEmbedder(model_path=str(model_path))
    else:
        console.print(
            "[yellow]⚠ Model not found locally, downloading from HuggingFace...[/yellow]"
        )
        console.print("[dim]This is a one-time download (~1.5GB)[/dim]")

        # Create embedder (will download model)
        embedder = StellaEmbedder()

        # Save for future use
        console.print(
            f"\n[blue]💾 Saving model to {model_path} for faster loading next time...[/blue]"
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        embedder.save_model(str(model_path))

        print_success(f"Model saved to {model_path}")
        return embedder


async def process_json_file_batch(
    file_path: Path,
    embedder: StellaEmbedder,
    splitter: URLAwareTextSplitter,
) -> List[EmbeddingChunk]:
    """Process JSON file with batch embedding for better performance"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        topic_data = TopicData(**data)
    except Exception as e:
        print_error(f"Error processing {file_path}: {e}")
        return []

    console.print(
        f" [yellow]Processing {file_path.stem} ({len(topic_data.posts)} posts)[/yellow]"
    )

    # Collect all chunks first for batch processing
    all_chunks_data = []

    for post in topic_data.posts:
        if not post.content.strip():
            console.print(f"  [dim]Skipping empty post {post.url}[/dim]")
            continue

        # Add image context inline
        content = post.content
        for img in post.images:
            if (img_url := img.get("url")) and (context := img.get("context")):
                content = content.replace(img_url, f"{img_url}\n[Image: {context}]")

        # Split first
        text_chunks = splitter.split_text(content)

        for chunk_text in text_chunks:
            # Apply authority weighting to each chunk
            if post.username in AUTHORITY_USERS:
                role = AUTHORITY_USERS[post.username]
                weighted_chunk = f"[{role} Response] {chunk_text}"
            else:
                weighted_chunk = chunk_text

            all_chunks_data.append(
                {
                    "weighted_text": weighted_chunk,
                    "original_text": post.content,
                    "url": post.url,
                }
            )

    # Batch embed all chunks
    if all_chunks_data:
        console.print(
            f"  [blue]Embedding {len(all_chunks_data)} chunks in batch...[/blue]"
        )
        weighted_texts = [chunk["weighted_text"] for chunk in all_chunks_data]
        embeddings = await embedder.embed_texts_async(weighted_texts)

        # Create EmbeddingChunk objects
        chunks = []
        for chunk_data, embedding in zip(all_chunks_data, embeddings):
            if embedding:  # Check if embedding is valid
                chunks.append(
                    EmbeddingChunk(
                        text=chunk_data["original_text"],
                        url=chunk_data["url"],
                        embedding=embedding,
                    )
                )

        console.print(f"  [green]✓ Processed {len(chunks)} embeddings[/green]")
        return chunks

    return []


@click.group(name="discourse-stella", help="Embed Discourse data using Stella")
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
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default="embeddings",
    help="Output directory for embeddings",
)
@click.option(
    "--model-path",
    "-m",
    type=click.Path(path_type=Path),
    default="models/stella_en_400M_v5",
    help="Path to save/load Stella model",
)
@click.option("--chunk-size", default=1500, help="Max chunk size")
@click.option("--chunk-overlap", default=200, help="Chunk overlap")
@click.option("--batch-size", default=32, help="Batch size for embedding")
@click.option("--device", default="cpu", help="Device to use (cuda/cpu/auto)")
def embed(
    input_dir: Path,
    output_dir: Path,
    model_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
    device: str,
) -> None:
    """Embed Discourse JSON files using Stella with authority weighting"""

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir.joinpath("discourse_stella.npz")

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
        console.print(f"🤖 Model: Stella EN 400M v5 (Local)")
        console.print(f"📁 Input directory: {input_dir}")
        console.print(f"📄 JSON files: {len(json_files)}")
        console.print(f"💬 Total posts: {total_posts}")
        console.print(f"📏 Total size: {total_size:.2f} MB")
        console.print(f"⚙️  Chunk size: {chunk_size} chars")
        console.print(f"🔗 Chunk overlap: {chunk_overlap} chars")
        console.print(f"📦 Batch size: {batch_size}")
        console.print(f"🖥️  Device: {device or 'auto-detect'}")
        console.print(
            f"👑 Authority weighting: Enabled for {len(AUTHORITY_USERS)} users"
        )
        console.print(f"💾 Output: {output_file}")

        if not click.confirm("\n🚀 Continue with embedding?", default=True):
            console.print("[yellow]❌ Cancelled by user[/yellow]")
            return

        console.print("\n[bold blue]🚀 Initializing Stella embedder...[/bold blue]")

        # Ensure model is available
        embedder = ensure_stella_model(model_path)
        if device:
            embedder.device = device
        embedder.batch_size = batch_size

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
            task = progress.add_task("[cyan]Processing files...", total=len(json_files))

            for file_path in json_files:
                progress.update(task, description=f"[cyan]Processing {file_path.name}")
                indi_file = output_dir.joinpath(file_path.stem + "_stella.npz")

                if indi_file.exists():
                    console.print(
                        f"[yellow]Skipping existing file: {indi_file}[/yellow]"
                    )
                    npz_files.append(indi_file)
                    progress.update(task, advance=1)
                    continue

                chunks = await process_json_file_batch(file_path, embedder, splitter)

                if chunks:
                    save_embeddings(chunks=chunks, output_path=indi_file)
                    npz_files.append(indi_file)

                progress.update(task, advance=1)

        if npz_files:
            console.print("\n[blue]📦 Concatenating embeddings...[/blue]")
            concatenate_embeddings(npz_files, output_file)
            size_mb = output_file.stat().st_size / (1024 * 1024)
            console.print(f"\n[bold green]✅ Success![/bold green]")
            console.print(f"💾 Saved: {output_file} ({size_mb:.2f} MB)")
            console.print(f"📊 Total files processed: {len(npz_files)}")
        else:
            console.print("[red]❌ No embeddings generated[/red]")

    asyncio.run(run())


# Add this to your main CLI in __init__.py or as a separate command
if __name__ == "__main__":
    cli()
