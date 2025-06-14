# cli/embed/course.py
import asyncio
import re
from pathlib import Path
from typing import List

import rich_click as click
from markdown_it import MarkdownIt
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
from slugify import slugify

from app.embed.split import URLAwareTextSplitter
from app.embed.utils import concatenate_embeddings
from app.embed.utils import save_embeddings
from app.embedder import OpenAIEmbedder
from app.models import EmbeddingChunk
from app.models import Settings
from app.utils import error_exit

console = Console()


class MarkdownSection(BaseModel):
    content: str
    headers: List[str] = Field(default_factory=list)
    level: int = 0
    file_path: str


def parse_markdown(content: str, file_path: str) -> List[MarkdownSection]:
    """Parse markdown into hierarchical sections"""
    md = MarkdownIt("commonmark", {"breaks": True, "html": True})
    tokens = md.parse(content)
    lines = content.split("\n")

    sections = []
    headers = [""] * 6  # h1-h6 tracking
    current_content = []
    line_idx = 0

    for token in tokens:
        if token.type == "heading_open":
            # Save previous section
            if current_content:
                active_headers = [h for h in headers if h]
                sections.append(
                    MarkdownSection(
                        content="\n".join(current_content).strip(),
                        headers=active_headers.copy(),
                        level=len(active_headers),
                        file_path=file_path,
                    )
                )
                current_content = []

            # Update header hierarchy
            level = int(token.tag[1]) - 1
            next_idx = tokens.index(token) + 1
            header_text = (
                tokens[next_idx].content
                if next_idx < len(tokens) and tokens[next_idx].type == "inline"
                else ""
            )

            headers[level:] = [header_text] + [""] * (5 - level)

        # Collect content lines
        if token.map and line_idx < token.map[1]:
            current_content.extend(lines[line_idx : token.map[1]])
            line_idx = token.map[1]

    # Add remaining content
    current_content.extend(lines[line_idx:])
    if current_content:
        active_headers = [h for h in headers if h]
        sections.append(
            MarkdownSection(
                content="\n".join(current_content).strip(),
                headers=active_headers.copy(),
                level=len(active_headers),
                file_path=file_path,
            )
        )

    return [s for s in sections if s.content.strip()]


async def process_file(
    file_path: Path,
    embedder: OpenAIEmbedder,
    splitter: URLAwareTextSplitter,
    progress: Progress = None,
) -> List[EmbeddingChunk]:
    """Process single markdown file"""
    try:
        content = file_path.read_text(encoding="utf-8")
        sections = parse_markdown(content, str(file_path))
    except Exception as e:
        console.print(f"[red]Error reading {file_path}: {e}[/red]")
        return []

    if not sections:
        return []

    section_task = progress.add_task(
        f" [yellow]Processing chunks of {file_path.stem}...",
        total=len(sections),
    )
    chunks = []

    for _i, section in enumerate(sections):
        if not section.content.strip():
            if progress and section_task:
                progress.update(section_task, advance=1)
            continue

        if len(section.headers) > 0:
            url = f"https://tds.s-anand.net/#{file_path.stem}?id={slugify(section.headers[-1])}"
        else:
            url = f"https://tds.s-anand.net/#{file_path.stem}"

        text_chunks = splitter.split_text(section.content)
        total_chunks = len(text_chunks)
        if total_chunks > 1:
            chunk_task = progress.add_task(
                f"  [dim]Embedding {total_chunks} chunks", total=total_chunks
            )

        for _, chunk_text in enumerate(text_chunks):
            embedding = await embedder.embed_text(chunk_text)

            if embedding:
                chunks.append(
                    EmbeddingChunk(
                        text=chunk_text,
                        url=url,
                        embedding=embedding,
                    )
                )
                console.print(f"[dim]{url}[/dim]")
            if total_chunks > 1:
                progress.update(chunk_task, advance=1)
        if progress and section_task:
            progress.update(section_task, advance=1)

    return chunks


@click.group(name="course", help="Embed course markdown data")
def cli() -> None:
    pass


@cli.command()
@click.option(
    "--input-dir",
    "-i",
    type=click.Path(exists=True, path_type=Path),
    default="data/course",
    help="Directory with markdown files",
)
@click.option(
    "--output-file",
    "-o",
    type=click.Path(path_type=Path),
    default="embeddings/course.npz",
    help="Output embeddings file",
)
@click.option("--chunk-size", default=1500, help="Max chunk size")
@click.option("--chunk-overlap", default=200, help="Chunk overlap")
def embed(
    input_dir: Path, output_file: Path, chunk_size: int, chunk_overlap: int
) -> None:
    """Embed course markdown files using Gemini"""

    try:
        settings = Settings()  # type: ignore
    except Exception as e:
        error_exit(f"Config error: {e}")

    output_file.parent.mkdir(parents=True, exist_ok=True)

    async def run() -> None:
        console.print("[bold blue]🔍 Analyzing input directory...[/bold blue]")

        md_files = list(input_dir.glob("**/*.md"))
        if not md_files:
            error_exit(f"No markdown files in {input_dir}")

        # Calculate total size
        total_size = sum(f.stat().st_size for f in md_files) / (1024 * 1024)

        # Show summary
        console.print("\n[bold yellow]📋 Embedding Summary[/bold yellow]")
        console.print(f"📁 Input directory: {input_dir}")
        console.print(f"📄 Files to process: {len(md_files)}")
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
            BarColumn(),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
            expand=True,
        ) as progress:
            # Calculate total chunks estimate (rough)

            task = progress.add_task("[cyan]Processing files...", total=len(md_files))

            for file_path in md_files:
                progress.update(
                    task, description=f"[cyan]Processing {file_path.name}..."
                )
                indi_file = output_file.parent.joinpath(file_path.stem + ".npz")
                if indi_file.exists():
                    console.print(
                        f"[yellow]Skipping existing file: {indi_file}[/yellow]"
                    )
                    npz_files.append(indi_file)
                    progress.update(task, advance=1)
                    continue
                chunks = await process_file(file_path, embedder, splitter, progress)
                save_embeddings(chunks=chunks, output_path=indi_file)
                npz_files.append(indi_file)
                progress.update(task, advance=1)

        concatenate_embeddings(npz_files, output_file)
        size_mb = output_file.stat().st_size / (1024 * 1024)
        console.print(f"[blue]💾 Saved: {output_file} ({size_mb:.2f} MB)[/blue]")

    asyncio.run(run())


if __name__ == "__main__":
    cli()
