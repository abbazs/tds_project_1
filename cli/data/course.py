import asyncio
import base64
import re
from pathlib import Path

import rich_click as click
from pydantic import ValidationError
from rich.console import Console
from rich.progress import BarColumn
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TaskProgressColumn
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn

from cli.image_confext import AIImageAnalyzer
from cli.models import Settings
from cli.utils import print_important
from cli.utils import print_success
from cli.utils import print_warning

console = Console()


class AIImage(AIImageAnalyzer):
    def __init__(self, api_key: str) -> None:
        """Initialize AIImage with Google Generative AI client."""
        super().__init__(api_key)

    async def process_markdown_file(self, md_file: Path) -> bool:
        """Process single markdown file and insert AI context for local images."""
        try:
            content = md_file.read_text(encoding="utf-8")
        except OSError:
            return False

        # Find all images in markdown
        img_pattern = r'!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)'
        images = list(re.finditer(img_pattern, content))
        
        if not images:
            return False

        console.print(f"[bold cyan]📄 Processing: {md_file.name} ({len(images)} images)[/bold cyan]")
        modified = False

        # Process images in reverse order to maintain string positions
        for match in reversed(images):
            image_url = match.group(2)
            image_url_short = image_url.split("/")[-1][:40] if "/" in image_url else image_url[:40]
            
            console.print(f"  🖼️  Processing: {image_url_short}")
            
            # Check if context already exists
            next_content = content[match.end():]
            if re.match(r'\s*\n\s*\*Image Description:', next_content):
                console.print("  ✅  Context already exists")
                continue
            
            # Skip non-local images
            if (image_url.startswith(('http://', 'https://', 'data:', 'ftp://')) or
                not any(image_url.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.svg', '.webp'])):
                print_warning(f"Skipping non-local image: {image_url_short}")
                continue

            # Load local image file
            console.print("  📥  Loading local image...")
            try:
                image_path = md_file.parent / image_url
                if not image_path.exists():
                    print_warning(f"Local file not found: {image_url_short}")
                    continue

                image_data = image_path.read_bytes()
                ext = image_path.suffix.lower()
                content_type_map = {
                    '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                    '.gif': 'image/gif', '.bmp': 'image/bmp', '.svg': 'image/svg+xml', '.webp': 'image/webp'
                }
                content_type = content_type_map.get(ext, 'image/png')
                
                b64_data = base64.b64encode(image_data).decode('utf-8')
                image_b64 = f"data:{content_type};base64,{b64_data}"
                
                console.print("  📁  Source: local file")
                console.print("  🤖  AI analyzing...")
                
                context = await self.analyze_image(image_b64)
                
                if context:
                    context_preview = context[:60] + "..." if len(context) > 60 else context
                    print_success(f"AI Result: {context_preview}")
                    
                    # Insert context after the image
                    context_text = f"\n\n*Image Description: {context}*\n"
                    content = content[:match.end()] + context_text + content[match.end():]
                    modified = True
                else:
                    print_warning(f"AI analysis failed for: {image_url_short}")
                    
            except OSError as e:
                print_warning(f"Cannot read file {image_url_short}: {str(e)[:50]}")

        # Save if modified
        if modified:
            md_file.write_text(content, encoding="utf-8")
            print_success(f"Updated markdown file: {md_file.name}")
            return True
        else:
            console.print("  📝  No changes needed")
            return False

    async def analyze_course_markdown(self, data_dir: Path, progress: Progress, task_id: int) -> None:
        """Analyze course images in markdown files and insert context."""
        markdown_files = list(data_dir.glob("**/*.md"))
        
        # Single pass: read files and filter those with images
        files_with_images = []
        total_images = 0
        
        for md_file in markdown_files:
            try:
                content = md_file.read_text(encoding="utf-8")
                images = re.findall(r'!\[([^\]]*)\]\(([^)\s]+)(?:\s+"[^"]*")?\)', content)
                
                if images:
                    files_with_images.append(md_file)
                    total_images += len(images)
                    
            except OSError:
                continue

        if not files_with_images:
            print_warning("No markdown files with images found")
            return

        progress.update(task_id, total=total_images) # type: ignore
        print_important(f"Found {total_images} images across {len(files_with_images)} markdown files")
        console.print()

        # Process only files that contain images
        for md_file in files_with_images:
            await self.process_markdown_file(md_file)
            console.print()


@click.group(name="course", help="Process course markdown files")
def cli() -> None:
    pass


@cli.command()
def scrape() -> None:
    """Scrape course content (placeholder - not implemented)."""
    print_warning("Course scraping not implemented")
    print_warning("Course content is expected in .md format in the data/course folder")
    print_warning("This command is reserved for future implementation")


@cli.command()
@click.argument("data_dir", default="data/course", type=click.Path(exists=True, file_okay=False, path_type=Path))
def image_context(data_dir: Path) -> None:
    """Analyze local images in markdown files with AI and add context."""
    try:
        settings = Settings() # type: ignore
    except ValidationError as e:
        raise click.ClickException(f"Configuration error: {e}")

    analyzer = AIImage(settings.api_key)

    async def run_analysis() -> None:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("[cyan]🤖 AI Local Image Analysis...", total=None)
            await analyzer.analyze_course_markdown(data_dir, progress, task_id)
            progress.update(task_id, description="[green]✅ AI analysis complete!")

    asyncio.run(run_analysis())


if __name__ == "__main__":
    cli()