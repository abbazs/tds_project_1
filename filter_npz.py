import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import rich_click as click
from rich.console import Console
from rich.progress import Progress
from rich.progress import SpinnerColumn
from rich.progress import TextColumn
from rich.progress import TimeElapsedColumn

console = Console()

# VIP user weights configuration
VIP_WEIGHTS = {
    "s.anand": 10.0,  # Faculty
    "carlton": 7.5,  # Instructor
    "iamprasna": 7.5,  # Instructor
    "jivraj": 5.0,  # TA
    "21f3002441": 5.0,  # TA (Suchintika)
    "hritikroshan_hrm": 5.0,  # TA (Hritik)
    "saransh_saini": 5.0,  # TA (Saransh)
}


def extract_last_two_numbers(url: str) -> tuple[int | None, int | None]:
    """Extract the last two numbers from URL. Returns (second_to_last, last)."""
    numbers = list(map(int, re.findall(r"\d+", url)))

    if len(numbers) == 0:
        return None, None
    elif len(numbers) == 1:
        return None, numbers[0]
    else:
        return numbers[-2], numbers[-1]


def convert_numpy_item(item: Any) -> Any:
    """Convert numpy items to JSON-serializable format."""
    if isinstance(item, np.ndarray):
        if item.size == 1:
            return item.item()
        else:
            return item.tolist()
    elif hasattr(item, "item") and callable(getattr(item, "item")):
        try:
            return item.item()
        except ValueError:
            return item.tolist() if hasattr(item, "tolist") else str(item)
    else:
        return item


def filter_npz_by_url_number(
    npz_path: str,
    topic_id: int,
    number_range: tuple[int, int],
    output_json_path: str,
) -> None:
    data = np.load(npz_path, allow_pickle=True)
    urls = data["urls"]
    start, end = number_range

    valid_indices = [
        idx
        for idx, url in enumerate(urls)
        if (numbers := extract_last_two_numbers(str(url)))
        and numbers[0] == topic_id
        and numbers[1] is not None
        and start <= numbers[1] <= end
    ]

    filtered: dict[str, list[Any]] = {
        key: [convert_numpy_item(data[key][i]) for i in valid_indices]
        for key in data.files
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)

    console.print(f"✅ Filtered {len(valid_indices)} rows from {len(urls)} total")
    console.print(f"💾 Saved to {output_json_path}", style="bold green")


def fetch_single_url_id(
    npz_path: str,
    topic_id: int,
    target_id: int,
    output_json_path: str,
) -> None:
    data = np.load(npz_path, allow_pickle=True)
    urls = data["urls"]

    found_indices = []
    for idx, url in enumerate(urls):
        if (
            (numbers := extract_last_two_numbers(str(url)))
            and numbers[0] == topic_id
            and numbers[1] == target_id
        ):
            found_indices.append(idx)

    if not found_indices:
        console.print(
            f"❌ No URL found with topic ID {topic_id} and target ID {target_id}",
            style="bold red",
        )
        return

    if len(found_indices) == 1:
        single_row: dict[str, Any] = {
            key: convert_numpy_item(data[key][found_indices[0]]) for key in data.files
        }
        result_data = single_row
    else:
        result_data: dict[str, list[Any]] = {
            key: [convert_numpy_item(data[key][i]) for i in found_indices]
            for key in data.files
        }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)

    console.print(
        f"✅ Found {len(found_indices)} URL(s) with topic ID {topic_id} and target ID {target_id} at rows: {found_indices}"
    )
    console.print(f"💾 Saved to {output_json_path}", style="bold green")


def apply_vip_weights_to_npz(
    npz_path: Path,
    json_folder: Path,
    dry_run: bool = False,
    no_backup: bool = False,
) -> None:
    """Apply VIP user weights to embeddings in NPZ file."""
    vip_weights_lower = {k.lower(): v for k, v in VIP_WEIGHTS.items()}

    # Show configuration
    console.print("[bold blue]💡 VIP Weight Configuration:[/bold blue]")
    for user, weight in VIP_WEIGHTS.items():
        console.print(f"  {user}: {weight}x")
    console.print()

    # Load NPZ data
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Loading NPZ file...", total=None)
        data = np.load(npz_path, allow_pickle=True)
        npz_data = {key: data[key] for key in data.files}

        # Validate required fields
        if "embeddings" not in npz_data or "urls" not in npz_data:
            console.print(
                "❌ NPZ file missing required fields: embeddings or urls",
                style="bold red",
            )
            return

        progress.update(task, completed=True)

    # Get all JSON files
    json_files = sorted(json_folder.glob("*.json"))
    if not json_files:
        console.print(f"❌ No JSON files found in {json_folder}", style="bold red")
        return

    console.print(f"📂 Found {len(json_files)} JSON files to process")

    if dry_run:
        console.print("[yellow]⚠️  DRY RUN MODE - No changes will be saved[/yellow]")

    # Process each JSON file
    total_vip = 0
    total_modified = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Processing topics...", total=len(json_files))

        for json_file in json_files:
            console.print(f"\n📄 Processing: {json_file.name}")

            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    topic_data = json.load(f)

                for post in topic_data.get("posts", []):
                    username_lower = post.get("username", "").lower()

                    if username_lower in vip_weights_lower:
                        total_vip += 1
                        weight = vip_weights_lower[username_lower]
                        post_url = post.get("url", "").strip()

                        # Find matching URLs in NPZ
                        indices = []
                        for idx, url in enumerate(npz_data["urls"]):
                            if str(url).strip() == post_url:
                                indices.append(idx)

                        if indices:
                            if not dry_run:
                                # Apply weight to embeddings
                                for idx in indices:
                                    npz_data["embeddings"][idx] *= weight

                            total_modified += len(indices)
                            console.print(
                                f"  👤 VIP: {post['username']} (weight={weight:.1f}) "
                                f"→ {len(indices)} embedding(s) modified"
                            )
                        else:
                            console.print(
                                f"  ⚠️  URL not found in NPZ: {post_url}", style="yellow"
                            )

            except Exception as e:
                console.print(f"  ❌ Failed to process: {e}", style="red")

            progress.update(task, advance=1)

    # Summary
    console.print("\n" + "=" * 50)
    console.print("[bold blue]💡 Summary:[/bold blue]")
    console.print(f"  Total VIP posts found: {total_vip}")
    console.print(f"  Total embeddings modified: {total_modified}")

    # Save if not dry run
    if not dry_run and total_modified > 0:
        if not no_backup:
            # Create backup
            backup_path = npz_path.with_suffix(".npz.bak")
            if npz_path.exists():
                import shutil

                shutil.copy2(npz_path, backup_path)
                console.print(f"📦 Backup created: {backup_path}")

        # Save modified data
        np.savez_compressed(npz_path, **npz_data)
        console.print(
            f"✅ Saved modified embeddings to: {npz_path}", style="bold green"
        )
    elif dry_run:
        console.print("\n💡 Run without --dry-run to apply changes")


@click.group()
def cli() -> None:
    """Filter NPZ files by URL numbers and apply VIP weights."""
    pass


@cli.command()
@click.argument("npz_file", type=click.Path(exists=True))
@click.option("--topic", "-t", "topic_id", type=int, required=True, help="Topic ID")
@click.option("--min-id", type=int, required=True, help="Minimum ID (inclusive)")
@click.option("--max-id", type=int, required=True, help="Maximum ID (inclusive)")
@click.option("--output", "-o", default="filtered_data.json", help="Output JSON file")
def filter_range(
    npz_file: str, topic_id: int, min_id: int, max_id: int, output: str
) -> None:
    """Filter NPZ file by URL ID range and save as JSON."""
    filter_npz_by_url_number(npz_file, topic_id, (min_id, max_id), output)


@cli.command()
@click.argument("npz_file", type=click.Path(exists=True))
@click.option("--topic", "-t", "topic_id", type=int, required=True, help="Topic ID")
@click.option("--id", "target_id", type=int, required=True, help="Target URL ID")
def fetch_one(npz_file: str, topic_id: int, target_id: int) -> None:
    """Fetch single row by exact URL ID match."""
    fetch_single_url_id(npz_file, topic_id, target_id, f"{topic_id}_{target_id}.json")


@cli.command()
@click.argument("npz_file", type=click.Path(exists=True, path_type=Path))
@click.argument(
    "json_folder", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--dry-run", is_flag=True, help="Preview changes without modifying NPZ")
@click.option("--no-backup", is_flag=True, help="Skip creating backup file")
def apply_vip_weights(
    npz_file: Path,
    json_folder: Path,
    dry_run: bool,
    no_backup: bool,
) -> None:
    """Apply VIP user weights to embeddings in NPZ file.

    Scans JSON files in the folder for VIP users and multiplies their
    corresponding embeddings by predefined weights.
    """
    apply_vip_weights_to_npz(npz_file, json_folder, dry_run, no_backup)


if __name__ == "__main__":
    cli()
