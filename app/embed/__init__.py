from pathlib import Path

import numpy as np
import rich_click as click

from app.embed import course
from app.embed import discourse
from app.utils import print_error
from app.utils import print_success


@click.group(name="embed", help="Embedding cli")
def cli() -> None:
    pass


@cli.command(name="join-npz")
@click.argument(
    "input_files",
    nargs=-1,
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default="embeddings.npz",
    show_default=True,
    help="Filename for the merged NPZ",
)
def join_npz(input_files: tuple[Path, ...], output: Path) -> None:
    """
    Concatenate multiple NPZ embedding files into a single compressed NPZ.
    """
    # 1. Collect arrays under each key
    all_arrays: dict[str, list[np.ndarray]] = {}
    for npz_path in input_files:
        with np.load(npz_path, allow_pickle=True) as data:
            for key in data.files:
                all_arrays.setdefault(key, []).append(data[key])

    if not all_arrays:
        print_error("No valid data found in the provided NPZ files.")
        return

    # 2. Concatenate, downcast embeddings to float32
    concatenated: dict[str, np.ndarray] = {}
    for key, chunks in all_arrays.items():
        arr = np.concatenate(chunks, axis=0)
        concatenated[key] = arr.astype(np.float32) if key == "embeddings" else arr

    # 3. Write out compressed NPZ
    np.savez_compressed(output, **concatenated)
    print_success(f"Merged {len(input_files)} files into {output}")


cli.add_command(course.embed, name="course")
cli.add_command(discourse.embed, name="discourse")

if __name__ == "__main__":
    cli()
