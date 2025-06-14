from pathlib import Path

import numpy as np

from app.models import EmbeddingChunk
from app.utils import print_error


def save_embeddings(chunks: list[EmbeddingChunk], output_path: Path) -> None:
    """Save embeddings to NPZ format"""
    if not chunks:
        print_error("No chunks to save")
        return

    valid_chunks = [c for c in chunks if c.embedding]
    if not valid_chunks:
        print_error("No valid embeddings to save")
        return
    # Collect all unique metadata keys
    metadata_keys = set()
    for chunk in valid_chunks:
        metadata_keys.update(chunk.metadata.keys())

    # Build arrays for all metadata fields
    arrays = {
        "embeddings": np.array([c.embedding for c in valid_chunks], dtype=np.float32),
        "texts": np.array([c.text for c in valid_chunks], dtype=object),
    }

    for key in metadata_keys:
        arrays[key] = np.array(
            [c.metadata.get(key, "") for c in valid_chunks], dtype=object
        )

    np.savez_compressed(output_path, **arrays)  # type: ignore


def concatenate_embeddings(npz_files: list[Path], output_path: Path) -> None:
    """Concatenate multiple NPZ embedding files into single file"""
    if not npz_files:
        print_error("No NPZ files provided")
        return

    # Collect all arrays from all files
    all_arrays = {}

    for npz_file in npz_files:
        if not npz_file.exists():
            print_error(f"File not found: {npz_file}")
            continue

        with np.load(npz_file, allow_pickle=True) as data:
            for key in data:
                if key not in all_arrays:
                    all_arrays[key] = []
                all_arrays[key].append(data[key])

    if not all_arrays:
        print_error("No valid data found in NPZ files")
        return

    # Concatenate arrays
    concatenated = {}
    for key, arrays in all_arrays.items():
        if key == "embeddings":
            concatenated[key] = np.concatenate(arrays, axis=0).astype(np.float32)
        else:
            concatenated[key] = np.concatenate(arrays, axis=0)

    # Overwrite output file
    np.savez_compressed(output_path, **concatenated)
