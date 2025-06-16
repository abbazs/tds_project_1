# app/embedder/stella.py
import asyncio
from pathlib import Path
from typing import List
from typing import Optional

import torch
from sentence_transformers import SentenceTransformer

from app.utils import print_error
from app.utils import print_success


class StellaEmbedder:
    """Local embedder using stella_en_400M_v5 with sentence-transformers"""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        """
        Initialize Stella embedder using sentence-transformers.

        Args:
            model_path: Path to local model. If None, downloads from HuggingFace
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            batch_size: Batch size for embedding multiple texts
        """
        self.batch_size = batch_size

        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Load model
        if model_path and Path(model_path).exists():
            self.model = SentenceTransformer(model_path, device=self.device)
            print_success(f"Loaded Stella model from {model_path}")
        else:
            print_success("Downloading Stella model from HuggingFace...")
            self.model = SentenceTransformer(
                "dunzhang/stella_en_400M_v5", device=self.device, trust_remote_code=True
            )

        print_success(f"Stella embedder initialized on {self.device}")

    async def embed_text(self, text: str) -> List[float]:
        """
        Embed single text asynchronously (compatible with OpenAI embedder interface).

        Args:
            text: Text to embed

        Returns:
            Embedding as list of floats
        """
        if not text or not text.strip():
            return []

        # Run synchronous embedding in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, lambda: self.model.encode(text, convert_to_numpy=True).tolist()
        )
        return embedding

    async def embed_texts_async(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts asynchronously in batches.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings
        """
        if not texts:
            return []

        # Run batch embedding in thread pool
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self.model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=True,
            ).tolist(),
        )
        return embeddings

    def save_model(self, save_path: str):
        """Save model locally for faster loading next time"""
        try:
            save_dir = Path(save_path)
            save_dir.mkdir(parents=True, exist_ok=True)
            self.model.save(str(save_dir))
            print_success(f"Model saved to {save_dir}")
        except Exception as e:
            print_error(f"Failed to save model: {e}")
            raise
