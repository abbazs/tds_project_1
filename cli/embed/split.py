import re
from typing import List


class URLAwareTextSplitter:
    """Text splitter preserving URLs urls with proper chunking.
    If the url is too long, it will be split into multiple chunks.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.url_pattern = re.compile(
            r"https?://[^\s\[\]()]+|ftp://[^\s\[\]()]+|\[[^\]]*\]\([^)]+\)",
            re.IGNORECASE,
        )
        # Hierarchical separators - order matters for quality splits
        self.separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks preserving URLs"""
        # Early return for small texts
        if len(text) <= self.chunk_size:
            return [text]

        # Tokenize URLs
        url_map = {}
        for i, match in enumerate(self.url_pattern.finditer(text)):
            token = f"__URL_{i}__"
            url_map[token] = match.group()
            text = text[: match.start()] + token + text[match.end() :]

        return [self._restore_urls(chunk, url_map) for chunk in self._chunk_text(text)]

    def _chunk_text(self, text: str) -> List[str]:
        """Core chunking logic with proper overlap"""
        chunks = []
        start = 0

        while start < len(text):
            # Calculate chunk boundary
            end = min(start + self.chunk_size, len(text))

            # Find best split point if not at text end
            if end < len(text):
                end = self._find_split_point(text, start, end)

            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)

            next_start = end - self.chunk_overlap if end < len(text) else len(text)
            start = max(next_start, start + 1)  # Guarantee forward progress

        return chunks

    def _find_split_point(self, text: str, start: int, end: int) -> int:
        """Find optimal split point using hierarchical separators"""
        for separator in self.separators:
            # Look for separator in the last quarter of the chunk
            search_start = max(start, end - self.chunk_size // 4)
            pos = text.rfind(separator, search_start, end)
            if pos > start:
                return pos + len(separator)
        return end  # Fallback to hard cut

    def _restore_urls(self, chunk: str, url_map: dict) -> str:
        """Restore original URLs from tokens"""
        for token, url in url_map.items():
            chunk = chunk.replace(token, url)
        return chunk
