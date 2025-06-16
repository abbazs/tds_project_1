# app/embed/split.py - Improved version
import re
from typing import List


class URLAwareTextSplitter:
    """Improved text splitter that handles URLs and creates consistent chunks"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Simplified separators in order of preference
        self.separators = ["\n\n", "\n", ". ", "! ", "? ", ", ", " "]

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks while trying to preserve sentences"""
        if not text or len(text) <= self.chunk_size:
            return [text] if text else []

        chunks = []
        start = 0

        while start < len(text):
            # Determine end position for this chunk
            end = min(start + self.chunk_size, len(text))

            # If we're not at the end of text, find a good split point
            if end < len(text):
                # Look for a separator in the last 20% of the chunk
                search_start = end - int(self.chunk_size * 0.2)
                split_point = self._find_split_point(text, search_start, end)

                # If no good split found, just split at chunk_size
                if split_point == -1:
                    split_point = end
            else:
                split_point = end

            # Extract chunk and add to results
            chunk = text[start:split_point].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position for next chunk
            if split_point < len(text):
                # Apply overlap
                start = max(split_point - self.chunk_overlap, start + 1)
            else:
                break

        return chunks

    def _find_split_point(self, text: str, search_start: int, end: int) -> int:
        """Find the best position to split text, looking for separators"""
        best_pos = -1

        # Try each separator in order of preference
        for separator in self.separators:
            pos = text.rfind(separator, search_start, end)
            if pos != -1:
                # Found a separator, return position after it
                return pos + len(separator)

        return best_pos


class RecursiveTextSplitter:
    """Alternative: Recursive character text splitter (simpler, often more reliable)"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        """Split text recursively using separators"""
        if not text:
            return []

        # If text fits in chunk size, return as is
        if len(text) <= self.chunk_size:
            return [text]

        # Try to split using separators in order
        for separator in self.separators:
            if separator == "":
                # Last resort: split by chunk size
                return self._split_by_size(text)

            if separator in text:
                chunks = []
                splits = text.split(separator)
                current_chunk = ""

                for i, split in enumerate(splits):
                    # Add separator back except for last split
                    if i < len(splits) - 1:
                        split += separator

                    # Check if adding this split exceeds chunk size
                    if len(current_chunk) + len(split) > self.chunk_size:
                        if current_chunk:
                            chunks.extend(self.split_text(current_chunk.strip()))
                        current_chunk = split
                    else:
                        current_chunk += split

                # Don't forget the last chunk
                if current_chunk:
                    chunks.extend(self.split_text(current_chunk.strip()))

                return self._merge_chunks(chunks)

        # If no separators found, split by size
        return self._split_by_size(text)

    def _split_by_size(self, text: str) -> List[str]:
        """Split text by chunk size as last resort"""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

    def _merge_chunks(self, chunks: List[str]) -> List[str]:
        """Merge small chunks and apply overlap"""
        if not chunks:
            return []

        merged = []
        current = chunks[0]

        for next_chunk in chunks[1:]:
            # If current chunk is too small, merge with next
            if (
                len(current) < self.chunk_size // 2
                and len(current) + len(next_chunk) <= self.chunk_size
            ):
                current += " " + next_chunk
            else:
                merged.append(current)
                # Apply overlap by including end of current chunk in next
                if len(current) > self.chunk_overlap:
                    overlap_text = current[-self.chunk_overlap :]
                    current = overlap_text + " " + next_chunk
                else:
                    current = next_chunk

        if current:
            merged.append(current)

        return merged
