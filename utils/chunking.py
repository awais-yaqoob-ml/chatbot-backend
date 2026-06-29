from typing import List


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping word-based chunks.
    """
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = words[start:end]
        if not chunk:
            break

        chunks.append(" ".join(chunk))
        start = end - overlap

        if start < 0:
            start = 0

    return chunks
