import base64
import mimetypes
import re
from pathlib import Path

from core.config import settings


def resolve_images(answer, retrieved_chunks):
    """
    Scan the LLM answer for ![Image](filename) tags and resolve
    them to base64-encoded images from the retrieved chunks.

    Args:
        answer: The LLM-generated answer text.
        retrieved_chunks: List of chunk dicts with 'doc_id' and 'image_paths'.

    Returns:
        List of dicts with filename, mime_type, and base64 data.
    """
    pattern = r'!\[Image\]\(([^)]+)\)'
    filenames_in_answer = set(re.findall(pattern, answer))

    if not filenames_in_answer:
        return []

    images = []
    seen = set()

    for chunk in retrieved_chunks:
        doc_id = chunk.get("doc_id", "")
        chunk_images = chunk.get("image_paths", []) or []

        for img_file in chunk_images:
            if img_file in filenames_in_answer and img_file not in seen:
                seen.add(img_file)
                image_path = (
                    Path(settings.assets_path)
                    / doc_id
                    / "images"
                    / img_file
                )

                if image_path.exists():
                    file_bytes = image_path.read_bytes()
                    encoded = base64.b64encode(file_bytes).decode()
                    mime_type = (
                        mimetypes.guess_type(str(image_path))[0]
                        or "image/png"
                    )
                    images.append({
                        "filename": img_file,
                        "mime_type": mime_type,
                        "data": encoded,
                    })

    return images
