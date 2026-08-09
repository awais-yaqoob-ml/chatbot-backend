import base64
import re
from pathlib import Path

from core.config import settings

# (magic bytes prefix, mime type). Checked in order.
_MAGIC_MIME = [
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
]

_TAG_PATTERN = r'!\[([^\]]+)\]\(([^)]+)\)'


def _parse_img_tag(filename: str):
    """Extract (page_index, img_index) from filenames like
    page_9_img_0.png, page10img_0.png, page_10_img0.png, etc."""
    m = re.search(r'page_?(\d+)_?img_?(\d+)', filename, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _detect_mime(data: bytes) -> str:
    """Detect image MIME type from magic bytes instead of the file extension.
    Extracted images are often written with a .png extension even when the
    underlying bytes are JPEG, so extension-based guessing is unreliable."""
    for magic, mime in _MAGIC_MIME:
        if data.startswith(magic):
            return mime
    if len(data) >= 12 and data[0:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _load_image(doc_id: str, filename: str):
    """Load and base64-encode an image from disk."""
    image_path = Path(settings.assets_path) / doc_id / "images" / filename
    if not image_path.exists():
        return None
    file_bytes = image_path.read_bytes()
    encoded = base64.b64encode(file_bytes).decode()
    mime_type = _detect_mime(file_bytes)
    return {
        "filename": filename,
        "mime_type": mime_type,
        "data": encoded,
    }


def resolve_images(answer, retrieved_chunks):
    """
    Scan the LLM answer for ![Image](filename) tags and resolve
    them to base64-encoded images from the retrieved chunks.

    Uses fuzzy matching (by page/image numbers) as a fallback when
    the LLM modifies the exact filename format.

    Returns:
        Tuple of (cleaned_answer, list_of_image_dicts).
    """
    # Build lookup: stored filename -> {doc_id, page_index, img_index}
    stored_map = {}
    index_to_file = {}

    for chunk in retrieved_chunks:
        doc_id = chunk.get("doc_id", "")
        for img_file in (chunk.get("image_paths", []) or []):
            if img_file in stored_map:
                continue
            nums = _parse_img_tag(img_file)
            if nums:
                page_idx, img_idx = nums
                stored_map[img_file] = {
                    "doc_id": doc_id,
                    "page_index": page_idx,
                    "img_index": img_idx,
                }
                # 0-indexed lookup
                index_to_file[(page_idx, img_idx)] = img_file
                # 1-indexed lookup (the LLM often adjusts page numbers)
                index_to_file[(page_idx + 1, img_idx)] = img_file

    if not stored_map:
        return answer, []

    # Scan the answer for image reference tags like ![Image](filename),
    # ![Screenshot](filename), etc. Broad matching so small LLM deviations
    # from the exact tag format still get resolved.
    refs_in_answer = re.findall(_TAG_PATTERN, answer)

    images = []
    seen_filenames = set()
    resolved_refs = {}  # reference url -> base64 <img> tag or removal marker

    for _, ref in refs_in_answer:
        if ref in resolved_refs:
            continue

        # 1. Try exact filename match
        matched_file = ref if ref in stored_map else None

        # 2. Try fuzzy match by extracted numbers
        if not matched_file:
            nums = _parse_img_tag(ref)
            if nums and nums in index_to_file:
                matched_file = index_to_file[nums]

        if not matched_file:
            # Can't resolve this reference — mark for removal
            resolved_refs[ref] = ""
            continue

        if matched_file in seen_filenames:
            resolved_refs[ref] = ""
            continue

        seen_filenames.add(matched_file)
        info = stored_map[matched_file]
        img_data = _load_image(info["doc_id"], matched_file)
        if img_data:
            images.append(img_data)
            resolved_refs[ref] = (
                f'<img src="data:{img_data["mime_type"]};'
                f'base64,{img_data["data"]}" '
                f'alt="{matched_file}" />'
            )
        else:
            resolved_refs[ref] = ""

    # Build cleaned answer: replace resolved tags with <img>, remove unresolved
    def _replace_tag(m):
        ref = m.group(2)
        return resolved_refs.get(ref, "")

    cleaned = re.sub(_TAG_PATTERN, _replace_tag, answer)

    return cleaned, images
