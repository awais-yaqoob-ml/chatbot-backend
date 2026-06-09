import os
from pathlib import Path
from uuid import uuid4


def ensure_dir(path: str | Path) -> Path:
    """
    Ensure directory exists.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def generate_doc_id() -> str:
    """
    Generate unique document ID.
    """
    return str(uuid4())


def save_bytes(file_bytes: bytes, output_path: str | Path) -> Path:
    """
    Save raw bytes to disk.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(file_bytes)

    return output_path


def get_file_extension(filename: str) -> str:
    """
    Return lowercase file extension.
    """
    return os.path.splitext(filename)[-1].lower()


def is_pdf(filename: str) -> bool:
    return get_file_extension(filename) == ".pdf"


def is_docx(filename: str) -> bool:
    return get_file_extension(filename) == ".docx"