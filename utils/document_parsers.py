import fitz  # PyMuPDF
from docx import Document
from pathlib import Path
from typing import Dict, List, Tuple


def parse_pdf(file_path: str, output_image_dir: str) -> Dict:
    """
    Extract text and images from PDF.
    """
    doc = fitz.open(file_path)

    full_text = []
    image_paths = []
    pages_processed = doc.page_count

    output_dir = Path(output_image_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for page_index in range(pages_processed):
        page = doc[page_index]

        # text
        text = page.get_text("text")
        full_text.append(text)

        # images
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            image_path = output_dir / f"page_{page_index}_img_{img_index}.png"

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            image_paths.append(str(image_path))

    return {
        "text": "\n".join(full_text),
        "image_paths": image_paths,
        "pages_processed": pages_processed,
    }


def parse_docx(file_path: str, output_image_dir: str) -> Dict:
    """
    Extract text and images from DOCX.
    """
    doc = Document(file_path)

    full_text = []
    image_paths = []

    output_dir = Path(output_image_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # text extraction
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text.strip())

    # image extraction (simplified)
    rels = doc.part._rels

    for rel in rels:
        target = rels[rel].target_ref
        if "media" in target:
            image_data = rels[rel].target_part.blob

            image_path = output_dir / f"{rel}.png"

            with open(image_path, "wb") as f:
                f.write(image_data)

            image_paths.append(str(image_path))

    return {
        "text": "\n".join(full_text),
        "image_paths": image_paths,
        "pages_processed": len(doc.paragraphs),
    }


def parse_document(file_path: str, output_image_dir: str) -> Dict:
    """
    Auto-detect file type and parse.
    """
    if file_path.lower().endswith(".pdf"):
        return parse_pdf(file_path, output_image_dir)

    if file_path.lower().endswith(".docx"):
        return parse_docx(file_path, output_image_dir)

    raise ValueError("Unsupported file format")