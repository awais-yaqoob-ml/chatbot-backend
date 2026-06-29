import fitz  # PyMuPDF
from docx import Document
from pathlib import Path
from typing import Dict, List


def parse_pdf(file_path: str, output_image_dir: str) -> Dict:
    """
    Extract text and images from PDF, preserving per-page structure.
    Image tags like ![Image](page_X_img_Y.png) are embedded in the text.
    """
    doc = fitz.open(file_path)

    full_text = []
    image_paths = []
    pages = []
    pages_processed = doc.page_count

    output_dir = Path(output_image_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for page_index in range(pages_processed):
        page = doc[page_index]

        text = page.get_text("text")

        images = page.get_images(full=True)

        page_image_tags = []

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            tag = f"page_{page_index}_img_{img_index}.png"
            image_path = output_dir / tag

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            image_paths.append(str(image_path))
            page_image_tags.append(tag)

        if page_image_tags:
            text += "\n" + "\n".join(f"![Image]({tag})" for tag in page_image_tags)

        full_text.append(text)

        pages.append({
            "page_number": page_index + 1,
            "text": text,
        })

    return {
        "text": "\n".join(full_text),
        "pages": pages,
        "image_paths": image_paths,
        "pages_processed": pages_processed,
    }


def parse_docx(file_path: str, output_image_dir: str) -> Dict:
    """
    Extract text and images from DOCX.
    Image tags are embedded at the end of the text.
    """
    doc = Document(file_path)

    full_text = []
    image_paths = []
    image_tags = []

    output_dir = Path(output_image_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text.strip())

    text_content = "\n".join(full_text)

    rels = doc.part._rels
    for rel in rels:
        target = rels[rel].target_ref
        if "media" in target:
            image_data = rels[rel].target_part.blob
            tag = f"{rel}.png"
            image_path = output_dir / tag
            with open(image_path, "wb") as f:
                f.write(image_data)
            image_paths.append(str(image_path))
            image_tags.append(tag)

    if image_tags:
        text_content += "\n" + "\n".join(f"![Image]({tag})" for tag in image_tags)

    return {
        "text": text_content,
        "pages": [{"page_number": 1, "text": text_content}],
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
