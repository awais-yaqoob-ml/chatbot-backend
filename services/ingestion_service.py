import os
import re
from datetime import datetime

from utils.file_utils import ensure_dir
from utils.chunking import chunk_text
from utils.document_parsers import parse_document
from services.embedding_service import embed_texts
from services.company_profile_service import (
    generate_doc_summary,
    consolidate_company_profile,
)


def _extract_image_tags(text: str) -> list:
    """Extract image filenames from ![Image](filename) tags in text."""
    return re.findall(r'!\[Image\]\(([^)]+)\)', text)


def ingest_document(
    client,
    model,
    llm,
    file_bytes,
    filename,
    doc_id,
    assets_dir,
):
    chunk_collection = client.collections.get("DocumentChunk")

    doc_assets_dir = ensure_dir(
        os.path.join(assets_dir, doc_id, "images")
    )

    temp_path = os.path.join(assets_dir, doc_id, filename)

    ensure_dir(os.path.dirname(temp_path))

    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    parsed = parse_document(temp_path, str(doc_assets_dir))

    chunks = []
    chunk_page_numbers = []

    pages = parsed.get("pages")
    if pages:
        for page in pages:
            page_chunks = chunk_text(page["text"])
            chunks.extend(page_chunks)
            chunk_page_numbers.extend([page["page_number"]] * len(page_chunks))
    else:
        chunks = chunk_text(parsed["text"])
        chunk_page_numbers = [1] * len(chunks)

    embeddings = embed_texts(model, chunks)

    with chunk_collection.batch.dynamic() as batch:
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            chunk_image_tags = _extract_image_tags(chunk)

            batch.add_object(
                properties={
                    "doc_id": doc_id,
                    "filename": filename,
                    "chunk_index": idx,
                    "text": chunk,
                    "page_number": chunk_page_numbers[idx],
                    "has_images": len(chunk_image_tags) > 0,
                    "image_paths": chunk_image_tags,
                },
                vector=vector,
            )

    doc_summary = generate_doc_summary(llm, parsed["text"], filename)

    doc_collection = client.collections.get("Document")
    doc_collection.data.insert(
        properties={
            "doc_id": doc_id,
            "filename": filename,
            "doc_summary": doc_summary["summary"],
            "topics": doc_summary["topics"],
            "uploaded_at": datetime.utcnow().isoformat(),
            "chunk_count": len(chunks),
        }
    )

    consolidate_company_profile(client, llm)

    return {
        "doc_id": doc_id,
        "filename": filename,
        "total_chunks": len(chunks),
        "images_extracted": len(parsed["image_paths"]),
        "pages_processed": parsed["pages_processed"],
    }
