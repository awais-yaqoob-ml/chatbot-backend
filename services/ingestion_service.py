import os

from utils.file_utils import ensure_dir
from utils.chunking import chunk_text, estimate_page_number
from utils.document_parsers import parse_document
from services.embedding_service import embed_texts


def ingest_document(
    client,
    model,
    file_bytes,
    filename,
    doc_id,
    assets_dir,
):
    collection = client.collections.get("DocumentChunk")

    doc_assets_dir = ensure_dir(
        os.path.join(assets_dir, doc_id, "images")
    )

    temp_path = os.path.join(
        assets_dir,
        doc_id,
        filename,
    )

    ensure_dir(os.path.dirname(temp_path))

    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    parsed = parse_document(
        temp_path,
        str(doc_assets_dir),
    )

    chunks = chunk_text(parsed["text"])

    embeddings = embed_texts(
        model,
        chunks,
    )

    with collection.batch.dynamic() as batch:
        for idx, (chunk, vector) in enumerate(
            zip(chunks, embeddings)
        ):
            batch.add_object(
                properties={
                    "doc_id": doc_id,
                    "filename": filename,
                    "chunk_index": idx,
                    "text": chunk,
                    "page_number": estimate_page_number(idx),
                    "has_images": len(parsed["image_paths"]) > 0,
                    "image_paths": parsed["image_paths"],
                },
                vector=vector,
            )

    return {
        "doc_id": doc_id,
        "filename": filename,
        "total_chunks": len(chunks),
        "images_extracted": len(parsed["image_paths"]),
        "pages_processed": parsed["pages_processed"],
    }