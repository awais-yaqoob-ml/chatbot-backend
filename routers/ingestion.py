import os
from uuid import uuid4

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends

from core.config import settings
from core.dependencies import get_weaviate_client_dep, get_embed_model
from models.schemas import IngestionResponse
from services.ingestion_service import ingest_document
from utils.file_utils import is_pdf, is_docx, generate_doc_id, ensure_dir

router = APIRouter(prefix="/api/v1", tags=["ingestion"])


@router.post("/ingest", response_model=IngestionResponse)
async def ingest_file(
    file: UploadFile = File(...),
    client=Depends(get_weaviate_client_dep),
    model=Depends(get_embed_model),
):
    filename = file.filename

    if not (is_pdf(filename) or is_docx(filename)):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX supported")

    file_bytes = await file.read()

    if len(file_bytes) > settings.max_upload_size_bytes:
        raise HTTPException(status_code=413, detail="File too large")

    doc_id = generate_doc_id()

    ensure_dir(settings.assets_path / doc_id)

    result = ingest_document(
        client=client,
        model=model,
        file_bytes=file_bytes,
        filename=filename,
        doc_id=doc_id,
        assets_dir=str(settings.assets_path),
    )

    return IngestionResponse(
        doc_id=result["doc_id"],
        filename=result["filename"],
        total_chunks=result["total_chunks"],
        images_extracted=result["images_extracted"],
        pages_processed=result["pages_processed"],
    )