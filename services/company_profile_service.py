import logging
from datetime import datetime

from core.config import settings
from services.llm_service import build_messages, run_llm

logger = logging.getLogger(__name__)


def generate_doc_summary(llm, doc_text: str, filename: str) -> dict:
    """
    Generate a structured summary of a single document using the LLM.
    Returns dict with 'summary' and 'topics'.
    """
    if not settings.company_profile_enabled:
        return {"summary": "", "topics": []}

    max_chars = 10000
    truncated = doc_text[:max_chars]

    system_prompt = (
        "You are a document analysis assistant. "
        "Given the following document text, produce a concise summary and key topics.\n\n"
        "Return your response in this exact format:\n"
        "SUMMARY:\n[2-3 sentence summary of what this document covers]\n\n"
        "TOPICS:\n[comma-separated list of key topics, policies, or subjects discussed]"
    )

    messages = build_messages(
        system_prompt=system_prompt,
        chat_history=[],
        user_message=f"Document filename: {filename}\n\nDocument content:\n{truncated}",
    )

    try:
        response = run_llm(llm, messages)
        summary = ""
        topics = []

        parts = response.split("TOPICS:")
        summary_section = parts[0].replace("SUMMARY:", "").strip()
        summary = summary_section

        if len(parts) > 1:
            topics_raw = parts[1].strip()
            topics = [t.strip() for t in topics_raw.split(",") if t.strip()]

        logger.info(f"Generated summary for {filename}: {summary[:80]}...")
        return {"summary": summary, "topics": topics}

    except Exception as e:
        logger.exception(f"Failed to generate doc summary for {filename}")
        return {"summary": "", "topics": []}


def consolidate_company_profile(client, llm):
    """
    Fetch all Document summaries and consolidate into a single company profile.
    Stores the result in the CompanyProfile collection.
    """
    if not settings.company_profile_enabled:
        return

    try:
        collection = client.collections.get("Document")
        response = collection.query.fetch_objects(limit=100)

        doc_summaries = []
        all_topics = set()

        for obj in response.objects:
            props = obj.properties
            summary = props.get("doc_summary", "")
            topics = props.get("topics", []) or []

            if summary:
                doc_summaries.append(f"Document: {props.get('filename', 'unknown')}\n{summary}")
            for t in topics:
                if t:
                    all_topics.add(t)

        if not doc_summaries:
            logger.info("No document summaries to consolidate")
            return

        combined = "\n\n".join(doc_summaries)

        system_prompt = (
            "You are a company profile generator. "
            "Given summaries of all documents in a company's knowledge base, "
            "produce a consolidated company overview in 2-3 paragraphs. "
            "Cover: what the company does, key policies, products/services, "
            "and any other notable information. Be factual and concise."
        )

        messages = build_messages(
            system_prompt=system_prompt,
            chat_history=[],
            user_message=f"Document summaries:\n\n{combined}",
        )

        profile_summary = run_llm(llm, messages)

        profile_collection = client.collections.get("CompanyProfile")
        profile_collection.data.insert(
            properties={
                "profile_id": "_primary",
                "summary": profile_summary,
                "topics": list(all_topics),
                "last_updated": datetime.utcnow().isoformat(),
            }
        )

        logger.info("Company profile consolidated and stored")

    except Exception as e:
        logger.exception("Failed to consolidate company profile")


def get_company_profile(client) -> str | None:
    """
    Retrieve the consolidated company profile summary.
    Returns None if no profile exists.
    """
    try:
        collection = client.collections.get("CompanyProfile")
        response = collection.query.fetch_objects(
            where={"path": ["profile_id"], "operator": "Equal", "valueText": "_primary"},
            limit=1,
        )
        for obj in response.objects:
            return obj.properties.get("summary")
        return None
    except Exception:
        return None


def delete_document(client, doc_id: str):
    """
    Delete a document and all its chunks from Weaviate.
    """
    try:
        chunk_collection = client.collections.get("DocumentChunk")
        chunk_collection.data.delete_many(
            where={"path": ["doc_id"], "operator": "Equal", "valueText": doc_id}
        )

        doc_collection = client.collections.get("Document")
        doc_collection.data.delete_many(
            where={"path": ["doc_id"], "operator": "Equal", "valueText": doc_id}
        )

        logger.info(f"Deleted document {doc_id}")
        return True
    except Exception as e:
        logger.exception(f"Failed to delete document {doc_id}")
        return False


def list_documents(client) -> list[dict]:
    """
    List all ingested documents.
    """
    try:
        collection = client.collections.get("Document")
        response = collection.query.fetch_objects(limit=100)
        docs = []
        for obj in response.objects:
            props = obj.properties
            docs.append({
                "doc_id": props.get("doc_id"),
                "filename": props.get("filename"),
                "doc_summary": props.get("doc_summary", ""),
                "topics": props.get("topics", []),
                "uploaded_at": props.get("uploaded_at", ""),
                "chunk_count": props.get("chunk_count", 0),
            })
        return docs
    except Exception as e:
        logger.exception("Failed to list documents")
        return []
