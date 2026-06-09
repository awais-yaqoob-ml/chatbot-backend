from core.config import settings


def retrieve_chunks(
    client,
    query_vector,
    top_k=None,
):
    collection = client.collections.get(
        "DocumentChunk"
    )

    top_k = top_k or settings.top_k

    results = collection.query.near_vector(
        near_vector=query_vector,
        limit=top_k,
        return_metadata=["distance"],
    )

    chunks = []

    for obj in results.objects:
        distance = obj.metadata.distance
        similarity = max(0.0, 1.0 - distance)

        chunks.append(
            {
                "doc_id": obj.properties.get("doc_id"),
                "filename": obj.properties.get("filename"),
                "chunk_index": obj.properties.get(
                    "chunk_index"
                ),
                "text": obj.properties.get("text"),
                "page_number": obj.properties.get(
                    "page_number"
                ),
                "image_paths": obj.properties.get(
                    "image_paths",
                    [],
                ),
                "score": similarity,
            }
        )

    return chunks