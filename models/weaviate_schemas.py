from weaviate.classes.config import (
    Configure,
    Property,
    DataType,
)


DOCUMENT_CHUNK_COLLECTION = {
    "name": "DocumentChunk",
    "properties": [
        Property(name="doc_id", data_type=DataType.TEXT),
        Property(name="filename", data_type=DataType.TEXT),
        Property(name="chunk_index", data_type=DataType.INT),
        Property(name="text", data_type=DataType.TEXT),
        Property(name="page_number", data_type=DataType.INT),
        Property(name="has_images", data_type=DataType.BOOL),
        Property(name="image_paths", data_type=DataType.TEXT_ARRAY),
    ],
}

CHAT_HISTORY_COLLECTION = {
    "name": "ChatHistory",
    "properties": [
        Property(name="session_id", data_type=DataType.TEXT),
        Property(name="role", data_type=DataType.TEXT),
        Property(name="content", data_type=DataType.TEXT),
        Property(name="timestamp", data_type=DataType.TEXT),
        Property(name="agent_used", data_type=DataType.TEXT),
    ],
}


def create_collections(client):
    existing = client.collections.list_all()

    if "DocumentChunk" not in existing:
        client.collections.create(
            name="DocumentChunk",
            vectorizer_config=Configure.Vectorizer.none(),
            properties=DOCUMENT_CHUNK_COLLECTION["properties"],
        )

    if "ChatHistory" not in existing:
        client.collections.create(
            name="ChatHistory",
            vectorizer_config=Configure.Vectorizer.none(),
            properties=CHAT_HISTORY_COLLECTION["properties"],
        )