import logging
import time

import weaviate

from core.config import settings
from models.weaviate_schemas import create_collections

logger = logging.getLogger(__name__)


def get_weaviate_client():
    """
    Connect to Docker-hosted Weaviate with retries.
    """

    client = weaviate.connect_to_local(
        host=settings.weaviate_host,
        port=settings.weaviate_port,
        grpc_port=settings.weaviate_grpc_port,
    )

    deadline = time.time() + 30
    while time.time() < deadline:
        if client.is_ready():
            logger.info("Connected to Weaviate")
            return client
        logger.info("Weaviate not ready yet, waiting...")
        time.sleep(2)

    raise RuntimeError("Weaviate did not become ready within 30 seconds")


def initialize_weaviate(client, expected_dim: int = 1024):
    create_collections(client)

    try:
        collection = client.collections.get("DocumentChunk")
        if collection.data.exists():
            first_obj = collection.query.fetch_objects(limit=1, return_vector=True)
            if first_obj.objects:
                actual_dim = len(first_obj.objects[0].vector.default)
                if actual_dim != expected_dim:
                    logger.warning(
                        f"Existing DocumentChunk collection has {actual_dim}-dim vectors "
                        f"but the model produces {expected_dim}-dim vectors. "
                        "Existing documents need to be re-ingested for search to work correctly."
                    )
    except Exception as e:
        logger.warning(f"Could not verify vector dimensions: {e}")

    logger.info("Weaviate collections initialized")