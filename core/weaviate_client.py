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


def initialize_weaviate(client):
    create_collections(client)
    logger.info("Weaviate collections initialized")