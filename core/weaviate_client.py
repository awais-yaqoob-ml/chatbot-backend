import logging

import weaviate

from core.config import settings
from models.weaviate_schemas import create_collections

logger = logging.getLogger(__name__)


def get_weaviate_client():
    """
    Connect to Docker-hosted Weaviate.
    """

    client = weaviate.connect_to_local(
        host=settings.weaviate_host,
        port=settings.weaviate_port,
        grpc_port=settings.weaviate_grpc_port,
    )

    if not client.is_ready():
        raise RuntimeError("Weaviate is not ready")

    logger.info("Connected to Weaviate")

    return client


def initialize_weaviate(client):
    create_collections(client)
    logger.info("Weaviate collections initialized")