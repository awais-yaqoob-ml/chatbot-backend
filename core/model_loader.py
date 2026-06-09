from functools import lru_cache
import logging

from sentence_transformers import SentenceTransformer

from core.config import settings

logger = logging.getLogger(__name__)


@lru_cache
def load_embedding_model() -> SentenceTransformer:
    """
    Load sentence-transformer model once at startup.
    Cached to avoid reloading.
    """
    logger.info(f"Loading embedding model: {settings.embed_model}")

    model = SentenceTransformer(settings.embed_model)

    logger.info("Embedding model loaded successfully")
    return model


def encode_text(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """
    Encode texts into embeddings.
    """
    return model.encode(texts, normalize_embeddings=True).tolist()