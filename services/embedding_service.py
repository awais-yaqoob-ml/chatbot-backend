from sentence_transformers import SentenceTransformer
from typing import List


def embed_texts(model: SentenceTransformer, texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts.
    """
    embeddings = model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(model: SentenceTransformer, text: str) -> List[float]:
    """
    Generate embedding for a single query.
    """
    return model.encode([text], normalize_embeddings=True)[0].tolist()