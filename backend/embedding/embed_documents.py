import logging
from typing import Any, Dict, List, TypedDict

from sentence_transformers import SentenceTransformer
from .document_builder import DocumentBuilder

logger = logging.getLogger(__name__)

class EmbeddedDocument(TypedDict):
    text: str
    embedding: List[float]
    dimension: int
    metadata: Dict[str, Any]

class EmbeddingGenerator:
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self, 
        model_name: str = DEFAULT_MODEL,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        show_progress_bar: bool = False
    ):
        self.builder = DocumentBuilder()
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar

        logger.info("Loading embedding model: %s", model_name)
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(
                f"Unable to load model '{model_name}'. Check internet connection or model cache."
            ) from e

    def embed_text(self, text: str) -> List[float]:
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True
        )
        return embedding.tolist()

    def embed_document(self, document: Dict[str, Any]) -> EmbeddedDocument:
        text = self.builder.build(document)
        embedding_list = self.embed_text(text)

        return {
            "text": text,
            "embedding": embedding_list,
            "dimension": len(embedding_list),
            "metadata": {
                "policy_id": document.get("policy", {}).get("policy_id", "unknown")
            }
        }

    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[EmbeddedDocument]:
        if not documents:
            return []

        texts = [self.builder.build(doc) for doc in documents]

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=self.show_progress_bar,
        )

        results = []
        for text, embedding, document in zip(texts, embeddings, documents):
            embed_list = embedding.tolist()
            results.append({
                "text": text,
                "embedding": embed_list,
                "dimension": len(embed_list),
                "metadata": {
                    "policy_id": document.get("policy", {}).get("policy_id", "unknown")
                }
            })

        return results
