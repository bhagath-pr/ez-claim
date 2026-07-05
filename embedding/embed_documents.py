"""
embed_documents.py

Generates semantic embeddings from structured JSON documents and raw text
using a Hugging Face Sentence Transformer model.

Pipeline

JSON 
    ↓
DocumentBuilder
    ↓
Readable Text
    ↓
Embedding Model
    ↓
Embedding Vector
"""

import logging
from typing import Any, Dict, List, TypedDict

from sentence_transformers import SentenceTransformer

# Assuming this module is part of a package (e.g., your_package.embedding)
from .document_builder import DocumentBuilder

logger = logging.getLogger(__name__)


class EmbeddedDocument(TypedDict):
    """Type definition for the output of document embedding methods."""
    text: str
    embedding: List[float]
    dimension: int
    metadata: Dict[str, Any]


class EmbeddingGenerator:
    """
    Generates semantic embeddings from structured JSON documents and raw queries.
    """

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(
        self, 
        model_name: str = DEFAULT_MODEL,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        show_progress_bar: bool = False
    ):
        self.builder = DocumentBuilder()
        
        # Expose encode parameters for flexibility
        self.normalize_embeddings = normalize_embeddings
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar

        logger.info("Loading embedding model: %s", model_name)
        
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(
                f"Unable to load model '{model_name}'. Check your internet connection or local model cache."
            ) from e
            
        logger.info("Model loaded successfully.")

    def embed_text(self, text: str) -> List[float]:
        """
        Convert a single text string into an embedding vector.
        Essential for embedding user queries before vector search (RAG).
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True
        )
        return embedding.tolist()

    def embed_document(self, document: Dict[str, Any]) -> EmbeddedDocument:
        """
        Convert a single structured JSON document into an embedding dictionary.
        """
        text = self.builder.build(document)
        embedding_list = self.embed_text(text)

        return {
            "text": text,
            "embedding": embedding_list,
            "dimension": len(embedding_list),
            # TODO: Once schema is finalized, extract only IDs (e.g., patient_id, claim_id) 
            # to prevent duplicating large JSON payloads in the vector DB.
            "metadata": document 
        }

    def embed_documents(self, documents: List[Dict[str, Any]]) -> List[EmbeddedDocument]:
        """
        Convert multiple structured JSON documents into normalized embedding vectors 
        using batch inference.
        """
        if not documents:
            logger.warning("Empty document list provided. Returning empty list.")
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
                "metadata": document  # TODO: Optimize metadata storage (see above)
            })

        return results