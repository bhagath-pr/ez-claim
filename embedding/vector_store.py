"""
vector_store.py

Persistent vector storage using ChromaDB.

Responsibilities
----------------
- Store EmbeddedDocument objects
- Perform semantic similarity search
- Delete documents
- Count stored documents
- Reset the collection

This module intentionally has NO dependency on Hugging Face or
DocumentBuilder. It only works with EmbeddedDocument objects.
"""

import logging
import uuid
from typing import List

import chromadb

from .embed_documents import EmbeddedDocument

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Base exception for vector store operations."""


class VectorStore:
    """
    Wrapper around a ChromaDB collection.
    """

    def __init__(
        self,
        collection_name: str = "insurance_claims",
        persist_directory: str = "./vector_db",
    ):

        self.client = chromadb.PersistentClient(
            path=persist_directory
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

        self.embedding_dimension = None

        logger.info(
            "Connected to ChromaDB collection '%s'",
            collection_name
        )

    # ---------------------------------------------------------
    # Internal
    # ---------------------------------------------------------

    def _validate_dimension(
        self,
        embedding: List[float]
    ):

        if self.embedding_dimension is None:

            self.embedding_dimension = len(embedding)
            return

        if len(embedding) != self.embedding_dimension:

            raise VectorStoreError(
                f"Inconsistent embedding dimension. "
                f"Expected {self.embedding_dimension}, "
                f"received {len(embedding)}."
            )

    # ---------------------------------------------------------
    # Add
    # ---------------------------------------------------------

    def add_document(
        self,
        document: EmbeddedDocument,
        document_id: str | None = None,
    ) -> str:

        self._validate_dimension(document["embedding"])

        if document_id is None:
            document_id = str(uuid.uuid4())

        self.collection.add(
            ids=[document_id],
            embeddings=[document["embedding"]],
            documents=[document["text"]],
            metadatas=[document["metadata"]],
        )

        logger.info(
            "Stored document %s",
            document_id
        )

        return document_id

    # ---------------------------------------------------------

    def add_documents(
        self,
        documents: List[EmbeddedDocument],
    ) -> List[str]:

        if not documents:
            return []

        ids = []

        embeddings = []

        texts = []

        metadatas = []

        for doc in documents:

            self._validate_dimension(doc["embedding"])

            ids.append(str(uuid.uuid4()))

            embeddings.append(doc["embedding"])

            texts.append(doc["text"])

            metadatas.append(doc["metadata"])

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        logger.info(
            "Stored %d documents",
            len(ids)
        )

        return ids

    # ---------------------------------------------------------
    # Search
    # ---------------------------------------------------------

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ):

        self._validate_dimension(query_embedding)

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

    # ---------------------------------------------------------
    # Delete
    # ---------------------------------------------------------

    def delete_document(
        self,
        document_id: str,
    ):

        self.collection.delete(
            ids=[document_id]
        )

        logger.info(
            "Deleted document %s",
            document_id
        )

    # ---------------------------------------------------------
    # Utility
    # ---------------------------------------------------------

    def count(self) -> int:

        return self.collection.count()

    # ---------------------------------------------------------

    def reset(self):

        ids = self.collection.get()["ids"]

        if ids:
            self.collection.delete(ids=ids)

        logger.warning(
            "Collection cleared."
        )

    # ---------------------------------------------------------

    def peek(self, limit: int = 5):

        return self.collection.peek(limit=limit)