import logging
import uuid
from typing import List
import chromadb
from .embed_documents import EmbeddedDocument

logger = logging.getLogger(__name__)

class VectorStoreError(Exception):
    pass

class VectorStore:
    def __init__(
        self,
        collection_name: str = "insurance_claims",
        persist_directory: str = "./vector_db",
    ):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedding_dimension = None

    def _validate_dimension(self, embedding: List[float]):
        if self.embedding_dimension is None:
            self.embedding_dimension = len(embedding)
            return

        if len(embedding) != self.embedding_dimension:
            raise VectorStoreError(
                f"Inconsistent embedding dimension. Expected {self.embedding_dimension}, received {len(embedding)}."
            )

    def add_document(self, document: EmbeddedDocument, document_id: str | None = None) -> str:
        self._validate_dimension(document["embedding"])
        if document_id is None:
            document_id = str(uuid.uuid4())

        self.collection.add(
            ids=[document_id],
            embeddings=[document["embedding"]],
            documents=[document["text"]],
            metadatas=[document["metadata"]],
        )
        return document_id

    def add_documents(self, documents: List[EmbeddedDocument]) -> List[str]:
        if not documents:
            return []

        ids, embeddings, texts, metadatas = [], [], [], []
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
        return ids

    def search(self, query_embedding: List[float], top_k: int = 5):
        self._validate_dimension(query_embedding)
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

    def delete_document(self, document_id: str):
        self.collection.delete(ids=[document_id])

    def count(self) -> int:
        return self.collection.count()

    def reset(self):
        ids = self.collection.get()["ids"]
        if ids:
            self.collection.delete(ids=ids)

    def peek(self, limit: int = 5):
        return self.collection.peek(limit=limit)
