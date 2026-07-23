import os
import sys
from typing import List, Dict, Any

from embedding.embed_documents import EmbeddingGenerator
from embedding.vector_store import VectorStore

class Retriever:
    def __init__(
        self, 
        collection_name: str = "insurance_claims", 
        persist_directory: str = None
    ):
        if persist_directory is None:
            persist_directory = os.path.join(os.path.dirname(__file__), "..", "vector_db")

        self.generator = EmbeddingGenerator()
        self.store = VectorStore(
            collection_name=collection_name, 
            persist_directory=persist_directory
        )

    def _format_results(self, results) -> List[Dict[str, Any]]:
        formatted_results = []
        if results and results.get("ids") and len(results["ids"]) > 0:
            ids = results["ids"][0]
            distances = results["distances"][0] if results.get("distances") else []
            metadatas = results["metadatas"][0] if results.get("metadatas") else []
            documents = results["documents"][0] if results.get("documents") else []
            
            for i in range(len(ids)):
                formatted_results.append({
                    "id": ids[i],
                    "text": documents[i] if i < len(documents) else "",
                    "metadata": metadatas[i] if i < len(metadatas) else {},
                    "score": distances[i] if i < len(distances) else None
                })
        return formatted_results

    def search(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.generator.embed_text(query_text)
        results = self.store.search(query_embedding, top_k=top_k)
        return self._format_results(results)

    def search_by_claim(self, claim: dict, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.generator.embed_document(claim)["embedding"]
        results = self.store.search(query_embedding, top_k=top_k)
        return self._format_results(results)
