"""
retriever.py

Retrieval module that bridges the gap between raw text queries and the vector store.
"""

import os
import sys
from typing import List, Dict, Any

# Allow direct execution of this file
if __name__ == "__main__":
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from embedding.embed_documents import EmbeddingGenerator
from embedding.vector_store import VectorStore

class Retriever:
    """
    High-level interface to search the vector database using raw text queries.
    """
    def __init__(
        self, 
        collection_name: str = "insurance_claims", 
        persist_directory: str = None
    ):
        if persist_directory is None:
            # Default to the vector_db folder at the root of the project
            persist_directory = os.path.join(os.path.dirname(__file__), "..", "vector_db")

        self.generator = EmbeddingGenerator()
        self.store = VectorStore(
            collection_name=collection_name, 
            persist_directory=persist_directory
        )

    def _format_results(self, results) -> List[Dict[str, Any]]:
        """Helper to format ChromaDB results into a list of dicts."""
        formatted_results = []
        
        # ChromaDB query returns lists of lists since it supports batch querying
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
        """
        Search the vector store for documents semantically similar to the query text.
        
        Args:
            query_text: The raw text string to search for.
            top_k: Number of top results to return.
            
        Returns:
            A list of dictionaries representing the most relevant documents, 
            including their id, text, metadata, and similarity score (distance).
        """
        # Convert text query to embedding
        query_embedding = self.generator.embed_text(query_text)
        
        # Search the vector store
        results = self.store.search(query_embedding, top_k=top_k)
        
        return self._format_results(results)

    def search_by_claim(self, claim: dict, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the vector store for documents semantically similar to the structured claim dict.
        
        Args:
            claim: The structured claim dictionary to search for.
            top_k: Number of top results to return.
            
        Returns:
            A list of dictionaries representing the most relevant documents.
        """
        # Convert structured claim to embedding
        query_embedding = self.generator.embed_document(claim)["embedding"]
        
        # Search the vector store
        results = self.store.search(query_embedding, top_k=top_k)
        
        return self._format_results(results)

if __name__ == "__main__":
    print("Testing Retriever...")
    retriever = Retriever()
    # Search for our recently embedded document
    results = retriever.search("Patient admitted in ICU for dengue")
    
    if not results:
        print("No results found. (Did you run tests.run_embedding first?)")
    else:
        print(f"Found {len(results)} results:")
        for idx, res in enumerate(results, 1):
            print(f"\n--- Result {idx} ---")
            print(f"Score: {res['score']}")
            print(f"ID: {res['id']}")
            print(f"Metadata: {res['metadata']}")
            print(f"Text Snippet: {res['text'][:200]}...")
