import json
import os
import sys

# Add parent directory to path so we can import from 'embedding' module
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from embedding.embed_documents import EmbeddingGenerator
from embedding.vector_store import VectorStore

def main():
    json_path = os.path.join(os.path.dirname(__file__), "..", "extracted_json", "extracted_claim.json")
    
    if not os.path.exists(json_path):
        print(f"File not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        document = json.load(f)

    print("Generating embedding...")
    generator = EmbeddingGenerator()
    embedded = generator.embed_document(document)

    print("Adding to vector store...")
    # Target the persist_directory at project root
    store = VectorStore(
        collection_name="insurance_claims",
        persist_directory=os.path.join(os.path.dirname(__file__), "..", "vector_db")
    )
    doc_id = store.add_document(embedded)

    print(f"Successfully embedded and stored document with ID: {doc_id}")

if __name__ == "__main__":
    main()
