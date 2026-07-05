import json

from embedding.document_builder import DocumentBuilder
from embedding.embed_documents import EmbeddingGenerator
from embedding.vector_store import VectorStore


JSON_FILE = "data/claims.jsonl"


def main():

    builder = DocumentBuilder()

    generator = EmbeddingGenerator()

    store = VectorStore()

    with open(JSON_FILE, "r", encoding="utf-8") as file:

        first_line = file.readline()

    document = json.loads(first_line)

    print("=" * 60)
    print("ORIGINAL JSON")
    print("=" * 60)
    print(document)

    print()

    text = builder.build(document)

    print("=" * 60)
    print("DOCUMENT BUILDER OUTPUT")
    print("=" * 60)
    print(text)

    print()

    embedded = generator.embed_document(document)

    print("=" * 60)
    print("EMBEDDING")
    print("=" * 60)
    print("Dimension:", embedded["dimension"])

    document_id = store.add_document(embedded)

    print()

    print("=" * 60)
    print("VECTOR STORE")
    print("=" * 60)
    print("Stored ID:", document_id)
    print("Documents in DB:", store.count())

    print()
    print("=" * 60)
    print("SEARCH")
    print("=" * 60)

    query = "cashless diabetes insurance claim"

    query_embedding = generator.embed_text(query)

    results = store.search(query_embedding)

    print(f"Found {len(results['ids'][0])} result(s)\n")

    for i, (doc_id, metadata, distance) in enumerate(
        zip(
            results["ids"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        start=1,
    ):
        print(f"Result {i}")
        print(f"ID: {doc_id}")
        print(f"Distance: {distance:.4f}")
        print(f"Metadata: {metadata}")
        print("-" * 40)


if __name__ == "__main__":
    main()