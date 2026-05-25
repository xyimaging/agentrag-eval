"""
Build and load the ChromaDB vector index for RAG retrieval.

Passages are encoded with all-MiniLM-L6-v2 (runs locally, no API cost)
and stored with cosine similarity. The index persists to disk so it only
needs to be built once per dataset.
"""

import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from config import DATA_DIR, INDEX_DIR, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL


def build_index(data_path: str, collection_name: str = CHROMA_COLLECTION_NAME):
    """Embed all passages from a dataset file and store them in ChromaDB."""
    print(f"Loading dataset: {data_path}")
    with open(data_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    os.makedirs(INDEX_DIR, exist_ok=True)
    chroma_path = os.path.join(INDEX_DIR, "chroma_db")
    client = chromadb.PersistentClient(path=chroma_path)

    # Delete existing collection to guarantee a clean rebuild
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)
        print(f"  Deleted existing collection: {collection_name}")

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Deduplicate passages across questions (one entry per unique title)
    seen_titles: set[str] = set()
    passages, ids, metadatas = [], [], []
    for item in dataset:
        for p in item["passages"]:
            key = p["title"]
            if key not in seen_titles:
                seen_titles.add(key)
                passages.append(p["text"])
                ids.append(p["title"][:63])  # ChromaDB ID length limit
                metadatas.append({"title": p["title"], "source": "hotpotqa"})

    print(f"  Collected {len(passages)} unique passages. Embedding...")

    batch_size = 64
    for i in tqdm(range(0, len(passages), batch_size), desc="Embedding"):
        batch_texts      = passages[i:i+batch_size]
        batch_ids        = ids[i:i+batch_size]
        batch_meta       = metadatas[i:i+batch_size]
        batch_embeddings = embedder.encode(batch_texts, show_progress_bar=False).tolist()
        collection.add(
            documents=batch_texts,
            embeddings=batch_embeddings,
            ids=batch_ids,
            metadatas=batch_meta,
        )

    print(f"  Index built: {collection.count()} passages saved to {chroma_path}")
    return collection


def load_index(collection_name: str = CHROMA_COLLECTION_NAME):
    """Load an existing ChromaDB collection from disk."""
    chroma_path = os.path.join(INDEX_DIR, "chroma_db")
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection(collection_name)
    return collection, SentenceTransformer(EMBEDDING_MODEL)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str,
                        default=os.path.join(DATA_DIR, "hotpotqa_100.json"))
    args = parser.parse_args()
    build_index(args.data)
