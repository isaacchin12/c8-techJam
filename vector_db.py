import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os
import json
import uuid


# persistent store
client = chromadb.PersistentClient(path="./chroma_store")  # <-- make this a volume in Docker

collection = client.get_or_create_collection("geo_compliance")

with open("rag_chunks.json", "r", encoding="utf-8") as f:
    all_chunks = json.load(f)

for chunk in all_chunks:
    metadata = {
        "source": chunk["source"],
        "title": chunk["title"],
        "publisher": chunk.get("publisher"),
        "jurisdiction": chunk.get("jurisdiction"),
        "law_type": chunk.get("law_type"),
        "effective_date": chunk.get("effective_date"),
        "section": chunk.get("section"),
        "url": chunk.get("url"),
        "language": chunk.get("language"),
        "tags": chunk.get("tags")
    }

    collection.add(
        documents=[chunk["text"]],
        embeddings=[chunk["embedding"]],
        metadatas=[metadata],
        ids=[chunk.get("id", str(uuid.uuid4()))]
    )