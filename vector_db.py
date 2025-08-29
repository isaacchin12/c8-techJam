import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os
import json
import uuid

import requests


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
        embeddings=chunk["embedding"],
        #metadatas=metadata,
        ids=[chunk.get("id", str(uuid.uuid4()))]
    )


def get_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "nomic-embed-text:latest", "input": text}
    )
    return response.json()['embeddings']

query = "What are Florida’s restrictions on minors using social media?"
query_embedding = get_embedding(query)

results = collection.query(
    query_embeddings=query_embedding,
    n_results=5
)

retrieved_chunks = [doc for doc in results["documents"][0]]

context = "\n\n".join(retrieved_chunks)

prompt = f"""
You are an expert on geo-compliance laws.
Answer the following question based only on the context below:

Context:
{context}

Question: {query}

Answer:
"""


import requests

def query_ollama(prompt, model="gemma3"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt},
        stream=True
    )
    output = ""
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8"))
            if "response" in data:
                output += data["response"]
            if data.get("done", False):
                break
    return output

# Example usage
answer = query_ollama("Summarize Florida’s Online Protections for Minors law.")
print("LLM Answer:\n", answer)