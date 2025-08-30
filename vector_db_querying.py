import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os
import json
import uuid
import re

import requests




# persistent store
def set_up_chromadb():

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
            "url": chunk.get("url"),
            "language": chunk.get("language"),
            "tags": ", ".join(chunk["tags"])
        }

        collection.add(
            documents=[chunk["text"]],
            embeddings=chunk["embedding"],
            metadatas=metadata,
            ids=[chunk.get("id", str(uuid.uuid4()))]
        )

    return collection

#collection = set_up_chromadb()

def get_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "nomic-embed-text:latest", "input": text}
    )
    return response.json()['embeddings']

def query_ollama(query, expanded_query, collection, model="llama3"):

    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5
    )

    retrieved_chunks = [doc for doc in results["documents"][0]]

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
    You are an expert on geo-compliance laws. Use only the CONTEXT below.
    If abbreviations appear in the QUESTION, use the EXPANDED form provided.

    CONTEXT:
    {context}

    QUESTION:
    {query}

    EXPANDED:
    {expanded_query}

    TASK:
    - Determine if the feature artifact has geo-compliance implications.
    - If yes, identify relevant laws and explain why.
    - Always cite exact supporting text from the CONTEXT.
    - Provide confidence 0–10 as a number.
    - If insufficient information, set "implications" to "Insufficient".

    OUTPUT:
    Return ONLY valid JSON (no markdown fences, no extra text) matching exactly this structure:
    {
      "implications": "Required" / "Not required" / "Insufficient",
      "results": [
        {
          "law": : string,
          "reasoning": string,
          "highlight": string,
          "supporting_text": string,
          "confidence": number
        }
      ]
    }
    Ensure the JSON is syntactically valid and complete. Do not include comments or code blocks.
    """

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "temperature": 0.1, "format": "json", "max_tokens": 800},
        stream=True,
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

with open("data_sources/terminology.json", "r", encoding="utf-8") as f:
    glossary = json.load(f)


def expand_abbreviations(text, glossary):
    for abbr, definition in glossary.items():
        pattern = r"\b" + re.escape(abbr) + r"\b"
        replacement = f"{abbr} ({definition})"
        text = re.sub(pattern, replacement, text)
    return text

# Example usage
# query_1 = "User behavior scoring for policy gating: Behavioral scoring via Spanner will be used to gate access to certain tools. The feature tracks usage and adjusts gating based on BB divergence. "
# expanded_query = expand_abbreviations(query_1, glossary)


# answer = query_ollama(query_1, expanded_query, collection, model="llama3")
# print("LLM Answer:\n", answer)
