import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os
import json
import uuid
import re

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


def get_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "nomic-embed-text:latest", "input": text}
    )
    return response.json()['embeddings']

def query_ollama(query, model="llama3"):

    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5
    )

    retrieved_chunks = [doc for doc in results["documents"][0]]

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
    You are an expert on geo-compliance laws, and you are very familiar with the following extracted parts of laws:
    1. EU Digital Services Act (DSA)
    2. California Consumer Privacy Act (CCPA)
    3. Florida’s Online Protections for Minors
    4. US CyberTipline Modernization Act of 2018

    User input: Feature artifacts for certain tech products. This can be the title, description, or any other relevant text that describes the feature.
    Examples:
    - "Feature reads user location to enforce France's copyright rules (download blocking)"
    - "Requires age gates specific to Indonesia's Child Protection Law"
    - "Geofences feature rollout in US for market testing" (Business-driven ≠ legal requirement)
    - "A video filter feature is available globally except KR" (didn't specify the intention, need human evaluation)

    Output or goal: 
    Determine if the feature artifact has geo-compliance implications. If yes, identify the relevant laws and provide a brief explanation of why it is relevant. If no, simply state "No geo-compliance implications".
    Provide clear reasoning for your conclusions and also pointing out the source and the exact text.
    Provide a confidence score from 1-10 for each identified law, where 10 means absolutely certain and 1 means very uncertain.
    If you are unsure, say "Insufficient information to determine geo-compliance implications".

    Answer the following question based only on the context below:

    Context:
    {context}

    Question: {query}

    Answer:
    """

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
query_1 = "Feature reads user location to enforce France's copyright rules (download blocking)"
answer = query_ollama(query_1, model="llama3")
print("LLM Answer:\n", answer)


with open("data_sources/terminology.json", "r", encoding="utf-8") as f:
    glossary = json.load(f)


def expand_abbreviations(text, glossary):
    for abbr, definition in glossary.items():
        pattern = r"\b" + re.escape(abbr) + r"\b"
        replacement = f"{abbr} ({definition})"
        text = re.sub(pattern, replacement, text)
    return text