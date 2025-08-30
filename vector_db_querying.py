import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os
import json
import uuid
import re
import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from datetime import datetime

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


prompt_file_path = "prompts/geo_compliance_prompt.txt"


# Setting up abbreviations
def expand_abbreviations(text, glossary):

    for abbr, definition in glossary.items():
        pattern = r"\b" + re.escape(abbr) + r"\b"
        replacement = f"{abbr} ({definition})"
        text = re.sub(pattern, replacement, text)
    return text


# Persistent Store
def set_up_chromadb():

    client = chromadb.PersistentClient(path="./chroma_store")

    collection = client.get_or_create_collection("geo_compliance_v2")

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
        documents = collection.get()["documents"]

    return collection, documents


# ========== Embeddings ==========
def get_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "mxbai-embed-large", "input": text}
    )
    return response.json()['embeddings']

# ================================

# ========== Hybrid Search ==========
def hybrid_search(query, collection, all_documents, feedback_collection=None, top_k=5, alpha=0.7, beta = 0.2, gamma =0.1):

        # 1. Semantic search via Chroma
    query_embedding = get_embedding(query)
    vector_results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k * 2   # grab more for reranking
    )

    vector_docs = vector_results["documents"][0]
    vector_scores = vector_results["distances"][0]  # smaller = closer
    vector_scores = [1.0 - (s / max(vector_scores)) for s in vector_scores]  # normalize

    # 2. Keyword search via BM25
    tokenized_docs = [doc.split() for doc in all_documents]
    bm25 = BM25Okapi(tokenized_docs)
    keyword_scores = bm25.get_scores(query.split())

    # pick same candidates as vector (union)
    candidate_set = set(vector_docs)
    candidates = list(candidate_set)

    # 3. Fuse scores
    fused_results = []
    for doc in candidates:
        v_score = vector_scores[vector_docs.index(doc)] if doc in vector_docs else 0
        k_score = keyword_scores[all_documents.index(doc)] if doc in all_documents else 0
        #if feedback_collection and doc in feedback_collection:
        #    f_score = feedback_collection[doc]
        #else:
        f_score = 1  # set to 1 just for an example, to represent 1 good user feedback.

        final_score = alpha * v_score + beta * k_score + gamma * f_score
        fused_results.append((doc, final_score))

    # 4. Sort & return top_k
    fused_results = sorted(fused_results, key=lambda x: x[1], reverse=True)[:top_k]
    return [doc for doc, _ in fused_results]

# ================================

# ========== Reranked ==========
def rerank_results(query, retrieved_chunks):
    pairs = [(query, chunk) for chunk in retrieved_chunks]
    scores = reranker.predict(pairs)
    reranked = [chunk for _, chunk in sorted(zip(scores, retrieved_chunks), reverse=True)]
    return reranked
# ================================



# ========== Querying Function ==========
def query_ollama(query, expanded_query, model, documents, prompt_file_path):

    retrieved_chunks = hybrid_search(query, collection, documents, top_k=5, alpha=0.7)
    reranked_chunks = rerank_results(query, retrieved_chunks)
    context = "\n\n".join(reranked_chunks[:3])

    with open(prompt_file_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    prompt = prompt_template.format(
        query=query,
        expanded_query=expanded_query,
        context=context
    )

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
                "model": model, 
                "prompt": prompt, 
                "temperature": 0.1, 
                "format": "json", 
                "max_tokens": 700
            },
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
# ===================================




# ========== Save Feedback ==========

def save_feedback_to_chroma(feedback, client_path="./chroma_store"):

        client = chromadb.PersistentClient(path=client_path)
        feedback_col = client.get_or_create_collection("geo_feedback")

        feedback_col.add(
            documents=[feedback["answer"]],
            embeddings=get_embedding(feedback["query"]),
            ids=[str(uuid.uuid4())]
        )

# ===================================

# ========== Query + Feedback ==========
def query_with_feedback(query, expanded_query, model, documents, prompt_file_path):
    """
    Run query through RAG pipeline, collect human feedback, and store in ChromaDB.
    """
    # 1. Get model answer
    answer = query_ollama(query, expanded_query, model, documents, prompt_file_path)
    print("\n--- Response ---")
    print(answer)

    while True:
        feedback_rating = input("Thumbs up/down? (u/d): ").strip().lower()
        if feedback_rating in ["u", "d"]:
            break
        print("Invalid input. Type 'u' for thumbs up, 'd' for thumbs down.")

    feedback_text = input("Optional comments: ").strip()

    feedback_entry = {
        "query": query,
        "expanded_query": expanded_query,
        "answer": answer,
        "rating": 1 if feedback_rating == "u" else -1,
        "comments": feedback_text,
        "timestamp": datetime.utcnow().isoformat()
    }

    save_feedback_to_chroma(feedback_entry)
    print("Feedback saved successfully.\n")

    return answer, feedback_entry

# ========== Example Usage ==========
if __name__ == "__main__":
    query_1 = "User behavior scoring for policy gating: Behavioral scoring via Spanner will be used to gate access to certain tools. The feature tracks usage and adjusts gating based on BB divergence."
    with open("data_sources/terminology.json", "r", encoding="utf-8") as f:
        glossary = json.load(f)
    expanded_query = expand_abbreviations(query_1, glossary)


    answer = query_with_feedback(query_1, expanded_query, model="gemma3", documents=all_documents, prompt_file_path="prompts/geo_compliance_prompt.txt")
