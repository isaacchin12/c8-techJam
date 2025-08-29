def get_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embed",
        json={"model": "nomic-embed-text:latest", "input": text}
    )
    return response.json()['embeddings']