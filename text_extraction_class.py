import os
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import requests
from transformers import GPT2TokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE

class TextExtraction(self):
    def __init__(self):
        pass

    def chunk_text(text, max_tokens=500, overlap=100):
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = min(start + max_tokens, len(words))
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            start += max_tokens - overlap
        return chunks

    def extract_text_from_pdf(file_path: str) -> str:
        """Extracts text from a PDF file."""
        text = ""
        reader = PdfReader(file_path)
        for page in reader.pages:
            text += page.extract_text() + "\n\n"
        return text

    def extract_text_from_html(file_path: str) -> str:
        """Extracts text from an HTML file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            # Remove script and style elements
            for script_or_style in soup(['script', 'style']):
                script_or_style.decompose()
            text = soup.get_text(separator='\n')
            # Break into lines and remove leading/trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # Drop blank lines
            text = '\n'.join(line for line in lines if line)
        return text


    # -------------------------------
    # Combined Extraction
    # -------------------------------
    def extract_texts(file_paths: list) -> dict:
        """Extracts text from a list of file paths (PDF or HTML)."""
        texts = {}
        for file_path in file_paths:
            if file_path.lower().endswith('.pdf'):
                texts[file_path] = extract_text_from_pdf(file_path)
            elif file_path.lower().endswith('.htm') or file_path.lower().endswith('.html'):
                texts[file_path] = extract_text_from_html(file_path)
            else:
                print(f"Unsupported file format: {file_path}")
        return texts

    # -------------------------------
    # FUNCTION: get embeddings from Ollama
    # -------------------------------

    def get_embedding(text):
        response = requests.post(
            "http://localhost:11434/api/embed",
            json={"model": "nomic-embed-text:latest", "input": text}
        )
        return response.json()['embeddings']

