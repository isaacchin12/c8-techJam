import os
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import requests
from transformers import GPT2TokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE



PDF_FILES = ["data_sources/california-state-law.pdf", "data_sources/eu-regulations.pdf", "data_sources/florida-state-law.pdf"]

HTML_FILES = ["data_sources/us-law.htm"]

SOURCE_METADATA = {
        "data_sources/california-state-law.pdf": {
                                                        "title": "Protecting Our Kids from Social Media Addiction Act (California AB 2408)",
                                                        "jurisdiction": "US-CA",
                                                        "law_type": "ChildProtection",
                                                        "doc_type": "Act",
                                                        "section": "Sec. 2(b)(1)",
                                                        "effective_date": "2024-01-01",
                                                        "last_amended": "2024-07-15",
                                                        "publisher": "California Legislature",
                                                        "url": "https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240AB2408",
                                                        "language": "en",
                                                        "doc_version": "Amended 2024",
                                                        "tags": ["social media", "minors", "addiction prevention"],
                                                        "authenticity": "Official",
                                                        "source_file": "california-state-law.pdf.pdf"
                                                        },

        "data_sources/eu-regulations.pdf": {
                                            "title": "Digital Services Act (Regulation (EU) 2022/2065)",
                                            "jurisdiction": "EU",
                                            "law_type": "PlatformRegulation",
                                            "doc_type": "Regulation",
                                            "section": "Chapter III, Article 26",
                                            "effective_date": "2024-02-17",
                                            "last_amended": None,
                                            "publisher": "European Union",
                                            "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32022R2065",
                                            "language": "en",
                                            "doc_version": "Original 2022",
                                            "tags": ["intermediary liability", "content moderation", "platform duties"],
                                            "authenticity": "Official",
                                            "source_file": "eu-regulations.pdf"
                                            },

        "data_sources/florida-state-law.pdf": {
                                                "title": "Florida Online Protections for Minors Act (HB 3, 2024)",
                                                "jurisdiction": "US-FL",
                                                "law_type": "ChildProtection",
                                                "doc_type": "Act",
                                                "section": "Sec. 4(a)(1)",
                                                "effective_date": "2024-07-01",
                                                "last_amended": None,
                                                "publisher": "Florida Legislature",
                                                "url": "https://www.flsenate.gov/Session/Bill/2024/3",
                                                "language": "en",
                                                "doc_version": "Original 2024",
                                                "tags": ["social media", "minors", "age verification", "online safety"],
                                                "authenticity": "Official",
                                                "source_file": "florida-state-law.pdf"
                                                },

        "data_sources/us-law.htm": {
                        "title": "18 U.S. Code § 2258A - Reporting Requirements of Electronic Communication Service Providers and Remote Computing Service Providers",
                        "jurisdiction": "US-Federal",
                        "law_type": "ReportingRequirement",
                        "doc_type": "Statute",
                        "section": "§2258A(b)(1)",
                        "effective_date": "2008-07-27",
                        "last_amended": "2022-12-23",
                        "publisher": "US Congress",
                        "url": "https://www.law.cornell.edu/uscode/text/18/2258A",
                        "language": "en",
                        "doc_version": "Amended 2022",
                        "tags": ["child protection", "mandatory reporting", "NCMEC", "CSAM"],
                        "authenticity": "Official",
                        "source_file": "us-law.htm",
                        }

}

# -------------------------------
# TOKENIZER for chunking
# -------------------------------
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

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

# -------------------------------
# PDF Text Extraction
# -------------------------------
def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file."""
    text = ""
    reader = PdfReader(file_path)
    for page in reader.pages:
        text += page.extract_text() + "\n\n"
    return text


# -------------------------------
# HTML Text Extraction
# -------------------------------
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


#print("Testing embedding" + str(get_embedding("Hello world!")))

# -------------------------------
# MAIN: ingest & chunk
# -------------------------------
all_chunks = []

# PDFs
for pdf_file in PDF_FILES:
    text = extract_text_from_pdf(pdf_file)
    chunks = chunk_text(text)
    meta = SOURCE_METADATA[pdf_file]
    for i, chunk in enumerate(chunks):
        all_chunks.append({
            "text": chunk,
            "source": pdf_file,
            "title": meta["title"],
            "publisher": meta["publisher"],
            "section": meta["section"],
            "jurisdiction": meta["jurisdiction"],
            "law_type": meta["law_type"],
            "effective_date": meta["effective_date"],
            "url": meta["url"],
            "language": meta["language"],
            "tags": meta["tags"],
            "embedding": get_embedding(chunk)
        })

# HTML
for html_file in HTML_FILES:
    text = extract_text_from_html(html_file)
    chunks = chunk_text(text)
    meta = SOURCE_METADATA[html_file]
    for i, chunk in enumerate(chunks):
        all_chunks.append({
            "text": chunk,
            "source": pdf_file,
            "title": meta["title"],
            "publisher": meta["publisher"],
            "section": meta["section"],
            "jurisdiction": meta["jurisdiction"],
            "law_type": meta["law_type"],
            "effective_date": meta["effective_date"],
            "url": meta["url"],
            "language": meta["language"],
            "tags": meta["tags"],
            "embedding": get_embedding(chunk)
        })

# -------------------------------
# SAVE or INSERT into Vector DB
# -------------------------------
import json
with open("rag_chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, indent=2, ensure_ascii=False)

print(f"Completed ingestion: {len(all_chunks)} chunks ready for RAG pipeline!")