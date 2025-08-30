import os
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import requests
from transformers import GPT2TokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import BPE
<<<<<<< HEAD
=======
import re

import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt")
>>>>>>> frontend-testing-integration



PDF_FILES = ["data_sources/california-state-law.pdf", "data_sources/eu-regulations.pdf", "data_sources/florida-state-law.pdf"]

HTML_FILES = ["data_sources/us-law.htm"]

SOURCE_METADATA = {
        "data_sources/california-state-law.pdf": {
<<<<<<< HEAD
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
=======
                                                        "title": "SB-976 Protecting Our Kids from Social Media Addiction Act",
                                                        "jurisdiction": "US-CA",
                                                        "law_type": "ChildProtection",
                                                        "doc_type": "Senate Bill",
                                                        "effective_date": "2024-09-20",
                                                        "last_amended": "2024-07-15",
                                                        "publisher": "Senate, State of California",
                                                        "url": "https://leginfo.legislature.ca.gov/faces/billTextClient.xhtml?bill_id=202320240SB976",
                                                        "language": "en",
                                                        "doc_version": "Amended 2024",
                                                        "tags": ["social media", "minors", "addiction prevention", "California"],
                                                        "authenticity": "Official",
                                                        "source_file": "california-state-law.pdf"
>>>>>>> frontend-testing-integration
                                                        },

        "data_sources/eu-regulations.pdf": {
                                            "title": "Digital Services Act (Regulation (EU) 2022/2065)",
                                            "jurisdiction": "EU",
                                            "law_type": "PlatformRegulation",
                                            "doc_type": "Regulation",
<<<<<<< HEAD
                                            "section": "Chapter III, Article 26",
=======
>>>>>>> frontend-testing-integration
                                            "effective_date": "2024-02-17",
                                            "last_amended": None,
                                            "publisher": "European Union",
                                            "url": "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX%3A32022R2065",
                                            "language": "en",
                                            "doc_version": "Original 2022",
<<<<<<< HEAD
                                            "tags": ["intermediary liability", "content moderation", "platform duties"],
=======
                                            "tags": ["intermediary liability", "content moderation", "platform duties", "Europe"],
>>>>>>> frontend-testing-integration
                                            "authenticity": "Official",
                                            "source_file": "eu-regulations.pdf"
                                            },

        "data_sources/florida-state-law.pdf": {
<<<<<<< HEAD
                                                "title": "Florida Online Protections for Minors Act (HB 3, 2024)",
                                                "jurisdiction": "US-FL",
                                                "law_type": "ChildProtection",
                                                "doc_type": "Act",
                                                "section": "Sec. 4(a)(1)",
                                                "effective_date": "2024-07-01",
                                                "last_amended": None,
                                                "publisher": "Florida Legislature",
=======
                                                "title": "CS/CS/HB 3: Online Protections for Minors",
                                                "jurisdiction": "US-FL",
                                                "law_type": "ChildProtection",
                                                "doc_type": "Act",
                                                "effective_date": "2025-01-01",
                                                "last_amended": "2024-03-25",
                                                "publisher": "Florida House of Representatives",
>>>>>>> frontend-testing-integration
                                                "url": "https://www.flsenate.gov/Session/Bill/2024/3",
                                                "language": "en",
                                                "doc_version": "Original 2024",
                                                "tags": ["social media", "minors", "age verification", "online safety"],
                                                "authenticity": "Official",
                                                "source_file": "florida-state-law.pdf"
                                                },

        "data_sources/us-law.htm": {
<<<<<<< HEAD
                        "title": "18 U.S. Code § 2258A - Reporting Requirements of Electronic Communication Service Providers and Remote Computing Service Providers",
=======
                        "title": "18 U.S. Code § 2258A - Reporting requirements of providers",
>>>>>>> frontend-testing-integration
                        "jurisdiction": "US-Federal",
                        "law_type": "ReportingRequirement",
                        "doc_type": "Statute",
                        "section": "§2258A(b)(1)",
<<<<<<< HEAD
                        "effective_date": "2008-07-27",
=======
                        "effective_date": "2024-05-07",
>>>>>>> frontend-testing-integration
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
<<<<<<< HEAD
=======
def clean_text(text):

    text = text.replace("“", '"').replace("”", '"').replace("’", "'")

    #EU Doc
    text = re.sub(r"Official\s+Journal\s+of\s+the\s+European\s+Union", "", text)
    text = re.sub(r"27.10.2022", "", text)
    text = re.sub(r"L 277/.\d+", "", text)
    #California
    text = re.sub(r"Ch. 321", "", text)
    text = re.sub(r"—\s+\d+\s+— ", "", text)
    #Florida
    text = re.sub(r"F\s+L\s+O\s+R\s+I\s+D\s+A\s+H\s+O\s+U\s+S\s+E\s+O\s+F\s+R\s+E\s+P\s+R\s+E\s+S\s+E\s+N\s+T\s+A\s+T\s+I\s+V\s+E\s+S", "", text)
    text = re.sub(r"Page \d+ of \d+", "", text)
    text = re.sub(r"Page \d+ of \d+", "", text)
    text = re.sub(r"CS/CS/HB\s+3,\s+Engrossed\s+1", "", text)
    #US Federal
    text = re.sub(r"CODING:\s+Words\s+stricken\s+are\s+deletions;\s+words\s+underlined\s+are\s+additions.", "", text)
    text = re.sub(r"2024 Legislature", "", text)
    text = re.sub(r"hb0003-04-er", "", text)

    text = re.sub(r"\[\[Page\s+132\s+STAT\.\s+\d+\]\]", "", text)
    text = re.sub(r"From\s+the\s+U\.S\.\s+Government\s+Publishing\s+Office", "", text)


    return text.strip()
>>>>>>> frontend-testing-integration

# -------------------------------
# TOKENIZER for chunking
# -------------------------------
<<<<<<< HEAD
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

=======

def chunk_text(text, max_tokens=600, overlap=120):
    sentences = sent_tokenize(text)
    chunks, current_chunk = [], []
    current_len = 0

    for sent in sentences:
        words = sent.split()
        sent_len = len(words)

        if current_len + sent_len > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:]  # add overlap
            current_len = len(current_chunk)

        current_chunk.extend(words)
        current_len += sent_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


>>>>>>> frontend-testing-integration
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
<<<<<<< HEAD
            "section": meta["section"],
=======
>>>>>>> frontend-testing-integration
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
<<<<<<< HEAD
            "section": meta["section"],
=======
>>>>>>> frontend-testing-integration
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