# C8 - CompliAI: Geo-compliance Detection System

CompliAI is an advanced system designed to help businesses ensure that their operations, documentation, and code comply with regional regulations. By leveraging state-of-the-art AI technologies, CompliAI can detect potential geo-compliance violations, provide actionable insights, and continuously improve its performance through human feedback.

## Key Features

- RAG (Retrieval-Augmented Generation)
  Feeding relevant regulatory documents to an LLM to provide precise, context-aware responses.

- Ollama Integration
  Uses Ollama model for embedding and LLM tasks, and added confidence scores from LLMs.

- Hybrid Search
  Employs a combination of vector-based and keyword search to retrieve the most relevant regulatory and policy documents.

- Human Feedback Loop
  Allows users to rate AI assessments and provide comments, improving model accuracy over time.

## Architecture Overview

  - Query Handling
    User submits queries related to compliance checks (e.g., code, documentation, operational plans).
  
  - Hybrid Search
    Searches regulatory databases using both vector embeddings and keyword matching to retrieve relevant context.
  
  - RAG Processing
    The retrieved documents are fed to the Ollama LLM to generate compliance assessment and recommendations.
  
  - Human Feedback Integration
    Users can provide feedback on the AI-generated assessments, which is stored and used to fine-tune future predictions.


## Getting Started

### Prerequisites

- Python >= 3.10  
- Ollama models downloaded locally via app
- Required Python libraries: `langchain`, `nltk`, `openai`, `pandas`, etc.

### Installation

```bash
git clone https://github.com/{yourusername}/compliai.git
cd compliai
pip install -r requirements.txt
```
This step includes adding the ollama LLMs locally, via the Command Prompt. Download the models if you have yet to.
```cmd
ollama pull mxbai-embed-large
ollam run gemma3
```
Run text_extraction.py first, followed by vector_db_querying.py

Run the dashboard
```cmd
streamlit run app.py
```

## Libraries used
`transformers` `ollama` `chromadb` `nltk` `bs4` `PyPDF2`



