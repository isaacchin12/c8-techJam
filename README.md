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

  <img width="1920" height="1080" alt="Rag-pipeline" src="https://github.com/user-attachments/assets/a3e27377-97df-49b0-9ad6-c3de07d720fc" />

## Architecture Overview

  - Baseline RAG Processing
    The retrieved documents are fed to the Ollama LLM to generate compliance assessment and recommendations.
  
  - Hybrid Search
    Searches regulatory databases using both vector embeddings, keyword matching and feedback scores to retrieve relevant context.
    
  - Reranking
    Cross encoder is used to rank the final scores for the top k*2 chunks, to achieve a more clear ranking.
  
  - Human Feedback Integration
    Users can provide feedback on the AI-generated assessments, which is stored and used to enhance better selection of context chunks.


## Getting Started

### Prerequisites

- Python >= 3.10  
- Ollama models downloaded locally via app
- Required Python libraries: `langchain`, `nltk`, `openai`, `pandas`, etc.

### Installation

```bash
git clone https://github.com/{yourusername}/c8-techJam.git
cd c8-techJam
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



