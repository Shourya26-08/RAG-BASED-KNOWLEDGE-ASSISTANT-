# RAG-Based Knowledge Assistant

A production-style Retrieval-Augmented Generation (RAG) application for asking grounded questions over your own documents.

## Features
- Upload PDF, DOCX, TXT, or Markdown knowledge files
- Document extraction, cleaning, chunking, and overlap control
- Semantic embeddings with Sentence Transformers
- FAISS vector similarity search
- Configurable Top-K retrieval
- Grounded LLM answer generation through OpenAI API
- Source names and similarity scores for retrieval transparency
- Streamlit chat interface

## Architecture
Documents -> ingestion -> cleaning -> chunking -> embeddings -> FAISS
Question -> query embedding -> Top-K retrieval -> prompt with context -> LLM -> grounded answer

## Tech Stack
- Python 3.10+
- Streamlit
- Sentence Transformers
- FAISS
- OpenAI API
- PyPDF
- python-docx
- NumPy

## Setup
1. Create a virtual environment.
2. Run: pip install -r requirements.txt
3. Copy .env.example to .env.
4. Add OPENAI_API_KEY to .env.
5. Run: streamlit run app.py

## Environment
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TOP_K=4
CHUNK_SIZE=800
CHUNK_OVERLAP=120

## RAG Design
Chunking is configurable so retrieval units can be tuned for precision versus context. Sentence Transformer embeddings represent chunks and questions as dense vectors. FAISS IndexFlatIP searches normalized vectors using semantic similarity. The LLM prompt instructs the model to answer only from retrieved evidence and acknowledge when the knowledge base is insufficient.

## Resume-ready description
RAG-Based Knowledge Assistant — Built a Retrieval-Augmented Generation pipeline for domain-specific question answering using document ingestion, text chunking, semantic embeddings, FAISS vector search, and LLM response generation. Implemented grounded retrieval with configurable Top-K search and prompt controls, experimented with chunking and embedding parameters, and integrated an LLM API into an end-to-end Streamlit AI application.

## Security
API keys are loaded from environment variables and secrets are excluded through .gitignore. Uploaded documents are processed locally before retrieved context is sent to the configured LLM API.

## Author
Shourya Bhatnagar
GitHub: https://github.com/Shourya26-08