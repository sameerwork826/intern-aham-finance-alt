# Loan Application Assistant (Groq or Ollama)

An end-to-end prototype that combines:
- **ML risk model** (Logistic Regression or XGBoost) for approval prediction.
- **LLM assistant** (Groq or local Ollama) to summarize applicant documents, answer officer questions, and provide recommendations.
- **RAG** over applicant documents powered by FAISS + sentence-transformers.
- **Streamlit UI** for interactive demo.

## Features
1. Upload or use sample loan application data (structured).
2. Train a risk model (logreg default; XGBoost if installed).
3. Ingest per-applicant documents (PDF/TXT) → chunk → embed → FAISS index.
4. Ask questions about an applicant, get a synthesis grounded in their docs.
5. Summarize documents for a quick case brief.
6. Get an LLM-backed recommendation (approve/reject/need more docs) with justification that also considers the ML risk score.

## Quickstart

### 0) Python & Virtualenv
```bash
python -m venv .venv && source .venv/bin/activate  # on mac/linux
# or on Windows
# py -m venv .venv && .venv\Scripts\activate
```

### 1) Install requirements
```bash
pip install -r requirements.txt
```

> If installation fails for `faiss-cpu` on Windows, try:
> ```bash
> pip install faiss-cpu==1.7.4
> ```
> or comment it out and use a simple cosine search fallback (already included).

### 2) Choose your LLM provider
Copy `.env.example` to `.env` and choose **one**:

- **Groq** (cloud): set `LLM_PROVIDER=groq` and add `GROQ_API_KEY=...`
- **Ollama** (local): set `LLM_PROVIDER=ollama` and ensure `ollama serve` is running.
  - Pull a model, e.g. `ollama pull gemma:2b` or `ollama pull llama3:8b`

### 3) Run the app
```bash
streamlit run app.py
```

The app ships with a **toy dataset** and a folder of **sample docs** to try instantly.

## Project Layout
```
loan-llm-assistant/
├── app.py
├── chains.py
├── db.py
├── ingest_docs.py
├── llm_client.py
├── model_train.py
├── rag.py
├── utils.py
├── requirements.txt
├── .env.example
├── data/
│   ├── applications_sample.csv
│   └── sample_docs/
│       ├── 1001/
│       │   ├── bank_statement.txt
│       │   └── employment_letter.txt
│       └── 1002/
│           └── bank_statement.txt
└── storage/  (created at runtime: models/, indexes/, sqlite db)
```

## Notes
- This is a minimal demo—**do not** use in production as-is.
- For PDF parsing we rely on `pypdf`. For DOCX you could add `python-docx` similarly.
- RAG embeddings use `sentence-transformers` (MiniLM) to stay light and local.
- If FAISS is unavailable, we fallback to cosine similarity in-memory.

## Example Prompts (in UI)
- "Summarize this applicant's documents focusing on income stability, obligations, and anomalies."
- "List any missing documents needed for underwriting."
- "Does the declared salary reconcile with the bank statement inflows? Cite details."

Good luck and have fun! 🚀
