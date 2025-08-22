import os, re
from typing import List, Dict
from pypdf import PdfReader
from db import get_engine
from rag import RAGStore

CHUNK_SIZE = 600
CHUNK_OVERLAP = 120

def read_file_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        reader = PdfReader(path)
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

def chunk_text(text: str, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += (size - overlap)
    return chunks

def ingest_folder(applicant_id: str, folder: str):
    eng = get_engine()
    chunks_for_index = []
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if not os.path.isfile(path): continue
        if not any(name.lower().endswith(ext) for ext in [".pdf", ".txt"]): continue
        text = read_file_text(path)
        # Save full doc to DB
        with eng.begin() as conn:
            conn.exec_driver_sql(
                "INSERT INTO documents(applicant_id, doc_name, doc_path, doc_text) VALUES (:a,:n,:p,:t)",
                {"a": applicant_id, "n": name, "p": path, "t": text[:200000]}
            )
        # Create chunks for index
        for ch in chunk_text(text):
            chunks_for_index.append({"doc_name": name, "text": ch})
    # Build RAG index
    RAGStore().build(applicant_id, chunks_for_index)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--applicant_id", required=True)
    parser.add_argument("--folder", required=True)
    args = parser.parse_args()
    ingest_folder(args.applicant_id, args.folder)
    print("Ingestion complete.")
