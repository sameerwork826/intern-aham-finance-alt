import os, json, math
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
try:
    import faiss
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

class RAGStore:
    def __init__(self, base_dir="storage/indexes"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.model = SentenceTransformer(EMB_MODEL_NAME)

    def _index_paths(self, applicant_id: str) -> Tuple[str,str]:
        return (os.path.join(self.base_dir, f"{applicant_id}.faiss"),
                os.path.join(self.base_dir, f"{applicant_id}.json"))

    def build(self, applicant_id: str, chunks: List[Dict]):
        texts = [c["text"] for c in chunks]
        metas = [{"doc_name": c["doc_name"], "chunk_id": i} for i, c in enumerate(chunks)]
        embs = self.model.encode(texts, normalize_embeddings=True)
        faiss_path, json_path = self._index_paths(applicant_id)

        if HAVE_FAISS:
            index = faiss.IndexFlatIP(embs.shape[1])
            index.add(embs.astype(np.float32))
            faiss.write_index(index, faiss_path)
        else:
            # Save embeddings for cosine search
            np.save(faiss_path.replace(".faiss", ".npy"), embs.astype(np.float32))

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({"texts": texts, "metas": metas}, f, ensure_ascii=False)

    def search(self, applicant_id: str, query: str, top_k: int = 5) -> List[Dict]:
        faiss_path, json_path = self._index_paths(applicant_id)
        if not os.path.exists(json_path):
            return []
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        texts, metas = data["texts"], data["metas"]
        q_emb = self.model.encode([query], normalize_embeddings=True)[0].astype(np.float32)

        if HAVE_FAISS and os.path.exists(faiss_path):
            import faiss
            index = faiss.read_index(faiss_path)
            D, I = index.search(q_emb.reshape(1,-1), top_k)
            hits = []
            for score, idx in zip(D[0].tolist(), I[0].tolist()):
                if idx == -1: continue
                hits.append({"text": texts[idx], "meta": metas[idx], "score": float(score)})
            return hits
        else:
            emb_path = faiss_path.replace(".faiss", ".npy")
            if not os.path.exists(emb_path):
                return []
            embs = np.load(emb_path)
            sims = (embs @ q_emb)  # cosine since normalized
            idxs = np.argsort(-sims)[:top_k]
            return [{"text": texts[i], "meta": metas[i], "score": float(sims[i])} for i in idxs]
