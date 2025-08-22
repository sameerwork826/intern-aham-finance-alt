import json
from typing import List, Dict
from llm_client import LLMClient, sysmsg, usermsg
from rag import RAGStore
import numpy as np

SYS_BASE = (
    "You are a meticulous Loan Application Assistant helping loan officers. "
    "When answering, be concise, cite evidence from provided context snippets "
    "by quoting short phrases, and explicitly list any missing information."
)

def summarize_applicant(applicant_id: str) -> str:
    rag = RAGStore()
    # Retrieve a general context by asking for 'overall applicant summary' as proxy
    ctx = rag.search(applicant_id, "overall financial profile and risks", top_k=6)
    context = "\n\n".join([f"[{c['meta']['doc_name']}] {c['text']}" for c in ctx]) if ctx else "(no docs found)"
    client = LLMClient()
    prompt = (
        "Summarize this applicant's documents focusing on income stability, liabilities, employment, anomalies, "
        "and KYC consistency. Provide a 5-8 bullet executive brief."
        f"\n\nContext:\n{context}"
    )
    return client.chat([sysmsg(SYS_BASE), usermsg(prompt)], temperature=0.2, max_tokens=450)

def answer_query(applicant_id: str, question: str) -> str:
    rag = RAGStore()
    ctx = rag.search(applicant_id, question, top_k=6)
    context = "\n\n".join([f"[{c['meta']['doc_name']}] {c['text']}" for c in ctx]) if ctx else "(no docs found)"
    client = LLMClient()
    prompt = (
        f"Question: {question}\n\n"
        "Answer using the context. If uncertain, say what additional docs/data are needed."
        f"\n\nContext:\n{context}"
    )
    return client.chat([sysmsg(SYS_BASE), usermsg(prompt)], temperature=0.2, max_tokens=450)

def recommend(app_features: Dict, risk_score: float) -> str:
    # Simple policy + LLM rationale
    if risk_score >= 0.75:
        action = "REJECT"
    elif risk_score >= 0.45:
        action = "NEED_MORE_DOCS"
    else:
        action = "APPROVE"

    client = LLMClient()
    prompt = (
        "Given the applicant features and a risk score (higher means more risk), "
        "output a JSON with fields: action, rationale. "
        "Action must be one of APPROVE, REJECT, NEED_MORE_DOCS. "
        f"Applicant features: {json.dumps(app_features, ensure_ascii=False)}\n"
        f"Risk score: {risk_score:.3f}"
    )
    resp = client.chat([sysmsg("You write STRICT JSON only, no explanations."), usermsg(prompt)], temperature=0.1, max_tokens=200)
    # Guardrail for non-JSON outputs
    try:
        data = json.loads(resp)
        if not isinstance(data, dict) or "action" not in data:
            raise ValueError
        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return json.dumps({"action": action, "rationale": "Rule-based fallback rationale using risk thresholding."}, ensure_ascii=False)
