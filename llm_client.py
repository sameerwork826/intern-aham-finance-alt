import os, json, requests
from typing import List, Dict, Optional
from utils import get_env

class LLMClient:
    def __init__(self):
        self.provider = get_env("LLM_PROVIDER", "ollama")
        if self.provider == "groq":
            self.api_key = get_env("GROQ_API_KEY", required=True)
            self.model = get_env("GROQ_MODEL", "llama3-8b-instant")
        elif self.provider == "ollama":
            self.base_url = get_env("OLLAMA_BASE_URL", "http://localhost:11434")
            self.model = get_env("OLLAMA_MODEL", "gemma:2b")
        else:
            raise ValueError("LLM_PROVIDER must be 'groq' or 'ollama'")

    def chat(self, messages: List[Dict], temperature: float = 0.2, max_tokens: int = 512) -> str:
        if self.provider == "groq":
            # Lazy import to avoid dependency if unused
            from groq import Groq
            client = Groq(api_key=self.api_key)
            resp = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        else:
            # Ollama chat API
            url = f"{self.base_url}/api/chat"
            payload = {
                "model": self.model,
                "messages": messages,
                "options": {"temperature": temperature, "num_predict": max_tokens},
                "stream": False
            }
            r = requests.post(url, json=payload, timeout=120)
            r.raise_for_status()
            data = r.json()
            # Ollama returns an array of messages; last message is assistant
            if "message" in data and "content" in data["message"]:
                return data["message"]["content"].strip()
            # Some versions return "messages"
            if "messages" in data and len(data["messages"])>0:
                return data["messages"][-1].get("content","").strip()
            # Fallback
            return data.get("response","").strip()

def sysmsg(content: str) -> Dict:
    return {"role": "system", "content": content}

def usermsg(content: str) -> Dict:
    return {"role": "user", "content": content}

def asstmsg(content: str) -> Dict:
    return {"role": "assistant", "content": content}
