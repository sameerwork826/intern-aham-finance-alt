import os
from dotenv import load_dotenv

load_dotenv()

def get_env(name, default=None, required=False):
    val = os.getenv(name, default)
    if required and (val is None or val == ""):
        raise RuntimeError(f"Missing required env var: {name}")
    return val

def safe_float(x, default=None):
    try: return float(x)
    except Exception: return default

def ensure_dirs():
    os.makedirs("storage/models", exist_ok=True)
    os.makedirs("storage/indexes", exist_ok=True)
    os.makedirs("storage/db", exist_ok=True)
