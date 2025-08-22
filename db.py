from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
import os

DB_PATH = os.path.join("storage", "db", "app.db")

def get_engine() -> Engine:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return create_engine(f"sqlite:///{DB_PATH}", future=True)

def init_schema():
    eng = get_engine()
    with eng.begin() as conn:
        conn.exec_driver_sql("""
        CREATE TABLE IF NOT EXISTS applicants(
            applicant_id TEXT PRIMARY KEY,
            name TEXT,
            age INTEGER,
            income REAL,
            employment_status TEXT,
            credit_score REAL,
            loan_amount REAL,
            loan_purpose TEXT,
            existing_debt REAL,
            features_json TEXT
        );
        """)
        conn.exec_driver_sql("""
        CREATE TABLE IF NOT EXISTS documents(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            applicant_id TEXT,
            doc_name TEXT,
            doc_path TEXT,
            doc_text TEXT
        );
        """)
        conn.exec_driver_sql("""
        CREATE TABLE IF NOT EXISTS notes(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            applicant_id TEXT,
            note TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        conn.exec_driver_sql("""
        CREATE TABLE IF NOT EXISTS recommendations(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            applicant_id TEXT,
            risk_score REAL,
            action TEXT,
            rationale TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
