import os, json, pickle
import pandas as pd
import streamlit as st
from utils import ensure_dirs
from db import init_schema, get_engine
from model_train import train_model, MODEL_PATH, SCALER_PATH
from ingest_docs import ingest_folder
from chains import summarize_applicant, answer_query, recommend
from rag import RAGStore

st.set_page_config(page_title="Loan Application Assistant", layout="wide")
st.title("🏦 Loan Application Assistant (Groq or Ollama)")

ensure_dirs()
init_schema()

with st.sidebar:
    st.header("Setup")
    st.markdown("- Choose LLM provider in `.env` (Groq or Ollama).")
    st.markdown("- Pull a local model if using Ollama (e.g., `gemma:2b`).")
    st.markdown("- Use the sample data to get started quickly.")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Train Risk Model", "📁 Ingest Docs", "🧠 LLM Assistant", "🔎 RAG Debug"])

with tab1:
    st.subheader("Train / Load Risk Model")
    uploaded = st.file_uploader("Upload applications CSV (optional)", type=["csv"])
    data_path = "data/applications_sample.csv"
    if uploaded:
        data_path = os.path.join("storage", "uploaded.csv")
        os.makedirs("storage", exist_ok=True)
        with open(data_path, "wb") as f: f.write(uploaded.read())
        st.success(f"Loaded custom dataset: {data_path}")
    if st.button("Train Model"):
        auc = train_model(data_path)
        st.success(f"Model trained. AUC ≈ {auc:.3f}")
    if os.path.exists(MODEL_PATH):
        st.info("Model available.")
        df = pd.read_csv(data_path).head(5)
        st.dataframe(df)

with tab2:
    st.subheader("Ingest Applicant Documents")
    applicant_id = st.text_input("Applicant ID", value="1001")
    docs_folder = st.text_input("Folder path with PDFs/TXTs", value="data/sample_docs/1001")
    if st.button("Ingest Now"):
        try:
            ingest_folder(applicant_id, docs_folder)
            st.success(f"Ingested docs for {applicant_id}")
        except Exception as e:
            st.error(f"Failed: {e}")

with tab3:
    st.subheader("LLM Assistant")
    colA, colB = st.columns([1,2])
    with colA:
        applicant_id = st.text_input("Applicant ID to analyze", value="1001", key="aid2")
        # Load applicant features from CSV (toy) for now
        df = pd.read_csv("data/applications_sample.csv")
        row = df[df["applicant_id"]==int(applicant_id)]
        app_features = row.drop(columns=["approved"]).to_dict(orient="records")
        app_features = app_features[0] if app_features else {}
        st.json(app_features)
        if st.button("Summarize Documents"):
            st.write(summarize_applicant(applicant_id))
    with colB:
        st.write("")
        question = st.text_input("Ask a question about this applicant's docs", value="Any anomalies in income vs obligations?")
        if st.button("Ask"):
            st.write(answer_query(applicant_id, question))
        st.markdown("---")
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH,"rb") as f: model = pickle.load(f)
            with open(SCALER_PATH,"rb") as f: scaler = pickle.load(f)
            if app_features:
                # Preprocess features the same way as in training
                feature_df = row.drop(columns=["approved", "applicant_id", "name"])
                
                # Handle categorical variables by one-hot encoding
                categorical_columns = feature_df.select_dtypes(include=['object']).columns
                if len(categorical_columns) > 0:
                    # One-hot encode categorical variables
                    feature_df = pd.get_dummies(feature_df, columns=categorical_columns, drop_first=True)
                
                X = feature_df.values
                Xs = scaler.transform(X)
                try:
                    proba = model.predict_proba(Xs)[:,1][0]
                except Exception:
                    proba = float(model.decision_function(Xs)[0])
                    proba = 1/(1+pow(2.71828,-proba))
                st.info(f"Estimated risk score (higher= riskier): {proba:.3f}")
                if st.button("Recommend Action"):
                    st.code(recommend(app_features, float(proba)), language="json")
        else:
            st.warning("Train the model first.")

with tab4:
    st.subheader("RAG Debug / Inspect")
    applicant_id = st.text_input("Applicant ID", value="1001", key="aid3")
    query = st.text_input("Query to preview retrieved chunks", value="income stability and obligations")
    topk = st.slider("Top K", 1, 10, 5)
    if st.button("Search"):
        hits = RAGStore().search(applicant_id, query, top_k=topk)
        for h in hits:
            st.write(f"**{h['meta']['doc_name']}** (score={h['score']:.3f})")
            st.text(h["text"][:800])
