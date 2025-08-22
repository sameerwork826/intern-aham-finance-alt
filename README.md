# 🏦 Loan Application Assistant

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Prototype-orange.svg)](https://github.com/yourusername/loan-llm-assistant)

> **An intelligent loan processing system that combines ML risk assessment with LLM-powered document analysis and RAG-based insights.**

<div align="center">

![Loan Assistant Demo](https://img.shields.io/badge/Demo-Available-brightgreen?style=for-the-badge&logo=streamlit)

**Transform loan processing with AI-powered insights** 🚀

</div>

---

## ✨ Features

### 🤖 **AI-Powered Analysis**
- **ML Risk Model**: Logistic Regression/XGBoost for loan approval prediction
- **LLM Assistant**: Groq or Ollama integration for intelligent document analysis
- **RAG System**: FAISS + sentence-transformers for semantic document search

### 📊 **Smart Document Processing**
- **Multi-format Support**: PDF/TXT document ingestion and analysis
- **Intelligent Summarization**: AI-generated applicant profiles and risk assessments
- **Contextual Q&A**: Ask questions about applicants with grounded responses

### 🎯 **Loan Officer Tools**
- **Risk Scoring**: ML-based approval probability with explainable insights
- **Document Verification**: Cross-reference income, employment, and financial data
- **Recommendation Engine**: AI-backed approve/reject/need-more-docs decisions

### 🔧 **Developer Friendly**
- **One-Command Setup**: Automated environment and dependency management
- **Modular Architecture**: Clean separation of ML, LLM, and RAG components
- **Extensible Design**: Easy to add new models, data sources, or features

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Conda (recommended) or pip
- Groq API key (free tier available) or Ollama (local)

### One-Command Setup (Windows)

```bash
# Clone and setup everything automatically
git clone https://github.com/yourusername/loan-llm-assistant.git
cd loan-llm-assistant
run_all.cmd
```

### Manual Setup

```bash
# 1. Environment setup
conda create -n loanrisk python=3.10 -y
conda activate loanrisk
pip install -r requirements.txt

# 2. Configure LLM provider
cp .env.example .env
# Edit .env with your Groq API key or Ollama settings

# 3. Generate sample data and train model
python generate_data.py
python train_large.py

# 4. Launch the application
streamlit run app.py
```

---

## 🎯 Usage Guide

### 1. **Train Risk Model**
Navigate to the "Train Risk Model" tab:
- Upload your CSV dataset or use the generated sample data
- Click "Train Model" to build the ML risk assessment model
- View model performance metrics (AUC, precision, recall)

### 2. **Ingest Documents**
Use the "Ingest Docs" tab to process applicant documents:
```
Applicant ID: 1001
Folder Path: data/sample_docs/1001
```
- Supports PDF and TXT files
- Automatically chunks and embeds documents
- Builds searchable FAISS index

### 3. **AI Assistant**
The "LLM Assistant" tab provides:
- **Document Summarization**: AI-generated applicant profiles
- **Contextual Q&A**: Ask specific questions about applicants
- **Risk Assessment**: ML + LLM combined recommendations

### 4. **RAG Debug**
Use "RAG Debug" to:
- Preview document chunks and embeddings
- Test search relevance
- Fine-tune retrieval parameters

---

## 🏗️ Architecture

```
loan-llm-assistant/
├── 📁 Core Components
│   ├── app.py              # Streamlit UI
│   ├── chains.py           # LLM interaction chains
│   ├── model_train.py      # ML model training
│   ├── rag.py             # RAG system
│   └── llm_client.py      # LLM provider abstraction
│
├── 📁 Data & Models
│   ├── data/
│   │   ├── applications_sample.csv
│   │   └── sample_docs/    # Test documents
│   └── storage/           # Generated models & indexes
│
├── 📁 Utilities
│   ├── generate_data.py   # Synthetic data generator
│   ├── train_large.py     # Model training script
│   └── run_all.cmd        # One-command setup
│
└── 📁 Configuration
    ├── requirements.txt   # Dependencies
    └── .env.example      # Environment template
```

---

## 🔧 Configuration

### LLM Provider Setup

#### Groq (Recommended - Fast & Free)
```bash
# .env configuration
LLM_PROVIDER=groq
GROQ_API_KEY=your_api_key_here
GROQ_MODEL=llama3-8b-8192
```

#### Ollama (Local - Privacy First)
```bash
# .env configuration
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma:2b
```

### Supported Models
- **Groq**: `llama3-8b-8192`, `llama3-70b-8192`, `gemma2-9b-it`
- **Ollama**: Any local model (gemma:2b, llama3:8b, etc.)

---

## 📊 Sample Data

The project includes comprehensive sample data:

### Applicant Profiles
- **1001**: Salaried professional (₹8L income, good credit)
- **1002**: Self-employed business owner (₹4.5L income, moderate risk)
- **1003**: Senior analyst (₹12.5L income, stable employment)
- **1004**: Junior associate (₹3.6L income, entry-level)

### Document Types
- Bank statements with income/expense analysis
- Employment verification letters
- Credit history summaries
- Financial obligation details

---

## 🛠️ Development

### Adding New Features
1. **New LLM Provider**: Extend `llm_client.py`
2. **Custom Models**: Modify `model_train.py`
3. **Document Types**: Update `ingest_docs.py`
4. **UI Components**: Enhance `app.py`

### Testing
```bash
# Test Groq integration
python groq_smoke.py

# Test data generation
python generate_data.py

# Test model training
python train_large.py
```

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/yourusername/loan-llm-assistant.git
cd loan-llm-assistant
conda create -n loanrisk python=3.10 -y
conda activate loanrisk
pip install -r requirements.txt
```

---

## 📈 Performance

### Model Performance
- **Risk Model AUC**: ~0.97 (on synthetic data)
- **RAG Retrieval**: Sub-second response times
- **LLM Integration**: Real-time document analysis

### System Requirements
- **Memory**: 4GB+ RAM (8GB recommended)
- **Storage**: 2GB+ free space
- **Network**: Internet for Groq API calls

---

## 🚨 Troubleshooting

### Common Issues

#### Groq API Errors
```bash
# Model not found - try different models
GROQ_MODEL=llama3-8b-8192
GROQ_MODEL=llama3-70b-8192
GROQ_MODEL=gemma2-9b-it
```

#### Environment Issues
```bash
# Clean environment setup
conda deactivate
conda env remove -n loanrisk
conda create -n loanrisk python=3.10 -y
conda activate loanrisk
pip install -r requirements.txt
```

#### Windows-Specific
- Use `cmd` instead of PowerShell
- Ensure conda is in PATH
- Check for antivirus blocking

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<<<<<<< HEAD
## 🙏 Acknowledgments
=======
-Deployed webiste checkout
https://intern-aham-sameer.streamlit.app/
## Example Prompts (in UI)
- "Summarize this applicant's documents focusing on income stability, obligations, and anomalies."
- "List any missing documents needed for underwriting."
- "Does the declared salary reconcile with the bank statement inflows? Cite details."
>>>>>>> 61c5cd95767a40b8f534032f41d045185949264f

- **Streamlit** for the amazing web framework
- **Groq** for fast LLM inference
- **Hugging Face** for transformer models
- **FAISS** for efficient similarity search

---

<div align="center">

**Made with ❤️ for the AI/ML community**

[![GitHub stars](https://img.shields.io/github/stars/yourusername/loan-llm-assistant?style=social)](https://github.com/yourusername/loan-llm-assistant)
[![GitHub forks](https://img.shields.io/badge/GitHub-Fork-blue?style=social&logo=github)](https://github.com/yourusername/loan-llm-assistant/fork)

</div>
