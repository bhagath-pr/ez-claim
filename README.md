# EZ Claim: End-to-End Hospital Claims Ingestion & Triage Pipeline

EZ Claim is an intelligent medical claims processing and routing system. It streamlines the hospital discharge process by automatically extracting, classifying, evaluating, and reconciling patient insurance claims. 

Using **LangGraph** to orchestrate workflows, **ChromaDB** for similar case retrieval (RAG), and a hybrid model architecture (**scikit-learn** classifiers/regressors combined with **Ollama** and **Groq-hosted LLMs**), EZ Claim automates claim decisioning (Green/Yellow/Red paths), calculates required patient deposits, and provides clear, natural-language justifications for hospital administrators.

---

## Key Features

1. **AI-Powered Bill Extraction (Ollama & pdfplumber)**
   - Parses unstructured hospital invoices (PDF format) and extracts key primitives (e.g., diagnosis, policy terms, billing line items).
   - Utilizes a local `Qwen 2.5:7b` instance (running via Ollama) to structure raw text into a standardized JSON payload, automatically inferring **ICD-10 classification codes** and **global medical categories**.

2. **Graph-Driven Triage Orchestration (LangGraph)**
   - Manages the claim lifecycle across multiple states:
     - **Ingestion**: Retrieves extracted claim data.
     - **Reference Lookup (RAG)**: Queries a semantic database of historical claims for similar cases.
     - **ML Inference**: Uses custom scikit-learn models to predict approval probability and expected payout ratio. It also computes a deterministic "hard math cap" in Python (adhering to policy limits).
     - **Triage Matrix Routing**: Branches path based on ML scores:
       - **Green Path** (Low Risk): Fully approved, deposit = ₹0.
       - **Yellow Path** (Partial Risk): Pending deposit. Collects the difference between the claim amount and predicted payout.
       - **Red Path** (High Risk): Escalated/Rejected. Collects full claim amount.
     - **Reasoner Node**: Queries cloud-hosted Qwen via Groq to write a natural-language verdict explanation (4–6 sentences) for front-desk clerks.
     - **Persistence**: Persists transaction outcomes dynamically into a Postgres/SQLite database.

3. **Asynchronous Background Reconciliation (APScheduler)**
   - Simulates post-discharge bookkeeping once the real insurer settlement arrives.
   - Compares the actual shortfall against the deposit collected at the discharge desk to generate instructions for patient refunds or collections alerts.

4. **Web Frontend (Streamlit)**
   - Provides a clean, modern user interface where hospital staff can upload hospital invoices, track pipeline execution node-by-node, and view the final adjudication details.

---

## Repository Architecture

```
ez-claim/
├── app.py                      # Streamlit frontend application
├── main.py                     # CLI simulation entrypoint
├── run_pipeline.py             # E2E unified CLI pipeline runner
├── graph.py                    # LangGraph orchestration state machine
├── state.py                    # LangGraph claim state schema
├── models.py                   # SQLAlchemy ORM models (ClaimTransaction)
├── database.py                 # SQLite/PostgreSQL engine and session setup
├── requirements.txt            # Python environment dependencies
│
├── ingest_extractor.py         # PDF parsing + local Qwen Ollama JSON extractor
├── raw_bills/                  # Target directory for uploaded raw PDF invoices
├── extracted_json/             # Output directory for structured claim JSON
│
├── classifier/                 # Track 1 Machine Learning models
│   ├── approval_classifier.joblib
│   ├── payout_regressor.joblib
│   └── model_features.joblib
│
├── embedding/                  # Track 1/Track 2 Search & Indexing (RAG)
│   ├── document_builder.py     # Serializes claims into natural text
│   ├── embed_documents.py      # Generates vectors with Hugging Face Sentence Transformers
│   ├── vector_store.py         # Persistent ChromaDB collection wrapper
│   └── retriever.py            # High-level query/claim search interface
│
├── reasoner/                   # Track 3 LLM Justification Writer
│   ├── reasoner.py             # Groq API Qwen master prompt assembler
│   └── reasoner_output/        # Stores latest reasoner logs
│
├── scheduler/                  # Track 3 Reconciliation scheduler
│   └── reconciliation.py       # Asynchronous background loop using APScheduler
│
└── tests/                      # Unit testing scripts
    ├── run_embedding.py        # Populates vector DB with the latest claim
    └── test_pipeline.py        # Vector store + retriever flow integration test
```

---

## Setup & Installation

### 1. Prerequisites
- **Python**: version `3.10` or higher is recommended.
- **Ollama**: Required to run the local extraction model. Install it from [ollama.com](https://ollama.com).
- **PostgreSQL** (Optional): A local PostgreSQL server instance. By default, it falls back to a PostgreSQL connection URL but can also be adapted for SQLite/other databases.

### 2. Environment Setup
Clone the repository and set up a virtual environment:
```bash
# Set up virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Model Preparation
Download and pull the extraction model locally through Ollama:
```bash
ollama run qwen2.5:7b-instruct-q4_K_M
```

### 4. Configuration (`.env`)
Create a `.env` file in the root directory to store environment variables:
```env
# Groq API Configuration for the Reasoner LLM
GROQ_API_KEY=your_groq_api_key_here

# Model overrides (optional)
GROQ_MODEL=qwen/qwen3-32b

# Database connection URL (defaults to localhost PostgreSQL)
DATABASE_URL=postgresql://<db_user>:<db_password>@localhost:5432/<db_name>
```

---

### Run React Web Application (New)
Launch the unified React web application and FastAPI backend server:
```bash
# Launch backend server & React application
python run_react_app.py
```
Or run the React frontend in Vite development mode:
```bash
# Start backend API server
python server.py

# In another terminal, start React dev server
cd frontend
npm install
npm run dev
```
Open [http://localhost:8000](http://localhost:8000) (or [http://localhost:5173](http://localhost:5173) for Vite dev server).

### Run Legacy Streamlit Web Application
This runs the legacy Streamlit interface:
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.


### Run Unified CLI Pipeline
You can test the entire ingestion and triage pipeline directly on a raw invoice file:
```bash
# Put a PDF invoice inside raw_bills/ directory
python run_pipeline.py sample_bill.pdf
```

### Run Simulation Entrypoint
Simulates a triage run with either live files or mock fallbacks:
```bash
python main.py
```

### Run Reconciliation Loop
Runs the background service to process deposits against insurer settlement webhooks:
```bash
python scheduler/reconciliation.py
```

### Run Embedding & Indexing Tests
To populate the Chroma database with claim documents or test the semantic search retriever:
```bash
# To index the latest extracted claim
python tests/run_embedding.py

# To test the retriever search logic
python tests/test_pipeline.py
```
