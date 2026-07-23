import os
import time
import json
import uuid
import asyncio
from typing import AsyncGenerator
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Import existing pipeline modules
from ingest_extractor import extract_bill_content, query_qwen_extractor
from graph import claim_graph
from database import engine, Base

app = FastAPI(title="EZ Claim Pipeline API", version="1.0.0")

# Enable CORS for React frontend (Vite default dev port 5173 / 3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def setup_db():
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print(f"[Warning] Database initialization error: {e}")

@app.on_event("startup")
def startup_event():
    setup_db()

@app.get("/api/health")
def health_check():
    return {"status": "online", "message": "EZ Claim Pipeline API is running"}

@app.post("/api/process-claim")
async def process_claim(file: UploadFile = File(...)):
    """
    Synchronous processing endpoint for claim PDF upload.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        # Step 1: Save uploaded PDF to raw_bills directory
        INPUT_DIR = "raw_bills"
        os.makedirs(INPUT_DIR, exist_ok=True)
        pdf_path = os.path.join(INPUT_DIR, file.filename)
        
        contents = await file.read()
        with open(pdf_path, "wb") as f:
            f.write(contents)

        # Step 2: Ingestion & Text Extraction
        cleaned_markdown_text = extract_bill_content(pdf_path)
        if not cleaned_markdown_text:
            raise HTTPException(status_code=422, detail="Failed to extract text from the provided PDF.")

        # Step 3: AI Document Structuring
        raw_json_string = query_qwen_extractor(cleaned_markdown_text)
        if not raw_json_string:
            raise HTTPException(status_code=500, detail="AI extraction failed. Please check GROQ_API_KEY environment variable.")

        try:
            final_json_payload = json.loads(raw_json_string)
            OUTPUT_DIR = "extracted_json"
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            output_json_file = os.path.join(OUTPUT_DIR, "extracted_claim.json")
            with open(output_json_file, "w", encoding="utf-8") as json_out:
                json.dump(final_json_payload, json_out, indent=4)
        except json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="Invalid JSON returned by AI model.")

        # Step 4: Graph Reasoning Pipeline
        tx_id = str(uuid.uuid4())
        initial_state = {"transaction_id": tx_id}

        final_reasoning = None
        final_status = None
        required_deposit = 0
        predicted_approval_prob = None
        predicted_payout_ratio = None

        for output in claim_graph.stream(initial_state):
            for key, value in output.items():
                if isinstance(value, dict):
                    if value.get("reasoner_analysis"):
                        final_reasoning = value["reasoner_analysis"]
                    if value.get("final_status"):
                        final_status = value["final_status"]
                    if "required_deposit" in value:
                        required_deposit = value["required_deposit"]
                    if "predicted_approval_prob" in value:
                        predicted_approval_prob = value["predicted_approval_prob"]
                    if "predicted_payout_ratio" in value:
                        predicted_payout_ratio = value["predicted_payout_ratio"]

        return {
            "success": True,
            "transaction_id": tx_id,
            "filename": file.filename,
            "extracted_data": final_json_payload,
            "final_status": final_status or "PENDING_DEPOSIT",
            "required_deposit": required_deposit,
            "predicted_approval_prob": predicted_approval_prob,
            "predicted_payout_ratio": predicted_payout_ratio,
            "reasoner_analysis": final_reasoning or "No reasoning analysis produced."
        }

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/process-claim/stream")
async def process_claim_stream(file: UploadFile = File(...)):
    """
    Streaming Server-Sent Events (SSE) endpoint to provide real-time node-by-node updates to the React UI.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()

    async def event_generator() -> AsyncGenerator[str, None]:
        def send_event(step: str, status: str, message: str, payload: dict = None):
            data = {
                "step": step,
                "status": status,  # "running", "completed", "error"
                "message": message,
                "payload": payload or {}
            }
            return f"data: {json.dumps(data)}\n\n"

        # 1. File Upload
        yield send_event("upload", "completed", "File uploaded successfully.")
        await asyncio.sleep(0.5)

        # 2. Extract content from PDF
        yield send_event("extract_pdf", "running", "Extracting content from PDF...")
        INPUT_DIR = "raw_bills"
        os.makedirs(INPUT_DIR, exist_ok=True)
        pdf_path = os.path.join(INPUT_DIR, file.filename)
        with open(pdf_path, "wb") as f:
            f.write(contents)

        await asyncio.sleep(1.0)
        cleaned_markdown_text = extract_bill_content(pdf_path)
        if not cleaned_markdown_text:
            yield send_event("extract_pdf", "error", "Failed to extract content from PDF.")
            return

        yield send_event("extract_pdf", "completed", "PDF text & tables extracted.")
        await asyncio.sleep(0.5)

        # 3. AI Analysis
        yield send_event("ai_analysis", "running", "Analyzing document with AI...")
        await asyncio.sleep(1.0)
        raw_json_string = query_qwen_extractor(cleaned_markdown_text)

        if not raw_json_string:
            yield send_event("ai_analysis", "error", "AI Analysis Failed.")
            return

        try:
            final_json_payload = json.loads(raw_json_string)
            OUTPUT_DIR = "extracted_json"
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            output_json_file = os.path.join(OUTPUT_DIR, "extracted_claim.json")
            with open(output_json_file, "w", encoding="utf-8") as json_out:
                json.dump(final_json_payload, json_out, indent=4)
            yield send_event("ai_analysis", "completed", "AI Extraction Complete", {"extracted_json": final_json_payload})
        except json.JSONDecodeError:
            yield send_event("ai_analysis", "error", "Invalid AI JSON output.")
            return

        await asyncio.sleep(0.5)

        # 4. LangGraph Reasoning
        yield send_event("reasoning_graph", "running", "Running reasoning graph...")
        await asyncio.sleep(1.0)

        tx_id = str(uuid.uuid4())
        initial_state = {"transaction_id": tx_id}

        final_reasoning = None
        final_status = None
        required_deposit = 0

        for output in claim_graph.stream(initial_state):
            for key, value in output.items():
                if isinstance(value, dict):
                    if value.get("reasoner_analysis"):
                        final_reasoning = value["reasoner_analysis"]
                    if value.get("final_status"):
                        final_status = value["final_status"]
                    if "required_deposit" in value:
                        required_deposit = value["required_deposit"]

        yield send_event("reasoning_graph", "completed", "✓ Claim Processing Complete", {
            "transaction_id": tx_id,
            "final_status": final_status or "PENDING_DEPOSIT",
            "required_deposit": required_deposit,
            "reasoner_analysis": final_reasoning or "No reasoning analysis produced.",
            "extracted_data": final_json_payload
        })

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# Serve static React files if frontend is built
dist_dir = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.exists(dist_dir):
    app.mount("/", StaticFiles(directory=dist_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
