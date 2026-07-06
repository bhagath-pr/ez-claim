import streamlit as st
import time
import os
import uuid
import json

# Setup Streamlit page configuration first
st.set_page_config(page_title="EZ Claim Pipeline", layout="centered")

from ingest_extractor import extract_bill_content, query_qwen_extractor
from graph import claim_graph
from database import engine, Base

def setup_db():
    Base.metadata.create_all(bind=engine)

st.title("EZ Claim Processing")

st.markdown("Upload a hospital invoice (PDF) to automatically extract, process, and evaluate the claim using AI.")

uploaded_file = st.file_uploader("Select Medical Bill (PDF)", type=["pdf"])

if uploaded_file is not None:
    if st.button("Process Claim", use_container_width=True):
        # Save file to temporary location
        INPUT_DIR = "raw_bills"
        os.makedirs(INPUT_DIR, exist_ok=True)
        pdf_path = os.path.join(INPUT_DIR, uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # Unified status container to prevent section-wise blocky alerts
        with st.status("Processing Claim...", expanded=True) as status:
            st.write("📄 File uploaded successfully.")
            
            # Step 1: Ingestion
            st.write("🔍 Extracting content from PDF...")
            time.sleep(1.5) # Artificial delay
            cleaned_markdown_text = extract_bill_content(pdf_path)
            
            st.write("🧠 Analyzing document with AI...")
            time.sleep(1.5) # Artificial delay
            raw_json_string = query_qwen_extractor(cleaned_markdown_text)
            
            if not raw_json_string:
                status.update(label="AI Analysis Failed", state="error", expanded=True)
                st.stop()
                
            try:
                final_json_payload = json.loads(raw_json_string)
                OUTPUT_DIR = "extracted_json"
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                output_json_file = os.path.join(OUTPUT_DIR, "extracted_claim.json")
                with open(output_json_file, "w", encoding="utf-8") as json_out:
                    json.dump(final_json_payload, json_out, indent=4)
                st.write("✅ AI Extraction Complete")
            except json.JSONDecodeError:
                status.update(label="Invalid AI Output", state="error", expanded=True)
                st.stop()

            # Step 2: Graph Orchestration
            st.write("⚙️ Running reasoning graph...")
            time.sleep(2) # Artificial delay
            setup_db()
            tx_id = str(uuid.uuid4())
            initial_state = {"transaction_id": tx_id}
            
            final_reasoning = None
            final_status = None
            
            for output in claim_graph.stream(initial_state):
                for key, value in output.items():
                    if isinstance(value, dict):
                        if value.get("reasoner_analysis"):
                            final_reasoning = value["reasoner_analysis"]
                        if value.get("final_status"):
                            final_status = value["final_status"]
            
            status.update(label="Claim Processing Complete", state="complete", expanded=False)

        # Polished final output display without blocky alerts
        st.markdown("---")
        if final_reasoning:
            # Color code for headers based on verdict
            color_map = {
                "APPROVED": "#2E8B57", # Sea Green
                "PENDING_DEPOSIT": "#D2691E", # Chocolate/Orange-brown
            }
            status_color = color_map.get(final_status, "#8B0000") # Default Dark Red
            
            st.markdown(f"<h3 style='color: {status_color};'>Verdict: {final_status}</h3>", unsafe_allow_html=True)
            st.markdown(final_reasoning)
        else:
            st.error("No reasoning analysis was produced.")
