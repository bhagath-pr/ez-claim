"""
app.py — Track 3, Task 3: Minimal Streamlit UI
 
WHAT THIS DOES
---------------
A bare-bones frontend so a human can:
  1. Upload a hospital bill PDF
  2. See the extracted claim JSON (from Member A's pipeline, if wired up —
     otherwise falls back to dummy data)
  3. Click a button to run the Reasoner and see the plain-language
     explanation for the discharge desk
 
This is intentionally minimal — per the project doc, frontend is a
secondary priority. The goal is just to demo the pipeline end-to-end,
not to build a polished product UI.
 
HOW TO WIRE THIS UP LATER
--------------------------
- Replace `run_extraction_pipeline()` with a real call into Member A's
  `ingest_extractor.py` once you want the UI to do real PDF extraction
  instead of using dummy data.
- Replace the dummy historical examples / stats the same way you did in
  reasoner.py, once Member B's search and Track 1's models exist.
"""
 
import sys
import os
import json
 
import streamlit as st
 
# Make the reasoner module (in ../reasoner/) importable from here
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "reasoner"))
 
from reasoner import (
    get_dummy_extracted_claim,
    get_dummy_historical_examples,
    get_dummy_stats,
    build_master_prompt,
    call_reasoner_llm,
)
 
 
st.set_page_config(page_title="EZ Claim — Demo", layout="centered")
 
st.title("EZ Claim — Discharge Desk Demo")
st.caption("Upload a bill, see the extracted claim, and get an instant explanation.")
 
# ---------------------------------------------------------------------------
# Step 1: Upload
# ---------------------------------------------------------------------------
 
uploaded_file = st.file_uploader("Upload a hospital bill (PDF)", type=["pdf"])
 
if uploaded_file is not None:
    st.success(f"Received: {uploaded_file.name}")
    st.info(
        "Note: PDF extraction isn't wired up in this demo yet — showing "
        "sample extracted data below. Once Member A's extractor is "
        "connected, this will show the real parsed claim."
    )
else:
    st.info("No file uploaded yet — showing sample extracted data below.")
 
# ---------------------------------------------------------------------------
# Step 2: Show extracted claim (dummy for now)
# ---------------------------------------------------------------------------
 
extracted_claim = get_dummy_extracted_claim()
 
st.subheader("Extracted Claim")
st.json(extracted_claim)
 
# ---------------------------------------------------------------------------
# Step 3: Run the Reasoner
# ---------------------------------------------------------------------------
 
st.subheader("Discharge Recommendation")
 
if st.button("Run Reasoner"):
    historical_examples = get_dummy_historical_examples()
    stats = get_dummy_stats()
 
    with st.expander("See historical examples used"):
        st.json(historical_examples)
 
    with st.expander("See statistical calculations used"):
        st.json(stats)
 
    if not os.environ.get("REASONER_API_KEY"):
        st.error(
            "REASONER_API_KEY is not set. Set it before launching Streamlit, e.g.:\n\n"
            "  $env:REASONER_API_KEY=\"your_key\"\n"
            "  $env:REASONER_MODEL=\"openrouter/free\"\n"
            "  streamlit run frontend/app.py"
        )
    else:
        with st.spinner("Asking the Reasoner..."):
            prompt = build_master_prompt(extracted_claim, historical_examples, stats)
            try:
                explanation = call_reasoner_llm(prompt)
                st.success("Recommendation ready")
                st.write(explanation)
            except Exception as e:
                st.error(f"Reasoner call failed: {e}")
else:
    st.write("Click the button above to generate a recommendation.")
 