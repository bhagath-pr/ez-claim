import os
import json
import requests
import pdfplumber
from pyfiglet import figlet_format
from rich.console import Console
from rich.text import Text

console = Console()

def print_banner():
    fig = figlet_format("EZ Claim", font="slant")
    console.print(fig, style="bold cyan")
    console.print("   Hospital Claims Processor v0.1", style="dim")
    console.print("-" * 50, style="cyan")
    console.print()

def table_to_markdown(table_data):
    """Converts pdfplumber nested list tables into explicit Markdown syntax."""
    if not table_data:
        return ""
    markdown_lines = []
    for i, row in enumerate(table_data):
        clean_row = [str(cell).replace('\n', ' ').strip() if cell else "" for cell in row]
        if not any(clean_row):
            continue

        markdown_lines.append("| " + " | ".join(clean_row) + " |")
        if i == 0:
            separator = "| " + " | ".join(["---"] * len(clean_row)) + " |"
            markdown_lines.append(separator)

    return "\n".join(markdown_lines)

def extract_bill_content(pdf_path):
    """Passes through the PDF to capture raw text headers and formatted markdown tables."""
    assembled_markdown = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            assembled_markdown.append(f"\n--- PAGE {page_num} ---")

            raw_text = page.extract_text(layout=False)
            if raw_text:
                assembled_markdown.append(raw_text)

            tables = page.extract_tables()
            for table in tables:
                md_table = table_to_markdown(table)
                if md_table:
                    assembled_markdown.append("\n" + md_table + "\n")

    return "\n".join(assembled_markdown)

def query_qwen_extractor(prompt_content):
    """Dispatches the payload to Ollama locally, enforcing strict structural output format."""
    ollama_url = "http://localhost:11434/api/generate"

    system_instruction = (
        "You are a precise healthcare bill data parser. Your sole objective is to scan the text "
        "and return a valid JSON object matching the requested schema. Do not calculate totals, "
        "do not make conversational filler, and do not perform any math arithmetic yourself. "
        "If a field is completely missing from the document text, assign it a value of null."
    )

    user_prompt = f"""
    Analyze this hospital bill document and extract the following clinical and financial primitives:
    - patient_age (integer or null)
    - policy_number (string or null)
    - policy_year (string or null)
    - annual_sum_insured (integer or null)
    - diagnosis_code (string or null)
    - procedure_code (string or null)
    - treatment_category (string or null)
    - claim_amount (integer or null)

    Document Content:
    {prompt_content}
    """

    payload = {
        "model": "qwen2.5:7b",
        "prompt": user_prompt,
        "system": system_instruction,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.0
        }
    }

    with console.status("[bold green]Streaming data payload to local Qwen2.5 instance..."):
        try:
            response = requests.post(ollama_url, json=payload)
            response.raise_for_status()
            result_data = response.json()
            return result_data.get("response")
        except requests.exceptions.RequestException as e:
            console.print(f"[bold red]Ollama Connection Failure:[/bold red] {e}")
            return None

def main():
    print_banner()

    # --- DIRECTORY CONFIGURATION ---
    INPUT_DIR = "raw_bills"
    OUTPUT_DIR = "extracted_json"

    # Dynamically establish workspace folders if missing
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Target files mapped inside the specific directories
    target_bill_filename = "sample_bill.pdf"
    target_bill_pdf = os.path.join(INPUT_DIR, target_bill_filename)
    output_json_file = os.path.join(OUTPUT_DIR, "extracted_claim.json")

    # Check if a target file is ready for processing
    if not os.path.exists(target_bill_pdf):
        console.print(f"[yellow]⚠️  No target file found at: [cyan]{target_bill_pdf}[/cyan][/yellow]")
        console.print(f"[dim]Action Required: Create the '[cyan]{INPUT_DIR}/[/cyan]' folder and drop '[cyan]{target_bill_filename}[/cyan]' into it to test.[/dim]\n")
        return

    console.print(f"[bold blue]Processing Ingestion Stage for:[/bold blue] {target_bill_pdf}")

    # Step 1: Document Parsing via pdfplumber
    with console.status("[bold yellow]Re-assembling structural layouts into Markdown..."):
        cleaned_markdown_text = extract_bill_content(target_bill_pdf)

    console.print("[green]✓ Text and structural tables successfully extracted and serialized.[/green]")

    # Step 2: Local Model Extraction Inference
    raw_json_string = query_qwen_extractor(cleaned_markdown_text)

    if raw_json_string:
        try:
            final_json_payload = json.loads(raw_json_string)

            with open(output_json_file, "w", encoding="utf-8") as json_out:
                json.dump(final_json_payload, json_out, indent=4)

            console.print(f"\n[bold green]✓ Ingestion Stage Complete![/bold green]")
            console.print(f"Primitives successfully logged into [cyan]{output_json_file}[/cyan]")
            console.print("[dim]Handoff clean: Downstream modules can safely poll this JSON folder.[/dim]\n")

        except json.JSONDecodeError:
            console.print("[bold red]Critical Error: Received invalid non-JSON format string from local model.[/bold red]")

if __name__ == "__main__":
    main()
