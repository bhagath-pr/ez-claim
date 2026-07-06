import uuid
from pyfiglet import figlet_format
from rich.console import Console
from rich.panel import Panel

import argparse

# Import components from other modules
from ingest_extractor import main as run_ingestion
from graph import claim_graph
from database import engine, Base

console = Console()

def print_master_banner():
    fig = figlet_format("EZ Pipeline", font="slant")
    console.print(fig, style="bold magenta")
    console.print("   Unified End-to-End Runner v1.0", style="dim")
    console.print("=" * 60, style="magenta")
    console.print()

def setup_db():
    Base.metadata.create_all(bind=engine)

def main():
    parser = argparse.ArgumentParser(description="Run the EZ Claim Unified Pipeline.")
    parser.add_argument(
        "bill_filename", 
        nargs="?", 
        default="sample_bill.pdf", 
        help="The filename of the bill inside the raw_bills/ directory. (default: sample_bill.pdf)"
    )
    args = parser.parse_args()

    print_master_banner()
    
    console.print(f"[bold cyan]▶ STEP 1: Data Ingestion & LLM Extraction ({args.bill_filename})[/bold cyan]")
    try:
        # This will run the extraction and print its own banner and logs.
        run_ingestion(target_bill_filename=args.bill_filename)
    except Exception as e:
        console.print(f"[bold red]Ingestion failed:[/bold red] {e}")
        return
        
    console.print("\n[bold cyan]▶ STEP 2: Graph Orchestration & AI Reasoning[/bold cyan]")
    setup_db()
    tx_id = str(uuid.uuid4())
    console.print(f"[dim]Initialized Graph State for transaction {tx_id}[/dim]")
    
    initial_state = {"transaction_id": tx_id}
    final_reasoning = None
    final_status = None
    
    # Run the graph and stream outputs
    for output in claim_graph.stream(initial_state):
        for key, value in output.items():
            console.print(f"[bold green]✓[/bold green] [dim]Finished Node:[/dim] [white]{key}[/white]")
            
            # The LangGraph stream yields the state updates from each node.
            # We capture the reasoning analysis and final status from the state updates.
            if isinstance(value, dict):
                if "reasoner_analysis" in value and value["reasoner_analysis"]:
                    final_reasoning = value["reasoner_analysis"]

                if "final_status" in value and value["final_status"]:
                    final_status = value["final_status"]
                
    console.print("\n[bold cyan]▶ STEP 3: Final Output[/bold cyan]")
    if final_reasoning:
        # Determine panel color based on triage status
        if final_status == "APPROVED":
            color = "green"
        elif final_status == "PENDING_DEPOSIT":
            color = "yellow"
        else:
            color = "red"
            
        panel = Panel(
            final_reasoning, 
            title=f"[{color} bold]Verdict: {final_status}[/{color} bold]", 
            border_style=color,
            padding=(1, 2)
        )
        console.print(panel)
    else:
        console.print("[bold red]No reasoning analysis was produced.[/bold red]")

if __name__ == "__main__":
    main()
