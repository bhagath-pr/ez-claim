import uuid
from graph import claim_graph
from database import engine, Base

def setup_db():
    # Sync schema for testing purposes.
    Base.metadata.create_all(bind=engine)

def main():
    setup_db()
    
    tx_id = str(uuid.uuid4())
    print(f"--- Starting Triage Simulation for tx {tx_id} ---")
    
    initial_state = {
        "transaction_id": tx_id
    }
    
    # Run graph
    for output in claim_graph.stream(initial_state):
        for key, value in output.items():
            print(f"Finished Node: {key}")
            
    print("--- Simulation Complete ---")

if __name__ == "__main__":
    main()
