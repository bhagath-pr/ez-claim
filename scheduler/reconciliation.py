"""
reconciliation.py — Track 3, Task 2: Async Background Reconciliation Loop
 
WHAT THIS DOES
---------------
Simulates what happens *after* a patient is discharged:
  - A safety deposit was collected at the desk (from the Reasoner/Track 2 flow)
  - Some days later, the real insurer sends back the actual settlement amount
  - This script polls a MOCK insurer API on a schedule, and when a
    transaction's settlement "arrives", it compares the deposit collected
    against the real shortfall and decides: refund the patient, or send a
    collection alert.
 
GOLDEN RULE
-----------
The comparison here is simple subtraction on two numbers that already
exist (deposit_collected, actual_shortfall) — this is bookkeeping, not
a statistical/ML calculation, so it's fine to do directly in Python.
No LLM is involved in this file at all.
 
HOW TO WIRE THIS UP LATER
--------------------------
Right now:
  - `MOCK_TRANSACTIONS` stands in for Track 2's persistent "Claim State"
    (Supabase/Postgres). Replace `load_pending_transactions()` and
    `save_transaction_status()` with real DB calls once that exists.
  - `mock_insurer_api_check()` stands in for a real call to an insurer's
    settlement API. Replace it with a real HTTP call when available.
Nothing else needs to change.
"""
 
import random
import logging
from datetime import datetime
 
from apscheduler.schedulers.blocking import BlockingScheduler
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reconciliation")
 
 
# ---------------------------------------------------------------------------
# MOCK "CLAIM STATE" — stands in for Track 2's Supabase/Postgres table
# ---------------------------------------------------------------------------
 
MOCK_TRANSACTIONS = [
    {
        "claim_id": "CLM-1001",
        "deposit_collected": 8500,
        "status": "awaiting_settlement",
    },
    {
        "claim_id": "CLM-1002",
        "deposit_collected": 0,
        "status": "awaiting_settlement",
    },
    {
        "claim_id": "CLM-1003",
        "deposit_collected": 15000,
        "status": "awaiting_settlement",
    },
]
 
 
def load_pending_transactions():
    """Stands in for: SELECT * FROM claims WHERE status = 'awaiting_settlement'"""
    return [t for t in MOCK_TRANSACTIONS if t["status"] == "awaiting_settlement"]
 
 
def save_transaction_status(claim_id: str, new_status: str):
    """Stands in for: UPDATE claims SET status = ... WHERE claim_id = ..."""
    for t in MOCK_TRANSACTIONS:
        if t["claim_id"] == claim_id:
            t["status"] = new_status
 
 
# ---------------------------------------------------------------------------
# MOCK INSURER API — stands in for a real settlement webhook/endpoint
# ---------------------------------------------------------------------------
 
def mock_insurer_api_check(claim_id: str):
    """
    Simulates asking the insurer: "has this claim settled yet, and if so,
    what was the real shortfall?"
 
    Returns None if not settled yet, or a dict with the real numbers if it has.
    In real life this becomes a requests.get() to the insurer's API.
    """
    # Randomly simulate: not yet settled, settled with a shortfall, or
    # settled with no shortfall at all.
    outcome = random.choice([None, "shortfall", "no_shortfall"])
 
    if outcome is None:
        return None
 
    if outcome == "shortfall":
        return {"settled": True, "actual_shortfall": random.randint(1000, 12000)}
 
    return {"settled": True, "actual_shortfall": 0}
 
 
# ---------------------------------------------------------------------------
# RECONCILIATION LOGIC
# ---------------------------------------------------------------------------
 
def reconcile(transaction: dict, settlement: dict):
    """
    Compares the deposit collected against the real shortfall and decides
    the outcome. This is plain arithmetic on two already-known numbers —
    not a statistical prediction — so it's fine to do here directly.
    """
    claim_id = transaction["claim_id"]
    deposit = transaction["deposit_collected"]
    actual_shortfall = settlement["actual_shortfall"]
 
    if deposit > actual_shortfall:
        refund_amount = deposit - actual_shortfall
        logger.info(
            "[REFUND] Claim %s: deposit was ₹%s, real shortfall was ₹%s. "
            "Refund ₹%s to the patient.",
            claim_id, deposit, actual_shortfall, refund_amount,
        )
    elif deposit < actual_shortfall:
        collection_amount = actual_shortfall - deposit
        logger.warning(
            "[COLLECTION ALERT] Claim %s: deposit was ₹%s, real shortfall was ₹%s. "
            "Collect an additional ₹%s from the patient.",
            claim_id, deposit, actual_shortfall, collection_amount,
        )
    else:
        logger.info(
            "[SETTLED] Claim %s: deposit exactly matched the real shortfall (₹%s). No action needed.",
            claim_id, deposit,
        )
 
    save_transaction_status(claim_id, "reconciled")
 
 
# ---------------------------------------------------------------------------
# THE SCHEDULED JOB
# ---------------------------------------------------------------------------
 
def poll_insurer_job():
    """This is the function APScheduler calls on every interval."""
    pending = load_pending_transactions()
 
    if not pending:
        logger.info("No pending transactions to check.")
        return
 
    logger.info("Checking insurer settlement status for %d pending claim(s)...", len(pending))
 
    for transaction in pending:
        settlement = mock_insurer_api_check(transaction["claim_id"])
 
        if settlement is None:
            logger.info("Claim %s: not settled yet.", transaction["claim_id"])
            continue
 
        reconcile(transaction, settlement)
 
 
# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
 
def main():
    logger.info("Starting reconciliation scheduler. Press Ctrl+C to stop.")
 
    scheduler = BlockingScheduler()
    # Every 10 seconds here for easy testing/demo.
    # In production this would be more like every few hours (e.g. "interval", hours=6).
    scheduler.add_job(poll_insurer_job, "interval", seconds=10, next_run_time=datetime.now())
 
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")
 
 
if __name__ == "__main__":
    main()
 