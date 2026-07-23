import random
import logging
from datetime import datetime
from apscheduler.schedulers.blocking import BlockingScheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reconciliation")

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
    return [t for t in MOCK_TRANSACTIONS if t["status"] == "awaiting_settlement"]

def save_transaction_status(claim_id: str, new_status: str):
    for t in MOCK_TRANSACTIONS:
        if t["claim_id"] == claim_id:
            t["status"] = new_status

def mock_insurer_api_check(claim_id: str):
    outcome = random.choice([None, "shortfall", "no_shortfall"])
    if outcome is None:
        return None
    if outcome == "shortfall":
        return {"settled": True, "actual_shortfall": random.randint(1000, 12000)}
    return {"settled": True, "actual_shortfall": 0}

def reconcile(transaction: dict, settlement: dict):
    claim_id = transaction["claim_id"]
    deposit = transaction["deposit_collected"]
    actual_shortfall = settlement["actual_shortfall"]

    if deposit > actual_shortfall:
        refund_amount = deposit - actual_shortfall
        logger.info(
            "[REFUND] Claim %s: deposit was ₹%s, real shortfall was ₹%s. Refund ₹%s to the patient.",
            claim_id, deposit, actual_shortfall, refund_amount,
        )
    elif deposit < actual_shortfall:
        collection_amount = actual_shortfall - deposit
        logger.warning(
            "[COLLECTION ALERT] Claim %s: deposit was ₹%s, real shortfall was ₹%s. Collect an additional ₹%s from the patient.",
            claim_id, deposit, actual_shortfall, collection_amount,
        )
    else:
        logger.info(
            "[SETTLED] Claim %s: deposit exactly matched the real shortfall (₹%s). No action needed.",
            claim_id, deposit,
        )

    save_transaction_status(claim_id, "reconciled")

def poll_insurer_job():
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

def main():
    logger.info("Starting reconciliation scheduler. Press Ctrl+C to stop.")
    scheduler = BlockingScheduler()
    scheduler.add_job(poll_insurer_job, "interval", seconds=10, next_run_time=datetime.now())

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped.")

if __name__ == "__main__":
    main()
