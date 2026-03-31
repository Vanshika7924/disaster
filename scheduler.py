# =============================================================================
# scheduler.py
# Runs the full pipeline automatically every N minutes.
#
# Usage:
#   python scheduler.py                  → runs every 30 minutes (default)
#   python scheduler.py --interval 10   → runs every 10 minutes
#   python scheduler.py --once          → run once and exit (for testing)
# =============================================================================

import argparse
import logging
import time
import signal
import sys

from pipeline import run_pipeline
from config   import SCHEDULER_INTERVAL_MINUTES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("scheduler.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("scheduler")

# Graceful shutdown on Ctrl+C or SIGTERM
_running = True

def _handle_signal(sig, frame):
    global _running
    logger.info(f"Signal {sig} received — shutting down...")
    _running = False

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def run_forever(interval_minutes: int = SCHEDULER_INTERVAL_MINUTES):
    """Runs pipeline on a fixed interval until interrupted."""
    logger.info(f"Scheduler started — interval: {interval_minutes} min")
    logger.info("Press Ctrl+C to stop.")

    while _running:
        try:
            logger.info(f"Starting pipeline cycle...")
            result = run_pipeline()
            logger.info(f"Cycle done: {result}")
        except Exception as e:
            logger.error(f"Pipeline error: {e}", exc_info=True)

        if not _running:
            break

        interval_sec = interval_minutes * 60
        logger.info(f"Sleeping {interval_minutes} min until next cycle...")

        # Sleep in 5-second chunks so Ctrl+C wakes up quickly
        for _ in range(interval_sec // 5):
            if not _running:
                break
            time.sleep(5)

    logger.info("Scheduler stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disaster Eye Scheduler")
    parser.add_argument("--interval", type=int, default=SCHEDULER_INTERVAL_MINUTES,
                        help="Minutes between pipeline runs")
    parser.add_argument("--once", action="store_true",
                        help="Run pipeline once and exit")
    args = parser.parse_args()

    if args.once:
        logger.info("Running pipeline once...")
        result = run_pipeline()
        logger.info(f"Done: {result}")
    else:
        run_forever(interval_minutes=args.interval)
