# =============================================================================
# scheduler.py
# Runs the full Disaster Eye pipeline automatically every N minutes.
#
# Usage:
#   python scheduler.py                  -> runs every 2 minutes (default)
#   python scheduler.py --interval 5    -> runs every 5 minutes
#   python scheduler.py --once          -> run once and exit (for testing)
# =============================================================================

import argparse
import logging
import time
import signal
import sys
from datetime import datetime

from pipeline import run_pipeline
from config import SCHEDULER_INTERVAL_MINUTES

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Graceful shutdown
# -----------------------------------------------------------------------------
_running = True

def _handle_signal(sig, frame):
    global _running
    logger.info(f"🛑 Signal {sig} received — shutting down scheduler...")
    _running = False

signal.signal(signal.SIGINT, _handle_signal)   # Ctrl+C
signal.signal(signal.SIGTERM, _handle_signal)  # system stop


# -----------------------------------------------------------------------------
# Main scheduler loop
# -----------------------------------------------------------------------------
def run_forever(interval_minutes: int = SCHEDULER_INTERVAL_MINUTES):
    """
    Runs the Disaster Eye pipeline repeatedly every N minutes
    until interrupted.
    """
    logger.info("=" * 70)
    logger.info("🚀 Disaster Eye Scheduler Started")
    logger.info(f"⏱ Interval: {interval_minutes} minute(s)")
    logger.info("📂 Output files:")
    logger.info("   - data/disaster_news.csv")
    logger.info("   - data/classified_unlabelled_data_master.csv")
    logger.info("Press Ctrl+C to stop.")
    logger.info("=" * 70)

    print("\n🚀 Disaster Eye Scheduler Started")
    print(f"⏱ Running every {interval_minutes} minute(s)...")
    print("📂 Output files:")
    print("   - data/disaster_news.csv")
    print("   - data/classified_unlabelled_data_master.csv")
    print("Press Ctrl+C to stop.\n")

    cycle = 1

    while _running:
        try:
            run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"🔄 Starting pipeline cycle #{cycle} at {run_time}")
            print(f"\n🔄 Running pipeline cycle #{cycle} at {run_time}...")

            result = run_pipeline()

            logger.info(f"✅ Cycle #{cycle} completed successfully")
            logger.info(f"📊 Result: {result}")
            print(f"✅ Cycle #{cycle} completed successfully")
            print(f"📊 Result: {result}")

        except Exception as e:
            logger.error(f"❌ Pipeline error in cycle #{cycle}: {e}", exc_info=True)
            print(f"❌ Error in cycle #{cycle}: {e}")
            print("⚠ Check scheduler.log for details")

        if not _running:
            break

        interval_sec = interval_minutes * 60
        logger.info(f"⏳ Sleeping {interval_minutes} minute(s) until next cycle...")
        print(f"⏳ Waiting {interval_minutes} minute(s) for next cycle...\n")

        # Sleep in 5-second chunks so Ctrl+C works immediately
        for _ in range(interval_sec // 5):
            if not _running:
                break
            time.sleep(5)

        cycle += 1

    logger.info("🛑 Scheduler stopped.")
    print("\n🛑 Scheduler stopped.")


# -----------------------------------------------------------------------------
# Run from terminal
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Disaster Eye Scheduler")
    parser.add_argument(
        "--interval",
        type=int,
        default=SCHEDULER_INTERVAL_MINUTES,
        help="Minutes between pipeline runs"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run pipeline once and exit"
    )

    args = parser.parse_args()

    if args.once:
        logger.info("🚀 Running pipeline once...")
        print("🚀 Running pipeline once...\n")
        result = run_pipeline()
        logger.info(f"✅ Done: {result}")
        print(f"✅ Done: {result}")
    else:
        run_forever(interval_minutes=args.interval)