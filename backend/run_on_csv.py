# =============================================================================
# run_on_csv.py
# Run Disaster Eye on ANY CSV file using trained BERT model
# Keeps only:
#   - disaster = True
#   - location exists
#   - removes generic fire, keeps only forest_fire
# =============================================================================

import os
import argparse
import logging
from datetime import datetime

import pandas as pd

from classifier import get_classifier
from ner_model import extract_location
from time_extractor import extract_disaster_time


# =============================================================================
# LOGGER
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("run_on_csv")


# =============================================================================
# HELPERS
# =============================================================================

def safe_read_csv(path: str) -> pd.DataFrame:
    encodings = ["utf-8", "latin1", "cp1252", "utf-8-sig"]

    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            logger.info(f"CSV loaded with encoding: {enc}")
            return df
        except Exception:
            continue

    raise RuntimeError(f"❌ Could not read CSV file: {path}")


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["headline", "summary", "text", "link", "published_time"]

    for col in required_cols:
        if col not in df.columns:
            df[col] = ""

    return df


def build_full_text(row) -> str:
    parts = [
        str(row.get("headline", "")).strip(),
        str(row.get("summary", "")).strip(),
        str(row.get("text", "")).strip(),
    ]
    return " ".join([p for p in parts if p]).strip()


def is_valid_location(loc) -> bool:
    if pd.isna(loc):
        return False

    loc = str(loc).strip().lower()
    invalid_values = ["", "nan", "none", "null", "undefined", "unknown"]

    return loc not in invalid_values


def is_valid_disaster_label(label: str) -> bool:
    """
    Keep only allowed disaster labels.
    Generic 'fire' is removed.
    Only forest_fire is allowed.
    """
    if not label:
        return False

    label = str(label).strip().lower()

    allowed_labels = {
        "earthquake",
        "flood",
        "cyclone",
        "landslide",
        "forest_fire",   # keep this
        "drought",
        "tsunami",
        "storm",
        "heatwave",
        "coldwave",
        "avalanche",
        "cloudburst"
    }

    return label in allowed_labels


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_on_csv(input_path: str, output_path: str):
    logger.info("=" * 70)
    logger.info("🚀 Running Disaster Eye on CSV")
    logger.info("=" * 70)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"❌ File not found: {input_path}")

    logger.info(f"📥 Input: {input_path}")

    # -------------------------------------------------------------------------
    # 1) Load CSV
    # -------------------------------------------------------------------------
    df = safe_read_csv(input_path)

    if df.empty:
        logger.warning("⚠ CSV is empty")
        return

    logger.info(f"Rows loaded: {len(df)}")

    # -------------------------------------------------------------------------
    # 2) Prepare data
    # -------------------------------------------------------------------------
    df = ensure_columns(df)
    df["full_text"] = df.apply(build_full_text, axis=1)
    df = df[df["full_text"].str.strip() != ""].copy()

    if df.empty:
        logger.warning("⚠ No usable text found")
        return

    logger.info(f"Rows after cleaning: {len(df)}")

    # -------------------------------------------------------------------------
    # 3) Load model
    # -------------------------------------------------------------------------
    logger.info("🤖 Loading BERT model...")
    clf = get_classifier()
    logger.info("✅ Model loaded")

    results = []

    # -------------------------------------------------------------------------
    # 4) Process rows
    # -------------------------------------------------------------------------
    for idx, row in df.iterrows():
        text = row["full_text"]

        try:
            # -----------------------------
            # BERT classification
            # -----------------------------
            label, confidence = clf.predict(text)
            label = str(label).strip().lower()

            # -----------------------------
            # Keep ONLY allowed disaster labels
            # -----------------------------
            if not is_valid_disaster_label(label):
                continue

            # -----------------------------
            # Location extraction
            # -----------------------------
            location, state, country = extract_location(text)

            # Remove rows with invalid location
            if not is_valid_location(location):
                continue

            # -----------------------------
            # Time extraction
            # -----------------------------
            disaster_time, time_source = extract_disaster_time(text)

            results.append({
                "headline": row.get("headline", ""),
                "summary": row.get("summary", ""),
                "text": row.get("text", ""),
                "link": row.get("link", ""),
                "published_time": row.get("published_time", ""),

                "disaster_type": label,
                "confidence": round(float(confidence), 4),

                "location": location,
                "state": state,
                "country": country,

                "disaster_time": disaster_time,
                "time_source": time_source,

                "is_disaster": True,
                "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })

            if len(results) % 20 == 0:
                logger.info(f"Processed {len(results)} valid disaster rows...")

        except Exception as e:
            logger.error(f"Error at row {idx}: {e}")

    # -------------------------------------------------------------------------
    # 5) Save output
    # -------------------------------------------------------------------------
    if not results:
        logger.warning("⚠ No valid disaster rows found")
        return

    final_df = pd.DataFrame(results)

    # Final safety filtering
    final_df = final_df[final_df["location"].apply(is_valid_location)]
    final_df = final_df[final_df["disaster_type"].apply(is_valid_disaster_label)]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    logger.info("=" * 70)
    logger.info(f"✅ Output saved: {output_path}")
    logger.info(f"Final valid disaster rows: {len(final_df)}")
    logger.info("=" * 70)

    print("\n📊 FILTERED OUTPUT PREVIEW:\n")
    preview_cols = ["headline", "disaster_type", "confidence", "location"]
    print(final_df[preview_cols].head(10).to_string(index=False))


# =============================================================================
# CLI ENTRY
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Disaster Eye on CSV")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input CSV"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/final_disaster_alerts.csv",
        help="Path to save filtered output CSV"
    )

    args = parser.parse_args()

    run_on_csv(
        input_path=args.input,
        output_path=args.output
    )