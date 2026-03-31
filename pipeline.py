# =============================================================================
# pipeline.py — Final Version
# New in this version:
#   - is_real_disaster_news() filter added (removes workshops, seminars etc.)
#   - geopy used in ner_model for automatic city→state resolution
# =============================================================================

import logging
import pandas as pd
from datetime import datetime, timezone

from data_fetcher    import fetch_rss_news
from classifier      import get_classifier
from ner_model       import extract_location, is_india_news, is_real_disaster_news
from time_extractor  import extract_disaster_time
from db              import get_db
from config          import DISASTER_LABELS

logger = logging.getLogger(__name__)


def run_pipeline(rss_items: list = None, save_csv: bool = True) -> dict:
    logger.info("=" * 55)
    logger.info("Disaster Eye Pipeline — starting")
    logger.info("=" * 55)

    summary = {
        "fetched": 0, "bert_disaster": 0, "india_filter": 0,
        "real_disaster_filter": 0, "location_filter": 0,
        "inserted": 0, "duplicates": 0
    }

    # ── STEP 1: Fetch RSS ────────────────────────────────────────────────────
    if rss_items is None:
        logger.info("Step 1: Fetching RSS news...")
        rss_items = fetch_rss_news()

    summary["fetched"] = len(rss_items)
    logger.info(f"Step 1 done — {len(rss_items)} articles")

    if not rss_items:
        return summary

    df = pd.DataFrame(rss_items)

    # ── STEP 2: BERT predictions ─────────────────────────────────────────────
    logger.info("Step 2: Running BERT predictions...")
    clf     = get_classifier()
    results = clf.predict_batch(df["text"].tolist())

    df["disaster_type"] = [r[0] for r in results]
    df["confidence"]    = [r[1] for r in results]
    df["is_disaster"]   = df["disaster_type"].isin(DISASTER_LABELS)

    df_dis = df[df["is_disaster"]].copy().reset_index(drop=True)
    summary["bert_disaster"] = len(df_dis)
    logger.info(f"Step 2 done — {len(df_dis)}/{len(df)} classified as disaster")

    if df_dis.empty:
        return summary

    # ── STEP 3: Filter to India news ─────────────────────────────────────────
    logger.info("Step 3: Filtering foreign country news...")
    df_dis["is_india"] = df_dis["text"].apply(is_india_news)
    df_india = df_dis[df_dis["is_india"]].copy().reset_index(drop=True)
    summary["india_filter"] = len(df_india)
    logger.info(f"Step 3 done — {len(df_india)}/{len(df_dis)} are India news")

    if df_india.empty:
        return summary

    # ── STEP 4: is_real_disaster_news() filter ───────────────────────────────
    # Removes: workshops, seminars, election "landslide", meme floods, etc.
    logger.info("Step 4: Filtering non-real disaster news (workshops, seminars, etc.)...")
    df_india["is_real"] = df_india.apply(
        lambda row: is_real_disaster_news(row["title"], row["confidence"]),
        axis=1
    )
    before = len(df_india)
    df_india = df_india[df_india["is_real"]].copy().reset_index(drop=True)
    removed_fake = before - len(df_india)
    summary["real_disaster_filter"] = len(df_india)
    logger.info(f"Step 4 done — {removed_fake} non-real removed | {len(df_india)} remaining")

    if df_india.empty:
        return summary

    # ── STEP 5: Location extraction ───────────────────────────────────────────
    logger.info("Step 5: Extracting locations...")
    loc_results          = [extract_location(t) for t in df_india["text"]]
    df_india["location"] = [r[0] for r in loc_results]
    df_india["state"]    = [r[1] for r in loc_results]
    df_india["country"]  = [r[2] if r[2] else "India" for r in loc_results]
    loc_found = df_india["location"].notna().sum()
    logger.info(f"Step 5 done — location found for {loc_found}/{len(df_india)}")

    # ── STEP 5b: Remove rows where BOTH location AND state are null ───────────
    before = len(df_india)
    df_india = df_india[
        df_india["location"].notna() | df_india["state"].notna()
    ].copy().reset_index(drop=True)
    removed_no_loc = before - len(df_india)
    summary["location_filter"] = len(df_india)
    logger.info(f"Step 5b done — {removed_no_loc} removed (no location+state) | {len(df_india)} remaining")

    if df_india.empty:
        return summary

    # ── STEP 6: Time extraction ───────────────────────────────────────────────
    logger.info("Step 6: Extracting disaster times...")
    time_results = [
        extract_disaster_time(row["text"], row.get("published"))
        for _, row in df_india.iterrows()
    ]
    df_india["disaster_time"] = [r[0] for r in time_results]
    df_india["time_source"]   = [r[1] for r in time_results]

    # ── STEP 7: Build final records ───────────────────────────────────────────
    logger.info("Step 7: Building final records...")
    now = datetime.now(timezone.utc).isoformat()

    records = []
    for _, row in df_india.iterrows():
        records.append({
            "headline"      : row.get("title", ""),
            "summary"       : row.get("summary", ""),
            "link"          : row.get("link", ""),
            "published_time": row.get("published", ""),
            "disaster_time" : row.get("disaster_time"),
            "time_source"   : row.get("time_source"),
            "disaster_type" : row.get("disaster_type"),
            "confidence"    : float(row.get("confidence", 0.0)),
            "location"      : row.get("location"),
            "state"         : row.get("state"),
            "country"       : row.get("country", "India"),
            "is_disaster"   : True,
            "processed_at"  : now,
        })

    if save_csv and records:
        import os
        os.makedirs("data", exist_ok=True)
        pd.DataFrame(records).to_csv("data/cleaned_disaster_data.csv", index=False)
        logger.info(f"Saved CSV ({len(records)} rows)")

    # ── STEP 8: MongoDB ───────────────────────────────────────────────────────
    logger.info("Step 8: Inserting into MongoDB...")
    db_result = get_db().insert_alerts(records)
    summary["inserted"]   = db_result["inserted"]
    summary["duplicates"] = db_result["duplicates"]

    logger.info("=" * 55)
    logger.info(f"Pipeline complete: {summary}")
    logger.info("=" * 55)
    return summary
