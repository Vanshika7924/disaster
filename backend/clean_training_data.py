# =============================================================================
# clean_training_data.py
# Cleans dirty disaster training CSV for Disaster Eye
# FINAL 6-LABEL VERSION
#
# Usage:
#   python clean_training_data.py --input data/training_data_clean.csv --output data/training_data_final.csv
# =============================================================================

import re
import argparse
import pandas as pd


# -----------------------------------------------------------------------------
# FINAL LABELS FOR YOUR PROJECT
# -----------------------------------------------------------------------------
FINAL_LABELS = {
    "non-disaster",
    "earthquake",
    "flood",
    "cyclone",
    "forest fire",
    "landslide",
}


# -----------------------------------------------------------------------------
# Text cleaning
# -----------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.replace("\u2019", "'").replace("`", "'")
    text = re.sub(r"[^a-z0-9\s\-.,:/']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------------------------------------------------------
# Normalize labels
# -----------------------------------------------------------------------------
def normalize_label(label: str) -> str:
    label = str(label).strip().lower()

    if label in FINAL_LABELS:
        return label

    alias_map = {
        "earthquake, tremors": "earthquake",
        "tremor": "earthquake",
        "tremors": "earthquake",
        "quake": "earthquake",

        "floods": "flood",
        "flash flood": "flood",
        "flash floods": "flood",
        "rain": "flood",
        "rains": "flood",
        "rainfall": "flood",
        "inundation": "flood",

        "cyclonic": "cyclone",
        "storm": "cyclone",

        "wildfire": "forest fire",
        "wild fire": "forest fire",
        "fire": "non-disaster",

        "landslides": "landslide",

        "cloudburst": "non-disaster",
        "avalanche": "non-disaster",
        "hailstorm": "non-disaster",
        "collapse": "non-disaster",
        "collapses": "non-disaster",
        "collapsed": "non-disaster",
        "accident": "non-disaster",
        "crash": "non-disaster",
        "derailment": "non-disaster",
        "explosion": "non-disaster",
        "blaze": "non-disaster",
        "inferno": "non-disaster",
        "drought": "non-disaster",
        "tsunami": "non-disaster",
    }

    if "," in label:
        parts = [normalize_label(x.strip()) for x in label.split(",")]
        parts = [p for p in parts if p != "non-disaster"]
        if len(parts) == 0:
            return "non-disaster"
        return parts[0]

    return alias_map.get(label, "non-disaster")


# -----------------------------------------------------------------------------
# False-positive detector
# -----------------------------------------------------------------------------
REJECT_PATTERNS = [
    r"\bpolitic",
    r"\belection",
    r"\bpoll",
    r"\bwrangle",
    r"\bdebate",
    r"\bexplained\b",
    r"\bexplainer\b",
    r"\bwhat it means\b",
    r"\bhow .* was named\b",
    r"\bwhy is\b",
    r"\breport\b",
    r"\banalysis\b",
    r"\breview\b",
    r"\bopinion\b",
    r"\bhistory\b",
    r"\bhistoric\b",
    r"\bdeadliest\b",
    r"\bcauses\b",
    r"\btypes\b",
    r"\bchallenges\b",
    r"\bzones\b",

    r"\bmapping\b",
    r"\bsusceptibility\b",
    r"\bcomparison of different techniques\b",
    r"\bforecasting system\b",
    r"\bpreparedness\b",
    r"\bawareness\b",
    r"\bdrill\b",
    r"\btraining\b",
    r"\bworkshop\b",
    r"\bseminar\b",
    r"\bconference\b",
    r"\bpolicy\b",

    r"\bmemes?\b",
    r"\binternet\b",
    r"\bmarket\b",
    r"\bexports\b",
    r"\bmovie\b",
    r"\bfilm\b",
    r"\bbook\b",
    r"\bseries\b",
    r"\bshow\b",
    r"\bvideo game\b",

    r"\btrain accident\b",
    r"\btrain crash\b",
    r"\bbus fire\b",
    r"\bbuilding collapse\b",
    r"\broof collapse\b",
    r"\bwall collapse\b",
    r"\bshop fire\b",
    r"\bhospital fire\b",
    r"\bnightclub fire\b",
    r"\bfurniture shop fire\b",
    r"\bhighway fire\b",
    r"\bexpressway\b",
    r"\bderail",
    r"\bcollision\b",

    r"\b\d{4}\b",
    r"\b20 years\b",
    r"\b10 years\b",
    r"\blast year\b",
    r"\byears ago\b",
    r"\banniversary\b",
    r"\bremembering\b",
    r"\bretrospective\b",
]


CURRENT_EVENT_SIGNALS = [
    "hits", "hit", "strikes", "struck", "kills", "killed", "injures", "injured",
    "missing", "evacuated", "rescued", "dead", "death", "alert", "warning",
    "heavy rain", "flooded", "submerged", "washed away", "stranded",
    "landfall", "villages affected", "ndrf", "sdrf", "rescue operation",
    "tremors felt", "magnitude", "jolts", "quake", "earthquake", "landslide",
    "cyclone", "forest fire", "wildfire", "flash flood", "batters", "breaches danger mark"
]


def looks_like_false_positive(text: str) -> bool:
    text = clean_text(text)

    for pattern in REJECT_PATTERNS:
        if re.search(pattern, text):
            if not any(sig in text for sig in CURRENT_EVENT_SIGNALS):
                return True

    misleading_phrases = [
        "flood internet",
        "floods market",
        "landslide victory",
        "in landslide",
        "political wrangle",
        "what it means",
        "how cyclone was named",
        "issue advisory",
        "forecast",
        "outlook",
        "science behind",
        "support to",
        "humanitarian mission",
        "offers condolences",
        "appeals for help",
    ]

    for phrase in misleading_phrases:
        if phrase in text:
            return True

    return False


# -----------------------------------------------------------------------------
# Decide final label
# -----------------------------------------------------------------------------
def assign_final_label(text: str, raw_label: str) -> str:
    text = clean_text(text)
    label = normalize_label(raw_label)

    if looks_like_false_positive(text):
        return "non-disaster"

    if label == "earthquake":
        if any(x in text for x in ["earthquake", "quake", "tremor", "tremors", "magnitude", "jolts", "strikes"]):
            return "earthquake"
        return "non-disaster"

    if label == "flood":
        if any(x in text for x in ["flood", "floods", "flash flood", "flooded", "submerged", "waterlogging", "breaches danger mark", "washed away"]):
            return "flood"
        return "non-disaster"

    if label == "cyclone":
        if any(x in text for x in ["cyclone", "landfall", "storm", "severe cyclonic storm"]):
            return "cyclone"
        return "non-disaster"

    if label == "forest fire":
        if any(x in text for x in ["forest fire", "wildfire", "wild fire"]):
            return "forest fire"
        return "non-disaster"

    if label == "landslide":
        if any(x in text for x in ["landslide", "landslides", "mudslide"]):
            return "landslide"
        return "non-disaster"

    return "non-disaster"


# -----------------------------------------------------------------------------
# Main cleaner
# -----------------------------------------------------------------------------
def clean_dataset(input_csv: str, output_csv: str):
    try:
        df = pd.read_csv(input_csv, encoding="utf-8")
    except Exception:
        df = pd.read_csv(input_csv, encoding="latin1")

    if "DISASTER_TYPE" in df.columns and "label" not in df.columns:
        df["label"] = df["DISASTER_TYPE"]

    if "text" not in df.columns:
        if "title" in df.columns:
            df["text"] = df["title"]
        else:
            raise ValueError("CSV must contain 'text' or 'title' column")

    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].fillna("non-disaster").astype(str)

    df["clean_text"] = df["text"].apply(clean_text)
    df["final_label"] = df.apply(
        lambda row: assign_final_label(row["clean_text"], row["label"]),
        axis=1
    )

    out = pd.DataFrame({
        "text": df["clean_text"],
        "label": df["final_label"]
    })

    out = out[out["text"].str.len() > 5]
    out = out.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    out.to_csv(output_csv, index=False)

    print("\n✅ CLEANING COMPLETE")
    print(f"Input rows : {len(df)}")
    print(f"Output rows: {len(out)}")
    print("\nFinal label distribution:")
    print(out["label"].value_counts())


# -----------------------------------------------------------------------------
# Entry
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/training_data_clean.csv", help="Input raw CSV")
    parser.add_argument("--output", default="data/training_data_final.csv", help="Output cleaned CSV")
    args = parser.parse_args()

    clean_dataset(args.input, args.output)