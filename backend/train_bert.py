# =============================================================================
# train_bert.py
# FINAL CLEAN TRAINING FILE FOR DISASTER EYE
# FINAL 6-LABEL VERSION
#
# Usage:
#   python train_bert.py --csv data/training_data_final.csv
# =============================================================================

import os
import re
import time
import random
import argparse
import logging

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from config import (
    BERT_BASE, MODEL_DIR, MAX_LEN,
    LABEL2ID, ID2LABEL, NUM_LABELS,
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")

# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
BATCH_SIZE    = 16
EPOCHS        = 3
LEARNING_RATE = 2e-5
TEST_SPLIT    = 0.15
RANDOM_SEED   = 42


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Text Cleaning
# -----------------------------------------------------------------------------
def clean_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.replace("\u2019", "'").replace("`", "'")
    text = re.sub(r"[^a-z0-9\s\-.,:/']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------------------------------------------------------
# Hard Negative Rules
# -----------------------------------------------------------------------------
REJECT_KEYWORDS = {
    "political", "politics", "election", "poll", "polls", "wrangle",
    "debate", "blame", "speech", "campaign", "parliament", "assembly",
    "manifesto", "vote", "voting", "minister said", "opposition",
    "anniversary", "commemoration", "remembering", "tribute",
    "report", "review", "analysis", "editorial", "opinion",
    "discussion", "documentary", "history", "historic", "retrospective",
    "movie", "film", "song", "drama", "series", "book",
    "workshop", "seminar", "conference", "summit", "preparedness",
    "mock drill", "training", "awareness", "preparedness drill",
    "relief package", "compensation", "aid", "humanitarian",
    "visits", "visited", "visit", "tourism", "travel advisory",
    "impact on travel", "weather update", "forecast", "monsoon outlook",
    "memories of", "lessons from", "aftermath of",
    "goes missing", "missing student", "trekker missing", "hiker missing",
    "rescue team searches", "student missing", "tourist missing",
    "factory fire", "shop fire", "warehouse fire", "building fire",
    "gas leak", "boiler blast", "industrial accident",
    "wall collapse", "roof collapse", "bridge collapse", "building collapse",
}

CURRENT_DISASTER_SIGNALS = {
    "hits", "hit", "strikes", "struck", "kills", "killed", "injures",
    "injured", "dead", "death", "missing", "evacuated", "rescued",
    "burning", "burns", "destroyed", "damage",
    "floods", "flooded", "quake", "earthquake", "landslide", "wildfire",
    "forest fire", "cyclone", "storm", "heavy rain",
    "rescue", "alert", "warning", "emergency",
    "submerged", "washed away", "stranded", "trapped", "overflow",
    "landfall", "villages affected", "ndrf", "sdrf",
}

PAST_EVENT_PATTERNS = [
    r"\b(19|20)\d{2}\b",
    r"\b\d+\s+years?\s+ago\b",
    r"\blast year\b",
    r"\bprevious year\b",
    r"\bin\s+201[0-9]\b",
    r"\bin\s+202[0-4]\b"
]

WEAK_CONTEXT_PATTERNS = [
    "turns into",
    "issue of",
    "question over",
    "controversy over",
    "debate on",
    "discussion on",
    "politics of",
    "lessons from",
    "aftermath of",
    "memories of",
    "what it means",
    "impact on travel",
    "means for travel",
]


def looks_like_false_positive(text: str) -> bool:
    text = clean_text(text)

    if any(word in text for word in REJECT_KEYWORDS):
        if not any(sig in text for sig in CURRENT_DISASTER_SIGNALS):
            return True

    if any(re.search(pattern, text) for pattern in PAST_EVENT_PATTERNS):
        if not any(sig in text for sig in CURRENT_DISASTER_SIGNALS):
            return True

    if any(p in text for p in WEAK_CONTEXT_PATTERNS):
        if not any(sig in text for sig in CURRENT_DISASTER_SIGNALS):
            return True

    return False


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class DisasterDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]),
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# -----------------------------------------------------------------------------
# Label normalization
# -----------------------------------------------------------------------------
def normalize_label(label: str) -> str:
    label = str(label).strip().lower()

    if label in LABEL2ID:
        return label

    noisy_to_non = {
        "accident",
        "collapse",
        "explosion",
        "drought",
        "tsunami",
        "fire",
        "cloudburst",
        "avalanche",
    }
    if label in noisy_to_non:
        return "non-disaster"

    alias_map = {
        "forest_fire": "forest fire",
        "wildfire": "forest fire",
        "wild fire": "forest fire",
        "flash flood": "flood",
        "flash_flood": "flood",
        "quake": "earthquake",
    }

    return alias_map.get(label, "non-disaster")


# -----------------------------------------------------------------------------
# Load & clean training data
# -----------------------------------------------------------------------------
def load_data(csv_path: str):
    try:
        df = pd.read_csv(csv_path, encoding="utf-8")
    except Exception:
        df = pd.read_csv(csv_path, encoding="latin1")

    if "DISASTER_TYPE" in df.columns and "label" not in df.columns:
        df["label"] = df["DISASTER_TYPE"].apply(
            lambda x: str(x).strip().lower() if pd.notna(x) else "non-disaster"
        )

    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column or 'DISASTER_TYPE' column.")

    if "text" not in df.columns or df["text"].isna().all():
        if "title" in df.columns:
            df["text"] = df["title"].astype(str).str.strip()
        else:
            raise ValueError("CSV must contain a 'text' column or at least a 'title' column.")

    df = df[df["text"].notna() & df["label"].notna()].copy()

    df["text"] = df["text"].astype(str).apply(clean_text)
    df["label"] = df["label"].astype(str).apply(normalize_label)

    valid = set(LABEL2ID.keys())
    df = df[df["label"].isin(valid)]

    df = df[df["text"].str.len() > 5].reset_index(drop=True)

    changed = 0
    for i in range(len(df)):
        if df.loc[i, "label"] != "non-disaster":
            if looks_like_false_positive(df.loc[i, "text"]):
                df.loc[i, "label"] = "non-disaster"
                changed += 1

    logger.info(f"Hard-negative relabeled rows: {changed}")

    df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    logger.info(f"Loaded {len(df)} training examples")
    logger.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")

    return df


# -----------------------------------------------------------------------------
# Weighted loss
# -----------------------------------------------------------------------------
def compute_loss_with_class_weights(logits, labels, class_weights, device):
    weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    return F.cross_entropy(logits, labels, weight=weights)


# -----------------------------------------------------------------------------
# Main training function
# -----------------------------------------------------------------------------
def train(csv_path: str = "data/training_data_final.csv"):
    set_seed(RANDOM_SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    df = load_data(csv_path)
    texts = df["text"].tolist()
    labels = [LABEL2ID.get(l, 0) for l in df["label"].tolist()]

    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    logger.info(f"Train: {len(X_train)} | Val: {len(X_val)}")

    present_classes = np.unique(y_train)

    raw_weights = compute_class_weight(
        class_weight="balanced",
        classes=present_classes,
        y=np.array(y_train)
    )

    class_weights = np.ones(NUM_LABELS, dtype=np.float32)
    for cls, w in zip(present_classes, raw_weights):
        class_weights[cls] = w

    logger.info(f"Class weights: {class_weights}")

    tokenizer = AutoTokenizer.from_pretrained(BERT_BASE)

    train_ds = DisasterDataset(X_train, y_train, tokenizer)
    val_ds   = DisasterDataset(X_val, y_val, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_BASE,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    best_f1 = 0.0
    history = []

    logger.info(f"Training for {EPOCHS} epochs...")
    logger.info("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            logits = outputs.logits

            loss = compute_loss_with_class_weights(
                logits, batch["labels"], class_weights, device
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            if (step + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch} | step {step+1}/{len(train_loader)} | "
                    f"loss {total_loss/(step+1):.4f}"
                )

        model.eval()
        preds_all, true_all = [], []

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                ).logits

                preds_all.extend(torch.argmax(logits, -1).cpu().numpy())
                true_all.extend(batch["labels"].cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        val_f1   = f1_score(true_all, preds_all, average="weighted", zero_division=0)
        val_acc  = accuracy_score(true_all, preds_all)

        logger.info(
            f"Epoch {epoch} | loss={avg_loss:.4f} | "
            f"val_f1={val_f1:.4f} | val_acc={val_acc:.4f} | "
            f"{time.time()-t0:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "val_f1": val_f1,
            "val_acc": val_acc
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            os.makedirs(MODEL_DIR, exist_ok=True)
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            logger.info(f">>> Best model saved (F1={best_f1:.4f}) → {MODEL_DIR}/")

    present = sorted(set(true_all))
    names = [ID2LABEL[i] for i in present]

    logger.info("\nClassification report:")
    print(classification_report(
        true_all,
        preds_all,
        labels=present,
        target_names=names,
        zero_division=0
    ))

    os.makedirs("data", exist_ok=True)
    pd.DataFrame(history).to_csv("data/training_history.csv", index=False)

    logger.info(f"\nDone! Best F1 = {best_f1:.4f} | Model saved to {MODEL_DIR}/")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default="data/training_data_final.csv",
        help="Path to training CSV"
    )
    args = parser.parse_args()

    os.makedirs("data", exist_ok=True)
    train(csv_path=args.csv)