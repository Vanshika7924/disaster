# =============================================================================
# train_bert.py
# Trains BERT on your labeled disaster dataset.
#
# Usage:
#   python train_bert.py
#   python train_bert.py --csv data/merged_training_data.csv   (retraining)
#
# Input : CSV with columns: text, label
# Output: models/bert_disaster/  (model + tokenizer)
# =============================================================================

import os
import time
import argparse
import logging

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report

from config import (
    BERT_BASE, MODEL_DIR, MAX_LEN,
    LABEL2ID, ID2LABEL, NUM_LABELS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE    = 16
EPOCHS        = 4
LEARNING_RATE = 2e-5
TEST_SPLIT    = 0.15
RANDOM_SEED   = 42


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class DisasterDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts     = texts
        self.labels    = labels
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
            "input_ids"     : enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels"        : torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Load & clean training data
# ─────────────────────────────────────────────────────────────────────────────
def load_data(csv_path: str):
    """
    Loads CSV with 'text' and 'label' columns.
    Accepts both original labeled_text_dataset.csv (with DISASTER_TYPE)
    and the clean training_data_clean.csv format.
    """
    df = pd.read_csv(csv_path, encoding="latin1")

    # If original format with DISASTER_TYPE column — normalize it
    if "DISASTER_TYPE" in df.columns and "label" not in df.columns:
        from config import DISASTER_NORMALIZE  # optional mapping
        df["label"] = df["DISASTER_TYPE"].apply(
            lambda x: str(x).strip().lower() if pd.notna(x) else "non-disaster"
        )

    # Use title as text if text column is empty
    if "text" not in df.columns or df["text"].isna().all():
        if "title" in df.columns:
            df["text"] = df["title"].astype(str).str.strip()

    df = df[df["text"].notna() & df["label"].notna()].copy()
    df["text"]  = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    # Keep only valid labels
    valid = set(LABEL2ID.keys())
    df    = df[df["label"].isin(valid)]
    df    = df[df["text"].str.len() > 5].reset_index(drop=True)

    logger.info(f"Loaded {len(df)} training examples")
    logger.info(f"Label distribution:\n{df['label'].value_counts().to_string()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────
def train(csv_path: str = "data/training_data_clean.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load data
    df     = load_data(csv_path)
    texts  = df["text"].tolist()
    labels = [LABEL2ID.get(l, 0) for l in df["label"].tolist()]

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels,
        test_size=TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=labels,
    )
    logger.info(f"Train: {len(X_train)} | Val: {len(X_val)}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BERT_BASE)

    # Datasets & loaders
    train_ds = DisasterDataset(X_train, y_train, tokenizer)
    val_ds   = DisasterDataset(X_val,   y_val,   tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_BASE,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    ).to(device)

    # Optimizer + scheduler
    optimizer   = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps,
    )

    best_f1  = 0.0
    history  = []

    logger.info(f"Training for {EPOCHS} epochs...")
    logger.info("=" * 55)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        # Train
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
            if (step + 1) % 10 == 0:
                logger.info(f"  Epoch {epoch} | step {step+1}/{len(train_loader)} | loss {total_loss/(step+1):.4f}")

        # Validate
        model.eval()
        preds_all, true_all = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                preds_all.extend(torch.argmax(logits, -1).cpu().numpy())
                true_all.extend(batch["labels"].cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        val_f1   = f1_score(true_all, preds_all, average="weighted", zero_division=0)
        val_acc  = accuracy_score(true_all, preds_all)

        logger.info(f"Epoch {epoch} | loss={avg_loss:.4f} | val_f1={val_f1:.4f} | val_acc={val_acc:.4f} | {time.time()-t0:.1f}s")
        history.append({"epoch": epoch, "loss": avg_loss, "val_f1": val_f1, "val_acc": val_acc})

        if val_f1 > best_f1:
            best_f1 = val_f1
            os.makedirs(MODEL_DIR, exist_ok=True)
            model.save_pretrained(MODEL_DIR)
            tokenizer.save_pretrained(MODEL_DIR)
            logger.info(f"  >>> Best model saved (F1={best_f1:.4f}) → {MODEL_DIR}/")

    # Final report
    present = sorted(set(true_all))
    names   = [ID2LABEL[i] for i in present]
    logger.info("\nClassification report:")
    print(classification_report(true_all, preds_all, labels=present, target_names=names, zero_division=0))

    os.makedirs("data", exist_ok=True)
    pd.DataFrame(history).to_csv("data/training_history.csv", index=False)
    logger.info(f"\nDone! Best F1 = {best_f1:.4f} | Model saved to {MODEL_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/training_data_clean.csv",
                        help="Path to training CSV")
    args = parser.parse_args()
    os.makedirs("data", exist_ok=True)
    train(csv_path=args.csv)
