# =============================================================================
# classifier.py
# Loads your trained BERT model and predicts disaster type for any text.
#
# Usage:
#   from classifier import get_classifier
#   clf = get_classifier()
#   label, confidence = clf.predict("Earthquake hits Manipur")
# =============================================================================

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import MODEL_DIR, MAX_LEN, ID2LABEL, LABEL2ID, NUM_LABELS, DISASTER_LABELS

# Singleton — model loads once, reused for every request
_classifier_instance = None


class DisasterClassifier:
    """
    Wraps the trained BERT model.
    Loaded once at startup, then reused for all predictions.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Classifier] Loading model from: {MODEL_DIR}")
        print(f"[Classifier] Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR,
            num_labels=NUM_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )
        self.model.to(self.device)
        self.model.eval()
        print("[Classifier] Ready")

    # ─────────────────────────────────────────────────────────────────────────
    # Single prediction
    # ─────────────────────────────────────────────────────────────────────────
    def predict(self, text: str) -> tuple:
        """
        Predict disaster type for ONE text.

        Returns:
            (label_name: str, confidence: float)
            e.g. ("earthquake", 0.94) or ("non-disaster", 0.87)
        """
        if not text or not str(text).strip():
            return "non-disaster", 1.0

        enc = self.tokenizer(
            str(text).strip(),
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            probs = F.softmax(self.model(**enc).logits, dim=-1)[0]

        best_idx   = int(torch.argmax(probs).item())
        label      = ID2LABEL[best_idx]
        confidence = round(float(probs[best_idx].item()), 4)
        return label, confidence

    # ─────────────────────────────────────────────────────────────────────────
    # Batch prediction (much faster than looping predict())
    # ─────────────────────────────────────────────────────────────────────────
    def predict_batch(self, texts: list, batch_size: int = 32) -> list:
        """
        Predict disaster type for a LIST of texts.
        Returns list of (label_name, confidence) tuples.
        """
        results = []
        for i in range(0, len(texts), batch_size):
            chunk = [str(t) for t in texts[i: i + batch_size]]
            enc = self.tokenizer(
                chunk,
                max_length=MAX_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                all_probs = F.softmax(self.model(**enc).logits, dim=-1)
            for probs in all_probs:
                best_idx = int(torch.argmax(probs).item())
                results.append((
                    ID2LABEL[best_idx],
                    round(float(probs[best_idx].item()), 4),
                ))
        return results

    def is_disaster(self, label: str) -> bool:
        return label in DISASTER_LABELS


# ─────────────────────────────────────────────────────────────────────────────
# Singleton getter — always use this, never instantiate directly
# ─────────────────────────────────────────────────────────────────────────────
def get_classifier() -> DisasterClassifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = DisasterClassifier()
    return _classifier_instance
