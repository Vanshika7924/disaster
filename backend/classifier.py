# =============================================================================
# classifier.py
# FINAL CLEAN VERSION
# ONLY predicts disaster label + confidence
# =============================================================================

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import MODEL_DIR, MAX_LEN, ID2LABEL, LABEL2ID, NUM_LABELS

# Singleton — model loads once, reused for every request
_classifier_instance = None


class DisasterClassifier:
    """
    Pure BERT classifier.
    No heavy rule-based filtering here.
    Filtering will happen in pipeline.py
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
        )

        self.model.to(self.device)
        self.model.eval()
        print("[Classifier] Ready")

    # -------------------------------------------------------------------------
    # Single prediction
    # -------------------------------------------------------------------------
    def predict(self, text: str):
        if not text or not str(text).strip():
            return "non-disaster", 1.0

        text = str(text).strip()

        enc = self.tokenizer(
            text,
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            probs = F.softmax(self.model(**enc).logits, dim=-1)[0]

        best_idx = int(torch.argmax(probs).item())
        label = ID2LABEL[best_idx]
        confidence = round(float(probs[best_idx].item()), 4)

        return label, confidence

    # -------------------------------------------------------------------------
    # Batch prediction
    # -------------------------------------------------------------------------
    def predict_batch(self, texts, batch_size=32):
        results = []

        for i in range(0, len(texts), batch_size):
            chunk = [str(t) if t is not None else "" for t in texts[i:i + batch_size]]

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
                label = ID2LABEL[best_idx]
                confidence = round(float(probs[best_idx].item()), 4)
                results.append((label, confidence))

        return results


# -----------------------------------------------------------------------------
# Singleton getter
# -----------------------------------------------------------------------------
def get_classifier():
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = DisasterClassifier()
    return _classifier_instance