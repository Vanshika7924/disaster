import pandas as pd
from data_fetcher import fetch_rss_news
from classifier import get_classifier

# Fetch RSS news
articles = fetch_rss_news()

# Load classifier
clf = get_classifier()

rows = []

for article in articles:
    text = article.get("text", "").strip()

    if not text:
        continue

    label, confidence = clf.predict(text)

    rows.append({
        "text": text,
        "label": label,
        "confidence": confidence,
        "title": article.get("title", ""),
        "summary": article.get("summary", ""),
        "link": article.get("link", ""),
        "published": article.get("published", "")
    })

# Save full labeled dataset
df = pd.DataFrame(rows)
df.to_csv("data/auto_labeled_rss_data.csv", index=False, encoding="utf-8")

# Save training-ready 2-column file
train_df = df[["text", "label"]].copy()
train_df.to_csv("data/training_data_from_rss.csv", index=False, encoding="utf-8")

print("Done!")
print("Saved:")
print("1) data/auto_labeled_rss_data.csv")
print("2) data/training_data_from_rss.csv")
print(f"Rows: {len(df)}")