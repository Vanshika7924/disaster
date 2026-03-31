# =============================================================================
# config.py
# All project settings in one file.
# Every other file imports from here — never hardcode values anywhere else.
# =============================================================================

import os
from dotenv import load_dotenv

load_dotenv()   # reads .env file automatically

# ─────────────────────────────────────────────────────────────────────────────
# MongoDB
# ─────────────────────────────────────────────────────────────────────────────
MONGO_URI       = os.getenv("MONGO_URI",       "mongodb://localhost:27017/")
DB_NAME         = os.getenv("DB_NAME",         "disaster_eye")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "alerts")

# ─────────────────────────────────────────────────────────────────────────────
# BERT model
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DIR   = os.getenv("MODEL_DIR",   "models/bert_disaster")   # trained model path
BERT_BASE   = "bert-base-uncased"                                 # base model
MAX_LEN     = 128                                                  # max token length

# ─────────────────────────────────────────────────────────────────────────────
# Label definitions  (ORDER MUST MATCH TRAINING — never change)
# ─────────────────────────────────────────────────────────────────────────────
LABEL_LIST = [
    "non-disaster",   # 0
    "earthquake",     # 1
    "flood",          # 2
    "cyclone",        # 3
    "fire",           # 4
    "landslide",      # 5
    "explosion",      # 6
    "tsunami",        # 7
    "drought",        # 8
    "accident",       # 9
    "collapse",       # 10
]

LABEL2ID       = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL       = {i: label for i, label in enumerate(LABEL_LIST)}
NUM_LABELS     = len(LABEL_LIST)
DISASTER_LABELS = set(LABEL_LIST) - {"non-disaster"}

# ─────────────────────────────────────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────────────────────────────────────
SCHEDULER_INTERVAL_MINUTES = int(os.getenv("SCHEDULER_INTERVAL_MINUTES", 30))

# ─────────────────────────────────────────────────────────────────────────────
# RSS feeds — India-specific Google News
# ─────────────────────────────────────────────────────────────────────────────
RSS_FEEDS = [
    "https://news.google.com/rss/search?q=earthquake+India+when:1d&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=flood+India+when:1d&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=cyclone+India+when:1d&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=landslide+India+when:1d&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=forest+fire+India+when:1d&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=cloudburst+India+when:1d&hl=en-IN&gl=IN&ceid=IN:en",
]
