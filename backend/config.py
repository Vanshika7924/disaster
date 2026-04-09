# =============================================================================
# config.py
# Central configuration file for Disaster Eye
# FINAL CLEAN VERSION FOR RETRAINING
# =============================================================================

import os
from pathlib import Path
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# FORCE LOAD .env FROM PROJECT ROOT
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

# Load .env file explicitly
loaded = load_dotenv(dotenv_path=ENV_PATH)

print("DEBUG: ENV PATH =", ENV_PATH)
print("DEBUG: .env loaded =", loaded)
print("DEBUG: MONGO_URI from env =", os.getenv("MONGO_URI"))

# -----------------------------------------------------------------------------
# MongoDB
# -----------------------------------------------------------------------------
MONGO_URI       = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME         = os.getenv("DB_NAME", "disaster_eye")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "alerts")

# -----------------------------------------------------------------------------
# BERT Model
# -----------------------------------------------------------------------------
MODEL_DIR = os.getenv("MODEL_DIR", "models/bert_disaster")
BERT_BASE = os.getenv("BERT_BASE", "bert-base-uncased")
MAX_LEN   = int(os.getenv("MAX_LEN", 128))

# -----------------------------------------------------------------------------
# Label Definitions (FINAL WORKING LABEL SET)
# -----------------------------------------------------------------------------
LABEL_LIST = [
    "non-disaster",
    "earthquake",
    "flood",
    "cyclone",
    "forest fire",
    "landslide",
]

LABEL2ID        = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL        = {i: label for i, label in enumerate(LABEL_LIST)}
NUM_LABELS      = len(LABEL_LIST)
DISASTER_LABELS = set(LABEL_LIST) - {"non-disaster"}

# -----------------------------------------------------------------------------
# Scheduler
# -----------------------------------------------------------------------------
SCHEDULER_INTERVAL_MINUTES = int(os.getenv("SCHEDULER_INTERVAL_MINUTES", 2))

# -----------------------------------------------------------------------------
# File Paths
# -----------------------------------------------------------------------------
DATA_DIR        = os.getenv("DATA_DIR", "data")
RAW_RSS_FILE    = os.path.join(DATA_DIR, "disaster_news.csv")
MASTER_CSV_FILE = os.path.join(DATA_DIR, "classified_unlabelled_data_master.csv")
LOG_FILE        = os.getenv("LOG_FILE", "scheduler.log")

# -----------------------------------------------------------------------------
# RSS Feeds (CLEANED)
# -----------------------------------------------------------------------------
RSS_FEEDS = [
    "https://news.google.com/rss/search?q=earthquake+India+when:1d&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=flood+India+when:1d&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=cyclone+India+when:1d&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=landslide+India+when:1d&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=forest+fire+OR+wildfire+India+when:1d&hl=en-IN&gl=IN&ceid=IN:en",
    "https://news.google.com/rss/search?q=flash+flood+India+when:1d&hl=en-IN&gl=IN&ceid=IN:en",
]

# -----------------------------------------------------------------------------
# Runtime settings
# -----------------------------------------------------------------------------
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", 0.50))

PROJECT_NAME = "Disaster Eye"
COUNTRY_NAME = "India"