# Disaster Eye — Backend

Complete backend for the Disaster Eye Android app.

## Project structure

```
disaster_eye_backend/
├── app.py              ← FastAPI server (main entry point)
├── pipeline.py         ← Full RSS → predict → MongoDB pipeline
├── train_bert.py       ← BERT training script
├── classifier.py       ← Loads trained model, runs predictions
├── ner_model.py        ← Location extraction
├── time_extractor.py   ← Disaster time extraction
├── data_fetcher.py     ← RSS fetching
├── db.py               ← MongoDB operations
├── scheduler.py        ← Auto-runs pipeline every N minutes
├── config.py           ← All settings in one place
├── routers/
│   └── alerts.py       ← All /alerts API endpoints
├── requirements.txt
├── .env.example
├── models/
│   └── bert_disaster/  ← Trained model goes here (copy from Colab)
└── data/
    ├── training_data_clean.csv
    └── cleaned_disaster_data.csv
```

---

## Setup (one time)

```bash
# 1. Clone / download this folder
cd disaster_eye_backend

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 4. Setup environment file
cp .env.example .env
# Edit .env — paste your MongoDB URI

# 5. Copy trained model from Colab
# Download models/bert_disaster/ from Colab and put it here
```

---

## How to run

### Option A — API + Scheduler together (production)

Terminal 1 — start the API:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Terminal 2 — start the scheduler:
```bash
python scheduler.py --interval 30   # fetches every 30 minutes
```

---

### Option B — Test pipeline once (development)
```bash
python scheduler.py --once
```

---

### Option C — Train BERT (if model not ready)
```bash
# Make sure data/training_data_clean.csv exists (run clean_csv.py first)
python train_bert.py

# Retrain with merged data (after collecting more RSS)
python train_bert.py --csv data/merged_training_data.csv
```

---

## API endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/health` | API status + total alerts |
| GET | `/alerts` | All alerts (paginated) |
| GET | `/alerts?limit=10&skip=0` | Pagination |
| GET | `/alerts/latest?n=5` | Latest N alerts |
| GET | `/alerts/type/flood` | Filter by disaster type |
| GET | `/alerts/location/Assam` | Filter by location/state |
| GET | `/alerts/locations` | All distinct locations |
| POST | `/pipeline/run` | Trigger pipeline manually |
| GET | `/docs` | Swagger UI |

---

## Connecting to Android app

In your Android app (Retrofit):

```java
// Base URL
String BASE_URL = "http://YOUR_SERVER_IP:8000/";

// GET all alerts
Call<AlertsResponse> getAlerts(@Query("limit") int limit);

// GET by location
Call<AlertsResponse> getByLocation(@Path("location") String location);

// GET by type
Call<AlertsResponse> getByType(@Path("type") String type);
```

---

## Retraining BERT with more data

Every time the pipeline runs, `data/rss_labeled_format.csv` grows.
Use this to improve the model:

```bash
# Merge original + RSS auto-labels (confidence >= 0.85)
python merge_for_retrain.py

# Retrain
python train_bert.py --csv data/merged_training_data.csv
```
