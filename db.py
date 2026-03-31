# =============================================================================
# db.py
# MongoDB connection and all database operations.
#
# Usage:
#   from db import get_db
#   db = get_db()
#   db.insert_alerts(records)
#   db.get_all_alerts(limit=20)
# =============================================================================

import logging
from datetime import datetime, timezone
from typing import Optional

import certifi
from pymongo import MongoClient, DESCENDING
from pymongo.errors import DuplicateKeyError
from bson import ObjectId

from config import MONGO_URI, DB_NAME, COLLECTION_NAME

logger = logging.getLogger(__name__)

# Singleton
_db_instance = None


class DisasterDB:
    """
    All MongoDB operations for Disaster Eye.
    Instantiated once and reused.
    """

    def __init__(self):
        logger.info(f"[DB] Connecting to MongoDB: {DB_NAME}.{COLLECTION_NAME}")
        self._client     = MongoClient(MONGO_URI, tlsCAFile=certifi.where())
        self._db         = self._client[DB_NAME]
        self._collection = self._db[COLLECTION_NAME]
        self._ensure_indexes()
        logger.info("[DB] Connected")

    def _ensure_indexes(self):
        """Create indexes for performance and deduplication."""
        col = self._collection
        col.create_index("link",         unique=True, sparse=True)
        col.create_index("disaster_type")
        col.create_index("location")
        col.create_index("state")
        col.create_index([("ingested_at", DESCENDING)])
        col.create_index([("disaster_time", DESCENDING)])

    # ─────────────────────────────────────────────────────────────────────────
    # INSERT
    # ─────────────────────────────────────────────────────────────────────────
    def insert_alerts(self, records: list) -> dict:
        """
        Insert a list of alert dicts.
        Skips duplicates silently (unique index on 'link').

        Returns: {"inserted": N, "duplicates": M, "errors": K}
        """
        inserted = duplicates = errors = 0
        now = datetime.now(timezone.utc).isoformat()

        for rec in records:
            # Clean NaN values — MongoDB doesn't accept Python float NaN
            doc = {
                k: (None if (hasattr(v, "__float__") and str(v) == "nan") else v)
                for k, v in rec.items()
            }
            doc["ingested_at"] = now

            try:
                self._collection.insert_one(doc)
                inserted += 1
            except DuplicateKeyError:
                duplicates += 1
            except Exception as e:
                errors += 1
                logger.error(f"[DB] Insert error: {e}")

        logger.info(f"[DB] inserted={inserted} duplicates={duplicates} errors={errors}")
        return {"inserted": inserted, "duplicates": duplicates, "errors": errors}

    # ─────────────────────────────────────────────────────────────────────────
    # QUERY helpers
    # ─────────────────────────────────────────────────────────────────────────
    def get_all_alerts(self, limit: int = 50, skip: int = 0) -> list:
        """All alerts, newest first."""
        docs = (
            self._collection.find({})
            .sort("ingested_at", DESCENDING)
            .skip(skip)
            .limit(limit)
        )
        return [self._serialize(d) for d in docs]

    def get_latest_alerts(self, n: int = 10) -> list:
        """N most recent alerts."""
        return self.get_all_alerts(limit=n)

    def get_by_type(self, disaster_type: str, limit: int = 50) -> list:
        """Filter by disaster type (case-insensitive)."""
        docs = (
            self._collection.find(
                {"disaster_type": {"$regex": disaster_type, "$options": "i"}}
            )
            .sort("ingested_at", DESCENDING)
            .limit(limit)
        )
        return [self._serialize(d) for d in docs]

    def get_by_location(self, location: str, limit: int = 50) -> list:
        """Filter by city or state (case-insensitive partial match)."""
        query = {
            "$or": [
                {"location": {"$regex": location, "$options": "i"}},
                {"state":    {"$regex": location, "$options": "i"}},
            ]
        }
        docs = (
            self._collection.find(query)
            .sort("ingested_at", DESCENDING)
            .limit(limit)
        )
        return [self._serialize(d) for d in docs]

    def get_total_count(self) -> int:
        return self._collection.count_documents({})

    def get_distinct_locations(self) -> list:
        return sorted(
            loc for loc in self._collection.distinct("location") if loc
        )

    def get_distinct_types(self) -> list:
        return sorted(
            t for t in self._collection.distinct("disaster_type") if t
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _serialize(doc: dict) -> dict:
        """Convert MongoDB doc to JSON-safe dict."""
        if "_id" in doc:
            doc["_id"] = str(doc["_id"])
        return doc

    def close(self):
        if self._client:
            self._client.close()


# ─────────────────────────────────────────────────────────────────────────────
# Singleton getter
# ─────────────────────────────────────────────────────────────────────────────
def get_db() -> DisasterDB:
    global _db_instance
    if _db_instance is None:
        _db_instance = DisasterDB()
    return _db_instance
