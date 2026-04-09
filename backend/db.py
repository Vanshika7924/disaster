# =============================================================================
# db.py
# MongoDB connection and all database operations
# =============================================================================

import logging
from datetime import datetime, timezone

import certifi
from pymongo import MongoClient, DESCENDING
from pymongo.errors import DuplicateKeyError

from config import MONGO_URI, DB_NAME, COLLECTION_NAME

logger = logging.getLogger(__name__)

_db_instance = None


class DisasterDB:
    def __init__(self):
        logger.info(f"[DB] Connecting to MongoDB: {DB_NAME}.{COLLECTION_NAME}")

        print("\n================= MONGODB DEBUG =================")
        print(f"🔥 MONGO_URI: {MONGO_URI}")
        print(f"🔥 DB NAME: {DB_NAME}")
        print(f"🔥 COLLECTION NAME: {COLLECTION_NAME}")

        try:
            # Atlas vs Local Mongo handling
            if MONGO_URI.startswith("mongodb+srv://"):
                self._client = MongoClient(
                    MONGO_URI,
                    tls=True,
                    tlsCAFile=certifi.where(),
                    serverSelectionTimeoutMS=10000
                )
            else:
                self._client = MongoClient(
                    MONGO_URI,
                    serverSelectionTimeoutMS=10000
                )

            # Force connection test
            self._client.admin.command("ping")
            print("✅ MongoDB connection successful")

            self._db = self._client[DB_NAME]
            self._collection = self._db[COLLECTION_NAME]

            print(f"🔥 USING DB: {self._db.name}")
            print(f"🔥 USING COLLECTION: {self._collection.name}")

            self._ensure_indexes()
            logger.info("[DB] Connected successfully")

        except Exception as e:
            print(f"❌ MongoDB connection failed: {e}")
            logger.error(f"[DB] Connection failed: {e}")
            raise

        print("=================================================\n")

    # -------------------------------------------------------------------------
    # INDEXES
    # -------------------------------------------------------------------------
    def _ensure_indexes(self):
        try:
            col = self._collection
            col.create_index("link", unique=True, sparse=True)
            col.create_index("disaster_type")
            col.create_index("location")
            col.create_index("state")
            col.create_index([("ingested_at", DESCENDING)])
            col.create_index([("disaster_time", DESCENDING)])
            print("✅ MongoDB indexes ensured")
        except Exception as e:
            print(f"⚠️ Index creation warning: {e}")
            logger.warning(f"[DB] Index creation warning: {e}")

    # -------------------------------------------------------------------------
    # INSERT ALERTS
    # -------------------------------------------------------------------------
    def insert_alerts(self, records: list) -> dict:
        inserted = duplicates = errors = 0
        now = datetime.now(timezone.utc).isoformat()

        print(f"\n🚀 insert_alerts called with {len(records)} records")

        for i, rec in enumerate(records):
            try:
                doc = {}
                for k, v in rec.items():
                    if str(v) == "nan":
                        doc[k] = None
                    else:
                        doc[k] = v

                doc["ingested_at"] = now

                print(f"\n📌 RECORD {i+1}:")
                print(f"   headline: {doc.get('headline')}")
                print(f"   disaster_type: {doc.get('disaster_type')}")
                print(f"   location: {doc.get('location')}")
                print(f"   link: {doc.get('link')}")

                result = self._collection.insert_one(doc)
                inserted += 1
                print(f"✅ Inserted with _id = {result.inserted_id}")

            except DuplicateKeyError:
                duplicates += 1
                print("⚠️ Duplicate found (same link), skipped")

            except Exception as e:
                errors += 1
                print(f"❌ Insert error: {e}")
                logger.error(f"[DB] Insert error: {e}")

        print(f"\n📊 FINAL RESULT => inserted={inserted}, duplicates={duplicates}, errors={errors}")

        return {
            "inserted": inserted,
            "duplicates": duplicates,
            "errors": errors
        }

    # -------------------------------------------------------------------------
    # QUERY ALERTS
    # -------------------------------------------------------------------------
    def get_all_alerts(self, limit=50, skip=0):
        docs = (
            self._collection.find({})
            .sort("ingested_at", DESCENDING)
            .skip(skip)
            .limit(limit)
        )
        return [self._serialize(d) for d in docs]

    def get_latest_alerts(self, n=10):
        return self.get_all_alerts(limit=n)

    def get_by_location(self, location, limit=50):
        query = {
            "$or": [
                {"location": {"$regex": location, "$options": "i"}},
                {"state": {"$regex": location, "$options": "i"}},
            ]
        }
        docs = (
            self._collection.find(query)
            .sort("ingested_at", DESCENDING)
            .limit(limit)
        )
        return [self._serialize(d) for d in docs]

    def get_by_type(self, disaster_type, limit=50):
        docs = (
            self._collection.find({
                "disaster_type": {"$regex": disaster_type, "$options": "i"}
            })
            .sort("ingested_at", DESCENDING)
            .limit(limit)
        )
        return [self._serialize(d) for d in docs]

    def get_locations(self):
        locations = self._collection.distinct("location")
        states = self._collection.distinct("state")
        merged = sorted(set([x for x in locations + states if x]))
        return merged

    def get_total_count(self):
        return self._collection.count_documents({})

    # -------------------------------------------------------------------------
    # TOKEN STORAGE
    # -------------------------------------------------------------------------
    def save_token(self, token: str, city: str):
        tokens_collection = self._db["tokens"]

        tokens_collection.update_one(
            {"token": token},
            {
                "$set": {
                    "token": token,
                    "city": city,
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }
            },
            upsert=True
        )

    def get_tokens_by_city(self, city: str):
        tokens_collection = self._db["tokens"]

        return list(tokens_collection.find({
            "city": {"$regex": city, "$options": "i"}
        }))

    # -------------------------------------------------------------------------
    # SERIALIZE
    # -------------------------------------------------------------------------
    @staticmethod
    def _serialize(doc):
        if "_id" in doc:
            doc["_id"] = str(doc["_id"])
        return doc

    # -------------------------------------------------------------------------
    # CLOSE
    # -------------------------------------------------------------------------
    def close(self):
        if self._client:
            self._client.close()


# -----------------------------------------------------------------------------
# SINGLETON
# -----------------------------------------------------------------------------
def get_db():
    global _db_instance
    if _db_instance is None:
        _db_instance = DisasterDB()
    return _db_instance