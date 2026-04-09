# =============================================================================
# app.py
# FastAPI application for Disaster Eye backend
# =============================================================================

import logging
from datetime import datetime, timezone
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from db import get_db

# Optional router
try:
    from routers.alerts import router as alerts_router
    ROUTER_AVAILABLE = True
except Exception as e:
    ROUTER_AVAILABLE = False
    print(f"⚠️ alerts router not loaded: {e}")

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("app")

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Disaster Eye API",
    description="Real-time disaster alert backend for India",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

if ROUTER_AVAILABLE:
    app.include_router(alerts_router)

# -----------------------------------------------------------------------------
# Request Models
# -----------------------------------------------------------------------------
class TokenRequest(BaseModel):
    token: str
    city: str

# -----------------------------------------------------------------------------
# Root
# -----------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    return {"message": "Disaster Eye API is running", "docs": "/docs"}

# -----------------------------------------------------------------------------
# Health
# -----------------------------------------------------------------------------
@app.get("/health", tags=["System"])
def health():
    try:
        total = get_db().get_total_count()
        db_ok = True
    except Exception as e:
        total = 0
        db_ok = False
        logger.error(f"DB health check failed: {e}")

    return JSONResponse({
        "status": "ok" if db_ok else "degraded",
        "db_connected": db_ok,
        "total_alerts": total,
        "server_time": datetime.now(timezone.utc).isoformat(),
    })

# -----------------------------------------------------------------------------
# Register Token
# -----------------------------------------------------------------------------
@app.post("/register-token", tags=["Notifications"])
async def register_token(data: TokenRequest):
    try:
        db = get_db()
        db.save_token(data.token, data.city)

        return JSONResponse({
            "success": True,
            "message": "Token registered successfully",
            "token": data.token,
            "city": data.city
        })

    except Exception as e:
        logger.error(f"Token registration failed: {e}")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

# -----------------------------------------------------------------------------
# Trigger Pipeline
# -----------------------------------------------------------------------------
@app.post("/pipeline/run", tags=["Pipeline"])
def trigger_pipeline():
    try:
        from pipeline import run_pipeline

        def safe_run():
            try:
                print("\n🚀 Background pipeline started...")
                run_pipeline()
                print("✅ Background pipeline finished.")
            except Exception as e:
                print(f"❌ Pipeline crashed: {e}")
                logger.exception("Pipeline crashed")

        threading.Thread(target=safe_run, daemon=True).start()

        return JSONResponse({
            "message": "Pipeline triggered in background.",
            "check": "GET /health to see updated alert count."
        })

    except Exception as e:
        logger.exception("Failed to trigger pipeline")
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")