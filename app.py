# =============================================================================
# app.py
# FastAPI application for Disaster Eye backend.
#
# Run:
#   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
#
# Swagger UI available at:
#   http://localhost:8000/docs
# =============================================================================

import logging
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from routers.alerts import router as alerts_router
from db import get_db

# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("app")

# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "Disaster Eye API",
    description = "Real-time disaster alert backend for India",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# Allow Android app and web dashboard to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["GET", "POST"],
    allow_headers  = ["*"],
)

# Register all /alerts routes
app.include_router(alerts_router)


# ─────────────────────────────────────────────────────────────────────────────
# Root & Health endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return {"message": "Disaster Eye API is running", "docs": "/docs"}


@app.get("/health", tags=["System"])
def health():
    """
    Health check endpoint.
    Returns API status, total alerts, and server time.

    **Use this in your Android app to verify the backend is reachable.**
    """
    try:
        total = get_db().get_total_count()
        db_ok = True
    except Exception as e:
        total = 0
        db_ok = False
        logger.error(f"DB health check failed: {e}")

    return JSONResponse({
        "status"       : "ok" if db_ok else "degraded",
        "db_connected" : db_ok,
        "total_alerts" : total,
        "server_time"  : datetime.now(timezone.utc).isoformat(),
    })


@app.post("/pipeline/run", tags=["Pipeline"])
def trigger_pipeline():
    """
    Manually trigger one RSS fetch + predict + store cycle.
    Useful for testing without waiting for the scheduler.
    """
    from pipeline import run_pipeline
    import threading
    threading.Thread(target=run_pipeline, daemon=True).start()
    return JSONResponse({
        "message": "Pipeline triggered in background.",
        "check"  : "GET /health to see updated alert count."
    })


# ─────────────────────────────────────────────────────────────────────────────
# Entry point (for direct run)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
