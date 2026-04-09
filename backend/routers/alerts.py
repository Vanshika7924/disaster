# =============================================================================
# routers/alerts.py
# All /alerts endpoints for the Disaster Eye API.
# =============================================================================

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from db import get_db

router = APIRouter(prefix="/alerts", tags=["Alerts"])


@router.get("")
def get_alerts(
    limit: int = Query(20, ge=1, le=100, description="Max results (1-100)"),
    skip:  int = Query(0,  ge=0,          description="Pagination offset"),
):
    """
    Returns all disaster alerts, newest first.

    **Examples:**
    - `GET /alerts` — first 20 alerts
    - `GET /alerts?limit=50` — first 50 alerts
    - `GET /alerts?limit=10&skip=10` — page 2
    """
    db     = get_db()
    alerts = db.get_all_alerts(limit=limit, skip=skip)
    total  = db.get_total_count()
    return JSONResponse({
        "total"  : total,
        "count"  : len(alerts),
        "page"   : skip // limit + 1 if limit else 1,
        "alerts" : alerts,
    })


@router.get("/latest")
def get_latest(
    n: int = Query(10, ge=1, le=50, description="Number of latest alerts"),
):
    """
    Returns the N most recent alerts.

    **Example:** `GET /alerts/latest?n=5`
    """
    alerts = get_db().get_latest_alerts(n=n)
    return JSONResponse({"count": len(alerts), "alerts": alerts})


@router.get("/locations")
def get_locations():
    """Returns all distinct locations that have alerts."""
    db  = get_db()
    return JSONResponse({
        "locations": db.get_distinct_locations(),
        "types"    : db.get_distinct_types(),
    })


@router.get("/type/{disaster_type}")
def get_by_type(
    disaster_type: str,
    limit: int = Query(20, ge=1, le=100),
):
    """
    Filter alerts by disaster type.

    **Supported types:** earthquake, flood, cyclone, fire, landslide,
    explosion, tsunami, drought, accident, collapse

    **Example:** `GET /alerts/type/flood`
    """
    alerts = get_db().get_by_type(disaster_type=disaster_type, limit=limit)
    if not alerts:
        raise HTTPException(
            status_code=404,
            detail=f"No alerts found for type: '{disaster_type}'"
        )
    return JSONResponse({
        "disaster_type": disaster_type,
        "count"        : len(alerts),
        "alerts"       : alerts,
    })


@router.get("/location/{location}")
def get_by_location(
    location: str,
    limit: int = Query(20, ge=1, le=100),
):
    """
    Filter alerts by city, district, or state name (partial match).

    **Examples:**
    - `GET /alerts/location/Assam`
    - `GET /alerts/location/Delhi`
    - `GET /alerts/location/Uttarakhand`
    """
    alerts = get_db().get_by_location(location=location, limit=limit)
    if not alerts:
        raise HTTPException(
            status_code=404,
            detail=f"No alerts found for location: '{location}'"
        )
    return JSONResponse({
        "location": location,
        "count"   : len(alerts),
        "alerts"  : alerts,
    })
