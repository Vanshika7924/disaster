# =============================================================================
# time_extractor.py
# Extracts when a disaster HAPPENED from news text.
#
# Tries (in order):
#   1. Relative phrases: "yesterday", "last night", "this morning"
#   2. Day names: "Monday", "on Friday"
#   3. Explicit dates: "March 10", "10/03/2026"
#   4. Fallback: use published_time from RSS feed
# =============================================================================

import re
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from dateutil import parser as date_parser

DAYS_MAP = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}

MONTHS_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4,
    "may": 5, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

RELATIVE_MAP = [
    (r"\bjust now\b",         timedelta(minutes=0)),
    (r"\bthis morning\b",     timedelta(hours=6)),
    (r"\bearly morning\b",    timedelta(hours=8)),
    (r"\bthis afternoon\b",   timedelta(hours=4)),
    (r"\bthis evening\b",     timedelta(hours=2)),
    (r"\blast night\b",       timedelta(hours=12)),
    (r"\bovernight\b",        timedelta(hours=10)),
    (r"\byesterday\b",        timedelta(days=1)),
    (r"\btwo days ago\b",     timedelta(days=2)),
    (r"\bthree days ago\b",   timedelta(days=3)),
    (r"\blast week\b",        timedelta(weeks=1)),
    (r"\brecently\b",         timedelta(days=2)),
    (r"\bearlier today\b",    timedelta(hours=3)),
    (r"\bminutes ago\b",      timedelta(minutes=30)),
    (r"\ban? hour ago\b",     timedelta(hours=1)),
]


def _safe_parse(published) -> Optional[datetime]:
    """Safely convert published string/datetime to tz-aware datetime."""
    if not published:
        return None
    if isinstance(published, datetime):
        return published.replace(tzinfo=timezone.utc) if published.tzinfo is None else published
    s = str(published).strip()
    if s in ("", "nan", "None", "NaT", "NaN"):
        return None
    try:
        dt = date_parser.parse(s)
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt
    except Exception:
        return None


def extract_disaster_time(text: str, published=None) -> Tuple[Optional[str], str]:
    """
    Extract disaster event time from text.

    Args:
        text      : news headline or full text
        published : RSS published time (string or datetime)

    Returns:
        (time_string, source_label)

        source_label values:
            "extracted_relative"  → from phrase like "yesterday"
            "extracted_day"       → from weekday name like "Monday"
            "extracted_explicit"  → from date like "March 10"
            "published_fallback"  → from RSS published_time
            "unknown"             → nothing found
    """
    base = _safe_parse(published) or datetime.now(timezone.utc)
    tl   = str(text).lower()

    # ── 1. Relative phrases ───────────────────────────────────────────────────
    for pattern, delta in RELATIVE_MAP:
        if re.search(pattern, tl):
            return (base - delta).isoformat(), "extracted_relative"

    # ── 2. Day name ───────────────────────────────────────────────────────────
    day_pat = r"\b(" + "|".join(DAYS_MAP.keys()) + r")\b"
    dm = re.search(day_pat, tl)
    if dm:
        target_wd = DAYS_MAP[dm.group(1)]
        diff = (base.weekday() - target_wd) % 7 or 7
        return (base - timedelta(days=diff)).date().isoformat(), "extracted_day"

    # ── 3. Explicit date in text ─────────────────────────────────────────────
    month_pat = "|".join(MONTHS_MAP.keys())
    for pat in [
        rf"\b({month_pat})\s+(\d{{1,2}})(?:,?\s+(\d{{4}}))?\b",
        rf"\b(\d{{1,2}})\s+({month_pat})(?:\s+(\d{{4}}))?\b",
    ]:
        m = re.search(pat, tl)
        if m:
            try:
                g = [x for x in m.groups() if x]
                if g[0] in MONTHS_MAP:
                    month, day = MONTHS_MAP[g[0]], int(g[1])
                    year = int(g[2]) if len(g) > 2 else base.year
                else:
                    day, month = int(g[0]), MONTHS_MAP[g[1]]
                    year = int(g[2]) if len(g) > 2 else base.year
                return datetime(year, month, day, tzinfo=timezone.utc).isoformat(), "extracted_explicit"
            except Exception:
                continue

    # ── 4. Fallback: published_time ───────────────────────────────────────────
    pub = _safe_parse(published)
    if pub:
        return pub.isoformat(), "published_fallback"

    return None, "unknown"
