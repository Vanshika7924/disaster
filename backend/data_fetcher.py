# =============================================================================
# data_fetcher.py
# Fetches disaster news from Google News RSS feeds.
# Returns a clean list of dicts ready for the pipeline.
# =============================================================================

import re
import logging
import feedparser
import pandas as pd
from config import RSS_FEEDS

logger = logging.getLogger(__name__)


def _clean_title(raw: str) -> str:
    """Remove source suffix ('- The Hindu') and non-ASCII characters."""
    title = str(raw).encode("ascii", "ignore").decode().strip()
    title = re.sub(r"\s*-\s*[A-Z][^-]{2,40}$", "", title).strip()
    return title


def _clean_summary(raw: str) -> str:
    """Remove HTML tags, &nbsp; etc. from summary."""
    s = str(raw).encode("ascii", "ignore").decode()
    s = re.sub(r"<[^>]+>", "", s)
    s = re.sub(r"&\w+;", " ", s)
    return s.strip()


def fetch_rss_news(feeds: list = None) -> list:
    """
    Fetches news from all configured RSS feeds.
    Deduplicates by title.

    Args:
        feeds: optional list of RSS URLs (uses config.RSS_FEEDS by default)

    Returns:
        List of dicts:
        {
          "title"    : cleaned headline,
          "summary"  : cleaned summary (may be empty),
          "link"     : article URL,
          "published": publish timestamp string,
          "text"     : title + summary (used for BERT prediction)
        }
    """
    if feeds is None:
        feeds = RSS_FEEDS

    seen   = set()
    items  = []

    for url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                title   = _clean_title(entry.get("title", ""))
                summary = _clean_summary(entry.get("summary", ""))
                link    = entry.get("link", "")
                pub     = entry.get("published", "")

                if not title or len(title) < 10:
                    continue

                key = title.lower()
                if key in seen:
                    continue
                seen.add(key)

                text = (title + " " + summary).strip() if summary else title
                items.append({
                    "title"    : title,
                    "summary"  : summary,
                    "link"     : link,
                    "published": pub,
                    "text"     : text,
                })

        except Exception as e:
            logger.error(f"Feed error [{url[:60]}]: {e}")

    logger.info(f"fetch_rss_news → {len(items)} unique articles")
    return items


def fetch_to_dataframe(feeds: list = None) -> pd.DataFrame:
    """Convenience wrapper that returns a DataFrame instead of a list."""
    return pd.DataFrame(fetch_rss_news(feeds))
