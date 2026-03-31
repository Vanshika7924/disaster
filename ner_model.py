# =============================================================================
# ner_model.py  — Final Production Version
#
# PROBLEM 1 FIX: City → State automatic resolution
#   Uses geopy (Nominatim / OpenStreetMap) as PRIMARY method
#   Falls back to our hardcoded dictionary if geopy fails
#   This means ANY Indian city automatically gets correct state
#
# PROBLEM 2 FIX: Non-disaster news filter
#   Confidence threshold check
#   Non-disaster keyword patterns (workshop, seminar, election, etc.)
#   is_india_news() now requires explicit India signal
# =============================================================================

import re
import time
import logging
import spacy

logger = logging.getLogger(__name__)

try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError("Run: python -m spacy download en_core_web_sm")

# =============================================================================
# GEOPY — automatic city → state resolution (PRIMARY method)
# Requires: pip install geopy
# =============================================================================
_geocoder = None

def _get_geocoder():
    """Lazy load geopy geocoder — loads once, reused forever."""
    global _geocoder
    if _geocoder is None:
        try:
            from geopy.geocoders import Nominatim
            from geopy.extra.rate_limiter import RateLimiter
            geolocator = Nominatim(user_agent="disaster_eye_v1")
            # Rate limiter: max 1 request/second (Nominatim rule)
            _geocoder = RateLimiter(geolocator.geocode, min_delay_seconds=1.1)
            logger.info("geopy geocoder loaded")
        except ImportError:
            logger.warning("geopy not installed. Run: pip install geopy")
            _geocoder = False   # False = tried but failed
    return _geocoder


def get_state_from_city_geopy(city: str) -> str | None:
    """
    Uses OpenStreetMap (free, no API key) to get state from any city name.
    Returns state name string or None.

    Example:
        get_state_from_city_geopy("Gangtok")  → "Sikkim"
        get_state_from_city_geopy("Pasighat") → "Arunachal Pradesh"
        get_state_from_city_geopy("West Kameng") → "Arunachal Pradesh"
    """
    geocoder = _get_geocoder()
    if not geocoder:
        return None

    try:
        location = geocoder(city + ", India", language="en", timeout=5)
        if location and location.raw:
            addr = location.raw.get("address", {})
            state = addr.get("state") or addr.get("state_district")
            return state
    except Exception as e:
        logger.warning(f"geopy lookup failed for '{city}': {e}")
    return None


# =============================================================================
# FALLBACK: Hardcoded Indian cities dictionary
# Used when geopy is unavailable or times out
# =============================================================================
INDIAN_CITIES_FALLBACK: dict = {
    # Metros
    "mumbai": "Maharashtra", "delhi": "Delhi", "kolkata": "West Bengal",
    "chennai": "Tamil Nadu", "bangalore": "Karnataka", "bengaluru": "Karnataka",
    "hyderabad": "Telangana", "ahmedabad": "Gujarat", "pune": "Maharashtra",
    "surat": "Gujarat", "new delhi": "Delhi",
    # North
    "jaipur": "Rajasthan", "jodhpur": "Rajasthan", "udaipur": "Rajasthan",
    "kota": "Rajasthan", "ajmer": "Rajasthan", "bikaner": "Rajasthan",
    "sikar": "Rajasthan", "alwar": "Rajasthan",
    "lucknow": "Uttar Pradesh", "kanpur": "Uttar Pradesh", "agra": "Uttar Pradesh",
    "varanasi": "Uttar Pradesh", "prayagraj": "Uttar Pradesh",
    "allahabad": "Uttar Pradesh", "meerut": "Uttar Pradesh",
    "noida": "Uttar Pradesh", "ghaziabad": "Uttar Pradesh",
    "amritsar": "Punjab", "ludhiana": "Punjab", "jalandhar": "Punjab",
    "chandigarh": "Punjab", "patiala": "Punjab",
    "faridabad": "Haryana", "gurugram": "Haryana", "gurgaon": "Haryana",
    "rohtak": "Haryana", "panipat": "Haryana", "karnal": "Haryana",
    "sonipat": "Haryana",
    # Uttarakhand
    "dehradun": "Uttarakhand", "haridwar": "Uttarakhand", "rishikesh": "Uttarakhand",
    "uttarkashi": "Uttarakhand", "chamoli": "Uttarakhand", "tehri": "Uttarakhand",
    "pithoragarh": "Uttarakhand", "almora": "Uttarakhand", "nainital": "Uttarakhand",
    "rudraprayag": "Uttarakhand", "bageshwar": "Uttarakhand",
    "kedarnath": "Uttarakhand", "badrinath": "Uttarakhand",
    # Himachal
    "shimla": "Himachal Pradesh", "manali": "Himachal Pradesh",
    "dharamsala": "Himachal Pradesh", "mandi": "Himachal Pradesh",
    "kullu": "Himachal Pradesh", "solan": "Himachal Pradesh",
    # J&K / Ladakh
    "srinagar": "Jammu & Kashmir", "jammu": "Jammu & Kashmir",
    "poonch": "Jammu & Kashmir", "rajouri": "Jammu & Kashmir",
    "sonamarg": "Jammu & Kashmir", "zojila": "Jammu & Kashmir",
    "ramban": "Jammu & Kashmir", "kashmir": "Jammu & Kashmir",
    "leh": "Ladakh", "kargil": "Ladakh",
    # Bihar / Jharkhand
    "patna": "Bihar", "gaya": "Bihar", "muzaffarpur": "Bihar",
    "ranchi": "Jharkhand", "jamshedpur": "Jharkhand", "dhanbad": "Jharkhand",
    "ramgarh": "Jharkhand",
    # Odisha
    "bhubaneswar": "Odisha", "cuttack": "Odisha", "puri": "Odisha",
    "rourkela": "Odisha", "sambalpur": "Odisha",
    # West Bengal
    "siliguri": "West Bengal", "darjeeling": "West Bengal",
    "howrah": "West Bengal", "asansol": "West Bengal",
    # Assam
    "guwahati": "Assam", "silchar": "Assam", "dibrugarh": "Assam",
    "jorhat": "Assam", "tezpur": "Assam",
    # Northeast
    "imphal": "Manipur", "churachandpur": "Manipur",
    "shillong": "Meghalaya", "tura": "Meghalaya",
    "aizawl": "Mizoram",
    "kohima": "Nagaland", "dimapur": "Nagaland", "mokokchung": "Nagaland",
    "agartala": "Tripura",
    "gangtok": "Sikkim", "mangan": "Sikkim", "namchi": "Sikkim",
    # Arunachal Pradesh
    "itanagar": "Arunachal Pradesh", "pasighat": "Arunachal Pradesh",
    "yingkiong": "Arunachal Pradesh", "tawang": "Arunachal Pradesh",
    "west kameng": "Arunachal Pradesh", "east kameng": "Arunachal Pradesh",
    "dibang valley": "Arunachal Pradesh", "lower dibang valley": "Arunachal Pradesh",
    "upper siang": "Arunachal Pradesh", "lower siang": "Arunachal Pradesh",
    "east siang": "Arunachal Pradesh", "west siang": "Arunachal Pradesh",
    "arunachal": "Arunachal Pradesh",
    # Kerala
    "kochi": "Kerala", "thiruvananthapuram": "Kerala", "trivandrum": "Kerala",
    "kozhikode": "Kerala", "thrissur": "Kerala", "kollam": "Kerala",
    "palakkad": "Kerala", "malappuram": "Kerala", "wayanad": "Kerala",
    # Tamil Nadu
    "coimbatore": "Tamil Nadu", "madurai": "Tamil Nadu", "salem": "Tamil Nadu",
    "tirunelveli": "Tamil Nadu", "tiruchirappalli": "Tamil Nadu",
    "trichy": "Tamil Nadu", "vellore": "Tamil Nadu", "thanjavur": "Tamil Nadu",
    "thoothukudi": "Tamil Nadu",
    # Andhra Pradesh
    "visakhapatnam": "Andhra Pradesh", "vijayawada": "Andhra Pradesh",
    "tirupati": "Andhra Pradesh", "guntur": "Andhra Pradesh",
    "kakinada": "Andhra Pradesh", "nellore": "Andhra Pradesh",
    "andhra": "Andhra Pradesh", "vizag": "Andhra Pradesh",
    # Telangana
    "warangal": "Telangana", "karimnagar": "Telangana", "nizamabad": "Telangana",
    "rajendranagar": "Telangana",
    # Maharashtra
    "nagpur": "Maharashtra", "aurangabad": "Maharashtra", "nashik": "Maharashtra",
    "thane": "Maharashtra", "solapur": "Maharashtra", "kolhapur": "Maharashtra",
    "amravati": "Maharashtra",
    # MP
    "bhopal": "Madhya Pradesh", "indore": "Madhya Pradesh",
    "jabalpur": "Madhya Pradesh", "gwalior": "Madhya Pradesh",
    "meghnagar": "Madhya Pradesh",
    # Chhattisgarh
    "raipur": "Chhattisgarh", "bilaspur": "Chhattisgarh",
    # Gujarat
    "rajkot": "Gujarat", "vadodara": "Gujarat", "bhavnagar": "Gujarat",
    "jamnagar": "Gujarat", "gandhinagar": "Gujarat",
    # Andaman
    "port blair": "Andaman & Nicobar Islands",
    "andaman": "Andaman & Nicobar Islands",
    "andaman sea": "Andaman & Nicobar Islands",
    "little andaman": "Andaman & Nicobar Islands",
    # State names as city references
    "nagaland": "Nagaland", "manipur": "Manipur", "meghalaya": "Meghalaya",
    "assam": "Assam", "sikkim": "Sikkim", "tripura": "Tripura",
    "mizoram": "Mizoram", "ladakh": "Ladakh",
    "uttarakhand": "Uttarakhand", "odisha": "Odisha", "kerala": "Kerala",
    "goa": "Goa", "himachal": "Himachal Pradesh",
    "gujarat": "Gujarat", "rajasthan": "Rajasthan",
    "bihar": "Bihar", "jharkhand": "Jharkhand",
    "haryana": "Haryana", "punjab": "Punjab",
    "chhattisgarh": "Chhattisgarh",
}

# =============================================================================
# NON-DISASTER KEYWORDS — filter out news that is NOT an actual disaster
# =============================================================================
NON_DISASTER_HEADLINE_PATTERNS = [
    # Workshops / Seminars / Events
    r"\bworkshop\b", r"\bseminar\b", r"\bconference\b", r"\bsummit\b",
    r"\bwebinar\b", r"\bsymposium\b", r"\bexpo\b", r"\bexhibition\b",
    r"\btraining\b", r"\bawareness camp\b",
    # Prevention / Preparedness (not actual event)
    r"\bprevention\b", r"\bpreparedness\b", r"\bdrills?\b", r"\bmock drill\b",
    r"\bpreparation\b", r"\bplan(ning)?\b", r"\bpolicy\b",
    # Political / Elections
    r"\belection\b", r"\bvoting\b", r"\blandslide victory\b",
    r"\blandslide win\b", r"\blandslide defeat\b",
    r"\bpolitical\b", r"\bcampaign\b", r"\brally\b",
    # Sports / Entertainment metaphors
    r"\bflood(ed)? (of|with) (memes?|photos?|posts?|messages?|reactions?|tweets?|videos?|orders?|calls?|requests?|queries|complaints)\b",
    r"\bflood social media\b", r"\bmemes?\b", r"\bjokes?\b",
    r"\bcricket\b", r"\bipl\b", r"\bfootball\b", r"\bboxing\b",
    # Financial / Business
    r"\bstock market\b", r"\bshare(s)?\b", r"\bsensex\b", r"\bnifty\b",
    r"\bipo\b", r"\bquarterly\b", r"\brevenue\b", r"\bprofit\b",
    # Research / Historical / Anniversary
    r"\banniversary\b", r"\bcommemorate\b", r"\bhistorical\b",
    r"\bresearch\b", r"\bstudy\b", r"\breport(s)? (on|about|find)\b",
    r"\bscientists?\b", r"\bexperts?\b",
    # Relief / Aid (not the disaster itself)
    r"\brelief fund\b", r"\bfund(ing)?\b", r"\bdonation\b",
    r"\bscheme\b", r"\bpackage\b", r"\bcompensation\b",
    # Entertainment news with disaster words
    r"\bbollywood\b", r"\bnetflix\b", r"\bfilm\b", r"\bmovie\b",
    r"\bactor\b", r"\bactress\b", r"\bcelebrity\b",
]

# Compiled patterns for speed
_NON_DISASTER_COMPILED = [re.compile(p, re.IGNORECASE) for p in NON_DISASTER_HEADLINE_PATTERNS]

# =============================================================================
# FOREIGN PATTERNS
# =============================================================================
FOREIGN_PATTERNS = [
    r"\bvanuatu\b", r"\btonga\b", r"\bfiji\b", r"\bsamoa\b",
    r"\bsouth pacific\b", r"\bpacific ocean\b", r"\bpacific island",
    r"\bwestern australia\b", r"\bnew south wales\b", r"\bqueensland\b",
    r"\bvictoria\b", r"\bsouth australia\b", r"\btasmania\b",
    r"\balaska\b", r"\bcalifornia\b", r"\bflorida\b", r"\btexas\b",
    r"\bnevada\b", r"\bvirginia\b", r"\bcolorado\b", r"\boregon\b",
    r"\bmontana\b", r"\blouisiana\b", r"\bnebraska\b", r"\butah\b",
    r"\bidaho\b", r"\barizona\b", r"\bmichigan\b",
    r"\bsouth carolina\b", r"\bnorth carolina\b",
    r"\bnew jersey\b", r"\bnew york\b", r"\bconnecticut\b",
    r"\bsan diego\b", r"\blos angeles\b", r"\bhernando county\b",
    r"\bsantee\b", r"\bbarstow\b", r"\bweeki wachee\b", r"\belk park\b",
    r"\b\w[\w\s]+ county\b",
    r"\biranb\b", r"\biraq\b", r"\bisrael\b", r"\bafghanistan\b",
    r"\bpakistan\b", r"\bchina\b", r"\bjapan\b", r"\brussia\b",
    r"\bukraine\b", r"\bturkey\b", r"\bsyria\b", r"\byemen\b",
    r"\bindonesia\b", r"\bphilippines\b", r"\bthailand\b",
    r"\bmyanmar\b", r"\bsri lanka\b", r"\bbangladesh\b",
    r"\bmalaysia\b", r"\bsingapore\b", r"\bnew zealand\b",
    r"\baustralia\b", r"\bcanada\b",
    r"\bunited states\b", r"\busa\b",
    r"\boman\b", r"\buae\b", r"\bdubai\b", r"\bqatar\b",
    r"\btajikistan\b", r"\bkazakhstan\b",
    r"\bgfz\b", r"\bvolcano discovery\b",
    r"\bindian ocean\b", r"\barabian sea\b",
    r"\bchiang mai\b", r"\bchiang rai\b",
]

FOREIGN_KEYWORDS: set = {
    "vanuatu", "tonga", "fiji", "samoa", "iran", "iraq", "israel",
    "palestine", "afghanistan", "pakistan", "nepal", "bhutan", "bangladesh",
    "china", "japan", "russia", "ukraine", "usa", "united states", "canada",
    "uk", "united kingdom", "england", "france", "germany", "italy", "spain",
    "australia", "sri lanka", "myanmar", "thailand", "indonesia", "philippines",
    "turkey", "syria", "yemen", "saudi arabia", "egypt", "brazil", "mexico",
    "south korea", "north korea", "taiwan", "vietnam", "malaysia", "singapore",
    "new zealand", "nigeria", "kenya", "ethiopia", "cuba", "colombia",
    "chile", "peru", "argentina", "oman", "uae", "dubai", "qatar", "kuwait",
    "tajikistan", "uzbekistan", "kazakhstan", "western australia", "gfz",
    "south pacific", "indian ocean", "pacific ocean", "elk park", "san diego",
}

# =============================================================================
# MAIN PUBLIC FUNCTIONS
# =============================================================================

def extract_location(text: str) -> tuple:
    """
    Extract location from news text.

    Priority:
    1. Hardcoded Indian cities dict (fastest, most reliable for known cities)
    2. geopy Nominatim (automatic — any city → correct state via OpenStreetMap)
    3. spaCy NER fallback

    Returns: (location, state, country) or (None, None, None)
    """
    if not text:
        return None, None, None

    tl = str(text).lower()

    # ── 1. Hardcoded dict — longest match first ────────────────────────────
    for city, state in sorted(INDIAN_CITIES_FALLBACK.items(), key=lambda x: len(x[0]), reverse=True):
        if re.search(r"\b" + re.escape(city) + r"\b", tl):
            return city.title(), state, "India"

    # ── 2. spaCy NER — get candidate location ─────────────────────────────
    doc = _nlp(str(text))
    candidate = None
    for ent in doc.ents:
        if ent.label_ in ("GPE", "LOC"):
            name = ent.text.strip()
            name_lower = name.lower()
            if name_lower in ("india", "indian", "gfz"):
                continue
            if name_lower in FOREIGN_KEYWORDS:
                continue
            if any(w in name_lower for w in ["ocean", "sea", "bay", "strait", "pacific", "atlantic"]):
                continue
            if re.search(r"\bcyclone\b", name_lower):
                continue
            # Check foreign patterns
            is_foreign = False
            for pat in FOREIGN_PATTERNS:
                if re.search(pat, name_lower):
                    is_foreign = True
                    break
            if is_foreign:
                continue
            candidate = name
            break

    if candidate:
        # Try geopy first to get correct state
        state = get_state_from_city_geopy(candidate)
        if not state:
            # Fallback to dict
            state = INDIAN_CITIES_FALLBACK.get(candidate.lower())
        return candidate, state, "India"

    return None, None, None


def is_real_disaster_news(headline: str, confidence: float = 0.0) -> bool:
    """
    Returns True only if the headline describes an ACTUAL disaster event.
    Filters out:
    - Workshops, seminars, awareness events
    - Political "landslide victory" type headlines
    - "Flood of memes" type metaphorical uses
    - Research / anniversary / historical articles
    - Low confidence predictions

    Args:
        headline   : news headline text
        confidence : BERT confidence score (0.0 to 1.0)

    Returns:
        True  → real disaster news, keep it
        False → not a real disaster, remove it
    """
    if not headline:
        return False

    # Low confidence → reject
    if confidence < 0.50:
        return False

    tl = str(headline).lower()

    # Check non-disaster patterns
    for pattern in _NON_DISASTER_COMPILED:
        if pattern.search(tl):
            logger.debug(f"Non-disaster pattern matched: '{pattern.pattern}' in '{headline[:60]}'")
            return False

    return True


def is_india_news(text: str) -> bool:
    """
    Returns True ONLY if text is clearly about India.
    3-step logic:
      1. Foreign pattern found → False
      2. India keyword found → True
      3. Known Indian city/state found → True
      4. Nothing found → False (strict — unknown locations rejected)
    """
    tl = str(text).lower()

    # Step 1: Foreign patterns
    for pat in FOREIGN_PATTERNS:
        if re.search(pat, tl):
            return False
    for fk in FOREIGN_KEYWORDS:
        if re.search(r"\b" + re.escape(fk) + r"\b", tl):
            return False
    if re.search(r"\b\w[\w\s]+ county\b", tl):
        return False

    # Step 2: India mentioned
    if re.search(r"\bindia\b|\bindian\b", tl):
        if "indian ocean" in tl and "india" not in tl.replace("indian ocean", ""):
            return False
        return True

    # Step 3: Known Indian city/state in text
    for city in INDIAN_CITIES_FALLBACK:
        if re.search(r"\b" + re.escape(city) + r"\b", tl):
            return True

    return False
