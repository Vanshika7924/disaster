# =============================================================================
# ner_model.py — FINAL STRICT VERSION
# PURPOSE:
#   Extract ONLY valid Indian locations
#   Reject foreign / mixed-country / noisy locations
# =============================================================================

import re
import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError("Run: python -m spacy download en_core_web_sm")


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = str(text)
    text = text.replace("\u2019", "'").replace("`", "'")
    text = re.sub(r"[\u201c\u201d]", '"', text)
    text = re.sub(r"[\u2018\u2019]", "'", text)
    return re.sub(r"\s+", " ", text).strip()


# -----------------------------------------------------------------------------
# INDIA STATES
# -----------------------------------------------------------------------------
INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
    "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand",
    "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur",
    "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal",
    "Jammu and Kashmir", "Ladakh", "Delhi", "Puducherry", "Chandigarh",
    "Andaman and Nicobar Islands", "Lakshadweep"
]

STATE_SET = set(s.lower() for s in INDIAN_STATES)

# -----------------------------------------------------------------------------
# INDIA CITY/REGION MAP
# -----------------------------------------------------------------------------
INDIAN_CITIES = {
    # Gujarat
    "ahmedabad": "Gujarat", "surat": "Gujarat", "vadodara": "Gujarat",
    "rajkot": "Gujarat", "bhuj": "Gujarat", "navsari": "Gujarat",
    "jamnagar": "Gujarat", "gandhinagar": "Gujarat", "valsad": "Gujarat",

    # Rajasthan
    "jaipur": "Rajasthan", "sikar": "Rajasthan", "udaipur": "Rajasthan",
    "jodhpur": "Rajasthan", "ajmer": "Rajasthan", "kota": "Rajasthan",
    "bikaner": "Rajasthan", "alwar": "Rajasthan",

    # Madhya Pradesh
    "bhopal": "Madhya Pradesh", "indore": "Madhya Pradesh",
    "jabalpur": "Madhya Pradesh", "gwalior": "Madhya Pradesh",
    "rewa": "Madhya Pradesh", "satna": "Madhya Pradesh",
    "meghnagar": "Madhya Pradesh","berwani" : "Madhya Pradesh",

    # Uttar Pradesh
    "lucknow": "Uttar Pradesh", "kanpur": "Uttar Pradesh",
    "agra": "Uttar Pradesh", "varanasi": "Uttar Pradesh",
    "prayagraj": "Uttar Pradesh", "meerut": "Uttar Pradesh",
    "noida": "Uttar Pradesh", "ghaziabad": "Uttar Pradesh",

    # Bihar
    "patna": "Bihar", "gaya": "Bihar", "muzaffarpur": "Bihar",

    # Jharkhand
    "ranchi": "Jharkhand", "dhanbad": "Jharkhand",
    "jamshedpur": "Jharkhand", "ramgarh": "Jharkhand",

    # Maharashtra
    "mumbai": "Maharashtra", "pune": "Maharashtra",
    "nagpur": "Maharashtra", "nashik": "Maharashtra",
    "thane": "Maharashtra", "aurangabad": "Maharashtra",
    "amravati": "Maharashtra", "solapur": "Maharashtra",

    # Tamil Nadu
    "chennai": "Tamil Nadu", "coimbatore": "Tamil Nadu",
    "madurai": "Tamil Nadu", "salem": "Tamil Nadu",
    "tirunelveli": "Tamil Nadu", "trichy": "Tamil Nadu",
    "tiruchirappalli": "Tamil Nadu", "thoothukudi": "Tamil Nadu",

    # Telangana
    "hyderabad": "Telangana", "warangal": "Telangana",
    "karimnagar": "Telangana", "nizamabad": "Telangana",
    "rajendranagar": "Telangana", "nirmal": "Telangana",

    # Karnataka
    "bengaluru": "Karnataka", "bangalore": "Karnataka", "mysuru": "Karnataka",

    # Kerala
    "kochi": "Kerala", "kozhikode": "Kerala",
    "thiruvananthapuram": "Kerala", "trivandrum": "Kerala",
    "palakkad": "Kerala", "wayanad": "Kerala", "thrissur": "Kerala",

    # West Bengal
    "kolkata": "West Bengal", "siliguri": "West Bengal",
    "darjeeling": "West Bengal", "howrah": "West Bengal",

    # Assam
    "guwahati": "Assam", "dibrugarh": "Assam",
    "jorhat": "Assam", "silchar": "Assam", "tezpur": "Assam",

    # Odisha
    "bhubaneswar": "Odisha", "cuttack": "Odisha",
    "puri": "Odisha", "sambalpur": "Odisha", "rourkela": "Odisha",

    # Chhattisgarh
    "raipur": "Chhattisgarh", "bilaspur": "Chhattisgarh",

    # Uttarakhand
    "dehradun": "Uttarakhand", "haridwar": "Uttarakhand",
    "nainital": "Uttarakhand", "uttarkashi": "Uttarakhand",
    "chamoli": "Uttarakhand", "rudraprayag": "Uttarakhand",
    "kedarnath": "Uttarakhand", "badrinath": "Uttarakhand",
    "bageshwar": "Uttarakhand", "pithoragarh": "Uttarakhand",

    # Himachal Pradesh
    "shimla": "Himachal Pradesh", "manali": "Himachal Pradesh",
    "kullu": "Himachal Pradesh", "mandi": "Himachal Pradesh",
    "dharamsala": "Himachal Pradesh", "sundernagar": "Himachal Pradesh",

    # Jammu & Kashmir
    "srinagar": "Jammu and Kashmir", "baramulla": "Jammu and Kashmir",
    "anantnag": "Jammu and Kashmir", "jammu": "Jammu and Kashmir",
    "ramban": "Jammu and Kashmir", "mehar": "Jammu and Kashmir",
    "uri": "Jammu and Kashmir", "sonamarg": "Jammu and Kashmir",
    "zojila": "Jammu and Kashmir", "kashmir": "Jammu and Kashmir",

    # Ladakh
    "leh": "Ladakh", "kargil": "Ladakh",

    # Arunachal Pradesh
    "itanagar": "Arunachal Pradesh", "tawang": "Arunachal Pradesh",
    "west kameng": "Arunachal Pradesh", "east kameng": "Arunachal Pradesh",
    "dibang valley": "Arunachal Pradesh", "pasighat": "Arunachal Pradesh",
    "yingkiong": "Arunachal Pradesh", "arunachal": "Arunachal Pradesh",

    # Northeast
    "imphal": "Manipur", "churachandpur": "Manipur", "kamjong": "Manipur",
    "shillong": "Meghalaya", "tura": "Meghalaya",
    "aizawl": "Mizoram",
    "kohima": "Nagaland", "dimapur": "Nagaland", "mokokchung": "Nagaland",
    "agartala": "Tripura",
    "gangtok": "Sikkim", "mangan": "Sikkim", "namchi": "Sikkim",

    # Andhra Pradesh
    "visakhapatnam": "Andhra Pradesh", "vijayawada": "Andhra Pradesh",
    "tirupati": "Andhra Pradesh", "guntur": "Andhra Pradesh",
    "kakinada": "Andhra Pradesh", "nellore": "Andhra Pradesh",
    "andhra": "Andhra Pradesh", "vizag": "Andhra Pradesh",
    "paderu": "Andhra Pradesh",

    # Delhi
    "delhi": "Delhi", "new delhi": "Delhi",

    # Andaman
    "port blair": "Andaman and Nicobar Islands",
    "andaman": "Andaman and Nicobar Islands",
    "little andaman": "Andaman and Nicobar Islands",

    # State names as locations
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

# -----------------------------------------------------------------------------
# FOREIGN LOCATIONS
# -----------------------------------------------------------------------------
FOREIGN_KEYWORDS = {
    "egypt", "cairo", "alexandria", "giza",
    "usa", "united states", "america", "new york", "california", "texas",
    "uk", "united kingdom", "england", "london",
    "france", "paris", "germany", "berlin", "italy", "rome", "spain", "madrid",
    "australia", "sydney", "melbourne", "canada", "toronto", "vancouver",
    "brazil", "mexico", "argentina", "japan", "tokyo", "china", "beijing",
    "russia", "moscow", "ukraine", "turkey", "ankara", "istanbul",
    "iran", "iraq", "israel", "gaza", "pakistan", "lahore", "karachi", "islamabad",
    "nepal", "kathmandu", "bangladesh", "dhaka", "sri lanka", "colombo",
    "myanmar", "thailand", "indonesia", "jakarta", "philippines", "vietnam",
    "afghanistan", "uae", "dubai", "saudi arabia", "oman", "kuwait", "qatar",
    "vanuatu", "tonga", "fiji", "south pacific", "new zealand"
}

FOREIGN_PATTERNS = [
    r"\begypt\b", r"\bcairo\b", r"\balexandria\b", r"\bgiza\b",
    r"\busa\b", r"\bunited states\b", r"\baustralia\b",
    r"\bvanuatu\b", r"\bnew zealand\b", r"\bcanada\b",
    r"\bjapan\b", r"\bchina\b", r"\bpakistan\b",
    r"\bindian ocean\b", r"\bsouth pacific\b",
    r"\bwestern australia\b", r"\btonga\b", r"\bfiji\b",
    r"\bindonesia\b", r"\bphilippines\b", r"\bthailand\b",
    r"\bmyanmar\b", r"\bsri lanka\b", r"\bbangladesh\b",
    r"\belk park\b", r"\bhernando county\b", r"\bsan diego\b",
    r"\b\w[\w\s]+ county\b", r"\bgfz\b", r"\bturkey\b",
    r"\bnepal\b", r"\biran\b", r"\biraq\b", r"\bisrael\b",
    r"\buk\b", r"\bengland\b", r"\bfrance\b", r"\bgermany\b"
]

# -----------------------------------------------------------------------------
# Noise / fake location skip list
# -----------------------------------------------------------------------------
SKIP_AS_LOCATION = {
    "tribune india", "punjab kesari", "hindustan times",
    "times of india", "india today", "the hindu",
    "ai", "ml", "iot", "api",
    "northeast", "north east", "northwest", "north west",
    "southeast", "south east", "southwest", "south west",
    "north india", "south india", "east india", "west india",
    "central india", "northern india", "southern india",
    "himalaya", "himalayas", "western ghats", "eastern ghats",
    "indian ocean", "bay of bengal", "arabian sea",
    "pacific ocean", "atlantic ocean",
    "kolkata time", "ist",
    "gfz", "hits", "hit", "area", "region", "zone", "district",
    "block", "sector", "ward",
}

TIMEZONE_PATTERN = re.compile(r'\([A-Za-z\s]+time\)', re.IGNORECASE)

# -----------------------------------------------------------------------------
# Optional helper if any old code still imports it
# -----------------------------------------------------------------------------
def is_india_news(text: str) -> bool:
    text = normalize_text(text).lower()

    for pat in FOREIGN_PATTERNS:
        if re.search(pat, text):
            return False

    if re.search(r"\bindia\b", text) and "indian ocean" not in text:
        return True

    for city in INDIAN_CITIES:
        if re.search(rf"\b{re.escape(city)}\b", text):
            return True

    for state in STATE_SET:
        if re.search(rf"\b{re.escape(state)}\b", text):
            return True

    return False


# -----------------------------------------------------------------------------
# Main extractor
# -----------------------------------------------------------------------------
def extract_location(text: str):
    text = normalize_text(text)

    # Remove timezone references like "(Kolkata time)"
    text_clean = TIMEZONE_PATTERN.sub("", text).strip()
    tl = text_clean.lower()

    # 0) HARD FOREIGN REJECTION FIRST
    for pat in FOREIGN_PATTERNS:
        if re.search(pat, tl):
            return None, None, None

    # 1) Pattern: "Rajasthan's Sikar"
    match = re.search(
        r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)['\u2019]s\s+([A-Z][a-z]+)",
        text_clean
    )
    if match:
        state_name = match.group(1)
        loc = match.group(2)
        if state_name.lower() in STATE_SET and loc.lower() not in SKIP_AS_LOCATION:
            return loc, state_name, "India"

    # 2) Pattern: "Sikar, Rajasthan"
    match = re.search(
        r"\b([A-Z][a-z]+)\s*,\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        text_clean
    )
    if match:
        loc = match.group(1)
        state_name = match.group(2)
        if state_name.lower() in STATE_SET and loc.lower() not in SKIP_AS_LOCATION:
            return loc, state_name, "India"

    # 3) Pattern: "Gujarat Navsari district"
    for state_name in sorted(INDIAN_STATES, key=len, reverse=True):
        sp = re.escape(state_name)
        m = re.search(
            rf"\b{sp}\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:district|city|region)\b",
            text_clean, flags=re.IGNORECASE
        )
        if m:
            loc = m.group(1).strip()
            if loc.lower() not in SKIP_AS_LOCATION:
                return loc, state_name, "India"

    # 4) Direct city dictionary
    for city, state_name in sorted(INDIAN_CITIES.items(), key=lambda x: len(x[0]), reverse=True):
        if re.search(rf"\b{re.escape(city)}\b", tl):
            return city.title(), state_name, "India"

    # 5) Direct state detection
    for state_name in sorted(INDIAN_STATES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(state_name.lower())}\b", tl):
            return None, state_name, "India"

    # 6) spaCy fallback — STRICT Indian-only acceptance
    doc = nlp(text_clean)
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            name = ent.text.strip()
            nl = name.lower()

            if nl in SKIP_AS_LOCATION:
                continue

            if nl in FOREIGN_KEYWORDS:
                continue

            if any(re.search(pat, nl) for pat in FOREIGN_PATTERNS):
                continue

            if "cyclone" in nl or "ocean" in nl or "sea" in nl:
                continue

            if nl in INDIAN_CITIES:
                return name, INDIAN_CITIES[nl], "India"

            if nl in STATE_SET:
                return None, name, "India"

            # IMPORTANT:
            # DO NOT do:
            # if "india" in tl: return name, None, "India"
            # That was causing foreign/mixed articles to leak.

    return None, None, None