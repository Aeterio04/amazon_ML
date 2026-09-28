"""
normalization.py — Stage 0 Normalization Pipeline
Amazon ML Challenge 2026: Business Entity Resolution

Features:
1. Script detection & transliteration for Devanagari (and Indic) names & addresses (zero external APIs).
2. Legal suffix detection, standardization, and core name extraction (US, India, France).
3. DBA marker splitting ('dba', 'd/b/a', 'doing business as', 't/a').
4. US State abbreviation <-> full-name canonicalization.
5. Redacted house number detection ('##8' style) flagged as missing (not matched literally).
6. Country-aware address parsing with a robust, generic fallback for France & unseen regions.
"""

import re
import unicodedata
from typing import Dict, Tuple, Optional, List, Set

# --- DEVANAGARI & INDIC TRANSLITERATION TABLE ---

DEVA_CONSONANTS: Dict[str, str] = {
    '\u0915': 'k', '\u0916': 'kh', '\u0917': 'g', '\u0918': 'gh', '\u0919': 'ng',
    '\u091a': 'ch', '\u091b': 'chh', '\u091c': 'j', '\u091d': 'jh', '\u091e': 'ny',
    '\u091f': 't', '\u0920': 'th', '\u0921': 'd', '\u0922': 'dh', '\u0923': 'n',
    '\u0924': 't', '\u0925': 'th', '\u0926': 'd', '\u0927': 'dh', '\u0928': 'n',
    '\u092a': 'p', '\u092b': 'ph', '\u092c': 'b', '\u092d': 'bh', '\u092e': 'm',
    '\u092f': 'y', '\u0930': 'r', '\u0932': 'l', '\u0933': 'l', '\u0935': 'v',
    '\u0936': 'sh', '\u0937': 'sh', '\u0938': 's', '\u0939': 'h',
    '\u0958': 'q', '\u0959': 'kh', '\u095a': 'gh', '\u095b': 'z',
    '\u095c': 'd', '\u095d': 'dh', '\u095e': 'f', '\u095f': 'y'
}

DEVA_VOWELS: Dict[str, str] = {
    '\u0905': 'a', '\u0906': 'aa', '\u0907': 'i', '\u0908': 'ee', '\u0909': 'u',
    '\u090a': 'oo', '\u090b': 'ri', '\u090e': 'e', '\u090f': 'e', '\u0910': 'ai',
    '\u0911': 'o', '\u0912': 'o', '\u0913': 'o', '\u0914': 'au'
}

DEVA_MATRAS: Dict[str, str] = {
    '\u093e': 'a', '\u093f': 'i', '\u0940': 'ee', '\u0941': 'u', '\u0942': 'oo',
    '\u0943': 'ri', '\u0947': 'e', '\u0948': 'ai', '\u0949': 'o', '\u094a': 'o',
    '\u094b': 'o', '\u094c': 'au', '\u0946': 'e', '\u0945': 'e'
}

DEVA_VIRAMA = '\u094d'
DEVA_ANUSVARA = '\u0902'
DEVA_CANDRABINDU = '\u0901'
DEVA_VISARGA = '\u0903'
DEVA_NUKTA = '\u093c'

# Common Devanagari corporate/legal words to map directly to standard English tokens
DEVA_WORD_MAP: Dict[str, str] = {
    'प्राइवेट': 'private', 'लिमिटेड': 'limited', 'प्रा.': 'pvt', 'प्रा': 'pvt',
    'लि.': 'ltd', 'लि': 'ltd', 'एलएलपी': 'llp', 'कंपनी': 'company',
    'कॉर्पोरेशन': 'corporation', 'एंटरप्राइजेज': 'enterprises', 'इंडस्ट्रीज': 'industries',
    'सॉल्यूशंस': 'solutions', 'सर्विसेज': 'services', 'मार्केटिंग': 'marketing',
    'ट्रेडर्स': 'traders', 'प्रॉपर्टीज': 'properties', 'कंस्ट्रक्शंस': 'constructions',
    'टेक्नोलॉजीज': 'technologies', 'इन्वेस्टमेंट': 'investment', 'वेंचर्स': 'ventures',
    'डेवलपर्स': 'developers', 'इंटरनेशनल': 'international', 'इंडिया': 'india',
    'भारत': 'bharat', 'महाराष्ट्र': 'maharashtra', 'दिल्ली': 'delhi',
    'कर्नाटक': 'karnataka', 'गुजरात': 'gujarat', 'राजस्थान': 'rajasthan',
    'हरियाणा': 'haryana', 'पंजाब': 'punjab', 'बंगाल': 'bengal'
}

DEVA_REGEX = re.compile(r'[\u0900-\u097F]')

# Common Kannada state / city terms seen in Indian addresses (e.g. ಕರ್ನಾಟಕ)
INDIC_SCRIPT_MAP: Dict[str, str] = {
    'ಕರ್ನಾಟಕ': 'karnataka', 'ಬೆಂಗಳೂರು': 'bangalore', 'ಮಹಾರಾಷ್ಟ್ರ': 'maharashtra'
}

# --- US STATES CANONICALIZATION ---

US_STATES: Dict[str, str] = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR",
    "CALIFORNIA": "CA", "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE",
    "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI", "IDAHO": "ID",
    "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS",
    "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD",
    "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS",
    "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEW YORK": "NY",
    "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK",
    "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT",
    "VERMONT": "VT", "VIRGINIA": "VA", "WASHINGTON": "WA", "WEST VIRGINIA": "WV",
    "WISCONSIN": "WI", "WYOMING": "WY", "DISTRICT OF COLUMBIA": "DC"
}
# Invert for lookup
US_STATE_CODES: Set[str] = set(US_STATES.values())
# Regex pattern for full state names (sorted longest first to avoid partial matches)
US_STATE_NAME_PATTERNS = [(re.compile(r'\b' + re.escape(name) + r'\b', re.IGNORECASE), code)
                          for name, code in sorted(US_STATES.items(), key=lambda x: -len(x[0]))]

# --- LEGAL SUFFIXES (US, INDIA, FRANCE) ---

LEGAL_SUFFIXES_REGEX = re.compile(
    r'\b(private limited|pvt ltd|pvt\. ltd\.|pvt|ltd|limited|llc|pllc|llp|inc|incorporated|'
    r'corp|corporation|co|company|sarl|sas|sci|sa|eurl|snc|gmbh|public limited|'
    r'pub ltd|plc)\b',
    re.IGNORECASE
)

# DBA Markers
DBA_REGEX = re.compile(r'\b(d/?b/?a|doing business as|trading as|t/?a)\b', re.IGNORECASE)

# Redacted house number pattern: e.g. ##8, ***4, ##, XX1
REDACTED_NUM_REGEX = re.compile(r'(^[#\*X]{2,}\d*|\b[#\*X]{2,}\d*\b)')

# Landmark cues in Indian / international addresses
LANDMARK_REGEX = re.compile(r'\b(near|opp|opposite|behind|beside|adjacent to|next to|above|below)\b', re.IGNORECASE)


def transliterate_devanagari(text: str) -> str:
    """
    Deterministically transliterate Devanagari text to Latin phonetic text.
    Handles schwa deletion, matras, virama, and common corporate vocabulary.
    """
    if not text or not DEVA_REGEX.search(text):
        return text

    # Check for direct multi-word Indic matches (e.g. Kannada state names)
    for k, v in INDIC_SCRIPT_MAP.items():
        if k in text:
            text = text.replace(k, v)

    words = text.split()
    out_words = []
    for w in words:
        clean_w = w.strip('.,-–/()')
        if clean_w in DEVA_WORD_MAP:
            out_words.append(DEVA_WORD_MAP[clean_w])
            continue

        chars = list(w)
        res = []
        n = len(chars)
        for i in range(n):
            c = chars[i]
            if c in DEVA_CONSONANTS:
                res.append(DEVA_CONSONANTS[c])
                next_c = chars[i + 1] if i + 1 < n else None
                if next_c is None or (next_c not in DEVA_MATRAS and next_c != DEVA_VIRAMA):
                    # Inherent 'a', except at word boundary (schwa deletion in Hindi)
                    if i + 1 < n and (chars[i + 1] in DEVA_CONSONANTS or chars[i + 1] in DEVA_VOWELS):
                        res.append('a')
            elif c in DEVA_VOWELS:
                res.append(DEVA_VOWELS[c])
            elif c in DEVA_MATRAS:
                res.append(DEVA_MATRAS[c])
            elif c in (DEVA_ANUSVARA, DEVA_CANDRABINDU):
                res.append('n')
            elif c == DEVA_VISARGA:
                res.append('h')
            elif c == DEVA_VIRAMA or c == DEVA_NUKTA:
                pass
            else:
                res.append(c)
        out_words.append(''.join(res))
    return ' '.join(out_words)


def strip_accents(text: str) -> str:
    """
    Normalize unicode (NFKD) and strip accents (e.g. 'Bordeaux, Nouvelle-Aquitaine' / 'école' -> 'ecole').
    """
    if not text:
        return ""
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def clean_text_basic(text: str) -> str:
    """
    Lowercase, strip accents, collapse punctuation, normalize whitespace.
    """
    if not text or text == "nan":
        return ""
    text = strip_accents(text).lower()
    # Replace '&' with ' and '
    text = text.replace('&', ' and ')
    # Replace all punctuation and symbols with space (preserves all alphanumeric & whitespace)
    text = re.sub(r'[^\w\s]', ' ', text)
    # Collapse whitespace
    return re.sub(r'\s+', ' ', text).strip()


def extract_dba(name: str) -> Tuple[str, str]:
    """
    Split name on DBA markers into (primary_name, dba_name).
    If no DBA marker, returns (name, "").
    """
    match = DBA_REGEX.search(name)
    if match:
        primary = name[:match.start()].strip()
        dba = name[match.end():].strip()
        return primary, dba
    return name, ""


def extract_legal_form(name: str) -> Tuple[str, str]:
    """
    Extracts core business name and legal suffix.
    E.g. "Acme Robotics Inc." -> ("acme robotics", "inc")
    E.g. "LLC Moncada Learning Center" -> ("moncada learning center", "llc")
    """
    cleaned = clean_text_basic(name)
    # Check leading legal form (e.g. "llc moncada...")
    leading_match = re.match(r'^(pvt ltd|private limited|llc|llp|inc|corp|ltd|co|sarl|sas|sci|sa)\s+', cleaned)
    if leading_match:
        suffix = leading_match.group(1)
        core = cleaned[leading_match.end():].strip()
        return core, suffix

    # Check trailing legal form (e.g. "... inc")
    trailing_match = re.search(r'\s+(pvt ltd|private limited|llc|llp|inc|incorporated|corp|corporation|ltd|limited|co|company|sarl|sas|sci|sa|eurl|snc|gmbh)$', cleaned)
    if trailing_match:
        suffix = trailing_match.group(1)
        core = cleaned[:trailing_match.start()].strip()
        return core, suffix

    # General regex search
    matches = list(LEGAL_SUFFIXES_REGEX.finditer(cleaned))
    if matches:
        last_match = matches[-1]
        suffix = last_match.group(0)
        # remove it from core
        core = cleaned[:last_match.start()] + ' ' + cleaned[last_match.end():]
        core = re.sub(r'\s+', ' ', core).strip()
        return core, suffix

    return cleaned, ""


def normalize_business_name(raw_name: str) -> Dict[str, str]:
    """
    Comprehensive name normalization:
    - Transliterates Devanagari/Indic scripts.
    - Handles DBA markers.
    - Extracts name_core (stripped of legal forms and noise).
    - Preserves name_translit and legal_suffix.
    """
    if not raw_name or str(raw_name).lower() == 'nan':
        return {"name_raw": "", "name_core": "", "name_translit": "", "name_dba": "", "legal_suffix": ""}

    raw_str = str(raw_name).strip()
    # 1. Transliterate Devanagari/Indic
    translit = transliterate_devanagari(raw_str)

    # 2. Check for DBA
    primary, dba = extract_dba(translit)

    # 3. Extract legal suffix and name core
    core, suffix = extract_legal_form(primary)
    dba_core, _ = extract_legal_form(dba) if dba else ("", "")

    return {
        "name_raw": raw_str,
        "name_translit": translit,
        "name_core": core,
        "name_dba": dba_core,
        "legal_suffix": suffix
    }


def parse_and_canonicalize_address(raw_addr: str, country: str = "US") -> Dict[str, str]:
    """
    Country-aware address parsing with generic fallback:
    - US State canonicalization (full name <-> 2-letter code)
    - House number extraction & ##8 redaction flag
    - Landmark extraction
    - Generic normalization for France and unseen regions
    """
    if not raw_addr or str(raw_addr).lower() == 'nan':
        return {
            "addr_raw": "",
            "addr_core": "",
            "house_number": "",
            "is_house_redacted": "0",
            "state_code": "",
            "landmark": "",
            "country": country
        }

    raw_str = str(raw_addr).strip()
    # Transliterate any Indic text in address (e.g. Karnataka in Kannada or Delhi in Devanagari)
    addr_translit = transliterate_devanagari(raw_str)

    # 1. Check for redacted house number (e.g. ##8, ***4)
    is_house_redacted = "0"
    house_num = ""
    redacted_match = REDACTED_NUM_REGEX.search(addr_translit)
    if redacted_match:
        is_house_redacted = "1"
        # Remove the redacted token so it doesn't pollute string matching
        addr_clean = addr_translit[:redacted_match.start()] + ' ' + addr_translit[redacted_match.end():]
    else:
        addr_clean = addr_translit
        # Look for explicit door/unit/plot number with digits:
        door_match = re.search(r'\b(?:door\s*no\.?|h\.?\s*no\.?|plot\s*no\.?|kh\s*no\.?|unit\s*no\.?|no\.)\s*([0-9]+[a-zA-Z0-9/\-]*)\b', addr_clean, re.IGNORECASE)
        if door_match:
            house_num = door_match.group(1).lower()
        else:
            # Look for leading house/street number (e.g. "1795 Westchester...", "175 Boulevard...", "18 RUE...")
            leading_num = re.search(r'^\s*([0-9]+[a-zA-Z]?)\b', addr_clean)
            if leading_num:
                house_num = leading_num.group(1).lower()
            else:
                # Look for standalone digits near beginning of address (within first 30 chars)
                first_num = re.search(r'\b([0-9]+[a-zA-Z]?)\b', addr_clean[:35])
                if first_num:
                    house_num = first_num.group(1).lower()

    # 2. Extract landmark (especially for India)
    landmark = ""
    lm_match = LANDMARK_REGEX.search(addr_clean)
    if lm_match:
        landmark = addr_clean[lm_match.start():].strip()
        addr_core_raw = addr_clean[:lm_match.start()].strip()
    else:
        addr_core_raw = addr_clean

    # 3. State canonicalization (US and common India states)
    state_code = ""
    # Check US state full names first
    for pat, code in US_STATE_NAME_PATTERNS:
        if pat.search(addr_core_raw):
            state_code = code
            # Replace full state name with standard code
            addr_core_raw = pat.sub(f' {code} ', addr_core_raw)
            break

    if not state_code:
        # Check 2-letter state codes
        tokens = re.findall(r'\b[A-Za-z]{2}\b', addr_core_raw)
        for t in tokens:
            upper_t = t.upper()
            if upper_t in US_STATE_CODES:
                state_code = upper_t
                break

    # 4. Standardize common street abbreviations
    # US & international: rd -> road, st -> street, ave/av -> avenue, blvd/bd -> boulevard, etc.
    # France: r. / rue, bd / boulevard, av. / avenue
    addr_core = clean_text_basic(addr_core_raw)
    addr_core = re.sub(r'\b(rd)\b', 'road', addr_core)
    addr_core = re.sub(r'\b(st)\b', 'street', addr_core)
    addr_core = re.sub(r'\b(ave|av)\b', 'avenue', addr_core)
    addr_core = re.sub(r'\b(blvd|bd)\b', 'boulevard', addr_core)
    addr_core = re.sub(r'\b(ct)\b', 'court', addr_core)
    addr_core = re.sub(r'\b(dr)\b', 'drive', addr_core)
    addr_core = re.sub(r'\b(ln)\b', 'lane', addr_core)
    addr_core = re.sub(r'\b(hwy)\b', 'highway', addr_core)
    addr_core = re.sub(r'\b(pkwy)\b', 'parkway', addr_core)
    addr_core = re.sub(r'\b(apt|apartment)\b', 'apt', addr_core)
    addr_core = re.sub(r'\b(ste|suite)\b', 'suite', addr_core)
    addr_core = re.sub(r'\s+', ' ', addr_core).strip()

    return {
        "addr_raw": raw_str,
        "addr_core": addr_core,
        "house_number": house_num,
        "is_house_redacted": is_house_redacted,
        "state_code": state_code,
        "landmark": clean_text_basic(landmark),
        "country": country if country and country != 'nan' else 'UNKNOWN'
    }


def normalize_record(
    entity_id: str,
    raw_name: str,
    raw_addr: str,
    country: str
) -> Dict[str, str]:
    """
    Run complete normalization for a single business entity record.
    Returns dictionary with all clean normalized attributes.
    """
    name_norm = normalize_business_name(raw_name)
    addr_norm = parse_and_canonicalize_address(raw_addr, country)

    return {
        "entity_id": str(entity_id).strip(),
        "name_raw": name_norm["name_raw"],
        "name_core": name_norm["name_core"],
        "name_translit": name_norm["name_translit"],
        "name_dba": name_norm["name_dba"],
        "legal_suffix": name_norm["legal_suffix"],
        "addr_raw": addr_norm["addr_raw"],
        "addr_core": addr_norm["addr_core"],
        "house_number": addr_norm["house_number"],
        "is_house_redacted": addr_norm["is_house_redacted"],
        "state_code": addr_norm["state_code"],
        "landmark": addr_norm["landmark"],
        "country": addr_norm["country"],
    }
