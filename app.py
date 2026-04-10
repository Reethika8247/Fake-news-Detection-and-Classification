import os, re, html, traceback, json, time, hashlib, socket
from collections import OrderedDict
from threading import Lock

os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

import torch
import numpy as np
import requests as req_lib
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import quote_plus
from xml.etree import ElementTree
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
from transformers import BertTokenizer, BertForSequenceClassification

try:
    import google.generativeai as genai
    GEMINI_SDK = True
except ImportError:
    GEMINI_SDK = False
    print("WARNING: google-generativeai not installed.")

try:
    import spacy
    nlp_ner   = spacy.load("en_core_web_sm")
    USE_SPACY = True
    print("spaCy NER loaded.")
except Exception:
    USE_SPACY = False
    print("spaCy not found -- regex NER fallback.")

app      = Flask(__name__)
CORS(app)
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
DATA_DIR          = os.path.join(BASE_DIR, "data")
DATA_FILE         = os.path.join(DATA_DIR, "truthlens_data.json")

def _load_json(filepath):
    if not os.path.isfile(filepath):
        print(f"[DataLoader] WARNING: {filepath} not found — using empty defaults")
        return {}
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)

_data        = _load_json(DATA_FILE)
_celeb_data  = _data
_sports_data = _data
_topic_data  = _data

def _merge_data():
    import sys
    d   = _data
    mod = sys.modules[__name__]
    if not d: return
    if "celeb_map"           in d: mod.CELEB_DATA          = d["celeb_map"]
    if "known_living_people" in d: mod.KNOWN_LIVING_PEOPLE = set(d["known_living_people"])
    if "last_name_map"       in d: mod.LAST_NAME_MAP        = d["last_name_map"]
    if "celeb_only_names"    in d: mod._CELEB_ONLY_NAMES    = set(d["celeb_only_names"])
    if "celeb_map" in d:
        mod.CELEB_FULL_NAMES = set(d["celeb_map"].values())
        mod.CELEB_KEYWORDS   = (
            {v.split()[0].lower() for v in d["celeb_map"].values()} |
            {v.split()[-1].lower() for v in d["celeb_map"].values()} |
            {"bollywood","tollywood","kollywood","wedding","married","marriage"}
        )
    if "cricket_teams"     in d: mod.CRICKET_TEAMS     = d["cricket_teams"]
    if "country_alias_map" in d: mod.COUNTRY_ALIAS_MAP  = d["country_alias_map"]
    if "ipl_teams"         in d: mod.IPL_TEAMS          = d["ipl_teams"]
    if "football_teams"    in d: mod.FOOTBALL_TEAMS     = d["football_teams"]
    if "cricket_teams" in d and "ipl_teams" in d and "football_teams" in d:
        mod.ALL_SPORTS_TEAMS = d["cricket_teams"] + d["ipl_teams"] + d["football_teams"]
    if "sports_winners" in d:
        flat = {}
        for trn, yr_map in d["sports_winners"].items():
            for yr, winner in yr_map.items():
                flat[(trn, str(yr))] = winner
        mod.SPORTS_WINNERS = flat
    if "political_countries" in d: mod.POLITICAL_COUNTRIES = d["political_countries"]
    if "country_names"       in d: mod.COUNTRY_NAMES       = set(d["country_names"])
    if "global_indicators"       in d: mod.GLOBAL_INDICATORS       = set(d["global_indicators"])
    if "local_indicators"        in d: mod.LOCAL_INDICATORS        = set(d["local_indicators"])
    if "credible_source_signals" in d: mod.CREDIBLE_SOURCE_SIGNALS = set(d["credible_source_signals"])
    if "death_hoax_phrases"      in d: mod.DEATH_HOAX_PHRASES      = set(d["death_hoax_phrases"])
    if "tribute_phrases"         in d: mod.TRIBUTE_PHRASES         = set(d["tribute_phrases"])
    if "topic_confirm_words"     in d:
        mod.TOPIC_CONFIRM_WORDS = {k: set(v) for k, v in d["topic_confirm_words"].items()}
    if "topic_deny_words"        in d:
        mod.TOPIC_DENY_WORDS    = {k: set(v) for k, v in d["topic_deny_words"].items()}
    if "topic_phrases"  in d: mod.TOPIC_PHRASES  = d["topic_phrases"]
    if "topic_single"   in d: mod.TOPIC_SINGLE   = d["topic_single"]
    if "known_facts"    in d:
        mod.KNOWN_FACTS = [(set(kw), v, e) for kw, v, e in d["known_facts"]]
    if "science_regex" in d:
        mod.SCIENCE_REGEX = [(row[0], row[1], row[2]) for row in d["science_regex"]]
    if "not_names"            in d: mod.NOT_NAMES            = set(d["not_names"])
    if "role_patterns"        in d: mod.ROLE_PATTERNS        = [tuple(p) for p in d["role_patterns"]]
    if "role_detectors"       in d: mod.ROLE_DETECTORS       = [
        (row[0], row[1], row[2], row[3]) for row in d["role_detectors"]
    ]
    if "nli_model_candidates"     in d: mod.NLI_MODEL_CANDIDATES     = d["nli_model_candidates"]
    if "sensationalist_patterns"  in d: mod.SENSATIONALIST_PATTERNS  = d["sensationalist_patterns"]
    if "negation_patterns"        in d: mod.NEGATION_PATTERNS        = d["negation_patterns"]
    if "relation_keywords"        in d: mod.RELATION_KEYWORDS        = {k: set(v) for k, v in d["relation_keywords"].items()}
    if "celebrity_personal_words" in d: mod.CELEBRITY_PERSONAL_WORDS = set(d["celebrity_personal_words"])
    if "political_indicators"     in d: mod._POLITICAL_INDICATORS    = set(d["political_indicators"])
    if "uncertain_default_topics" in d: mod.UNCERTAIN_DEFAULT_TOPICS = set(d["uncertain_default_topics"])
    if "preserve_words"           in d: mod.PRESERVE_WORDS           = set(d["preserve_words"])
    if "generic_words"            in d: mod.GENERIC_WORDS            = set(d["generic_words"])

@app.after_request
def cors_headers(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok", "bert_loaded": BERT_LOADED,
                    "nli_backend": NLI_BACKEND,
                    "csv_fallback": CSV_FALLBACK_LOADED,
                    "cache_size": len(_cache)})

# =============================================================
# API KEYS
# =============================================================
NEWSAPI_KEY       = "c57c22340afb46419a1aa98323af72ef"
FACTCHECK_API_KEY = "566413c9c5d71988a1b192d5e8a37fa75f4c6422"
GEMINI_API_KEY    = "AIzaSyD8cnWiSaSHiiP_78IzuabRX3-od0nCaSM"

# =============================================================
# CONSTANTS
# =============================================================
FETCH_TIMEOUT    = 4
EVIDENCE_TIMEOUT = 12
MAX_EVIDENCE     = 12
NLI_BERT_TOP_N   = 5
MAX_WORKERS      = 16
NLI_MAX_CHARS    = 300
CACHE_TTL        = 1800
CACHE_MAX_SIZE   = 500
BERT_WEIGHT      = 0.45
EVIDENCE_WEIGHT  = 0.55
CSV_PATH         = os.path.join(BASE_DIR, "merged_news.csv")
BERT_MODEL_DIR   = os.path.join(BASE_DIR, "model", "fake_news_bert")
KB_PATH          = os.path.join(BASE_DIR, "knowledge_base.json")
ARTICLE_MAX_LENGTH = 256
CLAIM_MAX_LENGTH   = 128
PARTIALLY_TRUE     = "PARTIALLY_TRUE"
LABEL_0_MEANS = "FAKE"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9"
}

# =============================================================
# FIX: Safe CSV reader — tries multiple encodings
# Resolves: UnicodeDecodeError on merged_news.csv
# =============================================================
def _read_csv_safe(path, **kwargs):
    """Try reading CSV with utf-8, latin-1, cp1252 encodings in order."""
    import pandas as pd
    kwargs.setdefault("on_bad_lines", "skip")
    for enc in ("utf-8", "latin-1", "cp1252", "utf-8-sig"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
        except Exception:
            raise
    # Last resort: replace bad bytes
    return pd.read_csv(path, encoding="utf-8", errors="replace", **kwargs)

# =============================================================
# =============================================================
# DATA TABLES — loaded from data/truthlens_data.json via _merge_data()
# Minimal safe defaults so the app never crashes if JSON is missing.
# =============================================================
CELEB_DATA           = {}
CELEB_FULL_NAMES     = set()
CELEB_KEYWORDS       = {"bollywood","tollywood","kollywood","wedding","married","marriage"}
KNOWN_LIVING_PEOPLE  = set()
LAST_NAME_MAP        = {}
_CELEB_ONLY_NAMES    = set()
CRICKET_TEAMS        = []
COUNTRY_ALIAS_MAP    = {}
POLITICAL_COUNTRIES  = []
COUNTRY_NAMES        = set()
SPORTS_WINNERS       = {}
IPL_TEAMS            = []
FOOTBALL_TEAMS       = []
ALL_SPORTS_TEAMS     = []
GLOBAL_INDICATORS    = set()
LOCAL_INDICATORS     = set()
TOPIC_CONFIRM_WORDS  = {"general": {"confirmed","announced","officially","sources","reports"}}
TOPIC_DENY_WORDS     = {"general": {"false","hoax","rumor","not true","baseless","denied"}}
TOPIC_PHRASES        = []
TOPIC_SINGLE         = {}
CREDIBLE_SOURCE_SIGNALS = set()
DEATH_HOAX_PHRASES   = set()
TRIBUTE_PHRASES      = set()
KNOWN_FACTS          = []
SCIENCE_REGEX        = []   # loaded from science_regex key in truthlens_data.json
NOT_NAMES            = {"the","a","an","and","or","in","on","at","to","for","of","with","by"}
ROLE_PATTERNS        = []
ROLE_DETECTORS       = []
NLI_MODEL_CANDIDATES = [
    "cross-encoder/nli-deberta-v3-small",
    "typeform/distilbert-base-uncased-mnli",
    "facebook/bart-large-mnli",
]
SENSATIONALIST_PATTERNS  = []
NEGATION_PATTERNS        = [
    r"\bhas\s+not\b", r"\bhave\s+not\b", r"\bdid\s+not\b", r"\bis\s+not\b",
    r"\bare\s+not\b", r"\bwas\s+not\b", r"\bwere\s+not\b", r"\bhasn'?t\b",
    r"\bhaven'?t\b", r"\bdidn'?t\b", r"\bisn'?t\b", r"\baren'?t\b",
    r"\bwasn'?t\b", r"\bweren'?t\b", r"\bno\s+longer\b", r"\bnever\b",
    r"\bnot\b", r"\bcan'?t\b", r"\bcannot\b", r"\bwon'?t\b",
    r"\bdoesn'?t\b", r"\bdon'?t\b",
]
RELATION_KEYWORDS        = {}
CELEBRITY_PERSONAL_WORDS = set()
_POLITICAL_INDICATORS    = set()
UNCERTAIN_DEFAULT_TOPICS = set()
PRESERVE_WORDS           = set()
GENERIC_WORDS            = set()

_merge_data()
print(f"[DataLoader] Active — Celebs:{len(CELEB_DATA)} Sports:{len(SPORTS_WINNERS)} "
      f"Facts:{len(KNOWN_FACTS)} GlobalInd:{len(GLOBAL_INDICATORS)}")

NAME_EXTRACT_PATTERNS = [
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+(?:is|serves as|was named|was appointed|currently)\s+(?:the\s+)?(?:current\s+)?(?:Test\s+|ODI\s+|T20\s+)?captain",
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+\(captain\)",
    r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+is\s+(?:the\s+)?(?:current\s+)?(?:Prime Minister|President|Governor|Chief Justice)",
    r"(?:Prime Minister|President|Governor|Chief Justice)\s+(?:is\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})",
    r"^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\s+(?:is|was|has been|served)",
]

# =============================================================
# LOCAL KNOWLEDGE BASE
# =============================================================
def _load_knowledge_base(path: str) -> dict:
    if not os.path.isfile(path):
        print(f"[KB] knowledge_base.json not found — offline KB disabled")
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        flat = {}
        for cat, facts in raw.items():
            if cat == "_meta" or not isinstance(facts, dict): continue
            for claim_text, verdict in facts.items():
                key = re.sub(r"[^a-z0-9 ]", "", claim_text.lower().strip())
                key = re.sub(r"\s{2,}", " ", key).strip()
                flat[key] = verdict.upper()
        print(f"[KB] Loaded {len(flat)} facts from knowledge_base.json")
        return flat
    except Exception as e:
        print(f"[KB] Failed to load: {e}")
        return {}

_KB = _load_knowledge_base(KB_PATH)

def _ordered_word_similarity(s1, s2):
    w1, w2 = s1.split(), s2.split()
    if not w1 or not w2: return 0.0
    m, n = len(w1), len(w2)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i][j] = dp[i-1][j-1]+1 if w1[i-1]==w2[j-1] else max(dp[i-1][j], dp[i][j-1])
    return (2*dp[m][n]) / (m+n)

def _reverse_subject_object(text):
    words = text.split()
    PIVOTS = {"around","about","orbits","revolves","rotates","circles"}
    for i, w in enumerate(words):
        if w in PIVOTS and 0 < i < len(words)-1:
            return " ".join(words[i+1:] + [w] + words[:i])
    return " ".join(words[::-1]) if len(words) >= 2 else text

# _SCIENCE_REGEX and _BUILTIN_KW removed — all facts live in truthlens_data.json
# Add them under "known_facts" and "science_regex" keys in that file.

def _check_known_facts(claim, negated=False, positive_claim=None):
    """Check KNOWN_FACTS loaded from truthlens_data.json.
    Supports two fact types loaded via _merge_data():
      - SCIENCE_REGEX: order-sensitive regex patterns (key: 'science_regex' in JSON)
      - KNOWN_FACTS:   keyword-set facts (key: 'known_facts' in JSON)
    """
    cl = claim.lower()

    # Regex-based patterns loaded from JSON science_regex key
    for pat, verdict, explanation in SCIENCE_REGEX:
        if re.search(pat, cl):
            print(f"  [KnownFacts-Regex] matched '{pat[:40]}' -> {verdict}")
            if negated:
                flipped = {"REAL":"FAKE","FAKE":"REAL","PARTIALLY_TRUE":"PARTIALLY_TRUE"}
                return flipped.get(verdict, verdict)
            return verdict

    # Keyword-set facts loaded from JSON known_facts key
    cl_words = set(re.findall(r"[a-z0-9]+", cl))
    for keywords, verdict, explanation in KNOWN_FACTS:
        if keywords.issubset(cl_words):
            print(f"  [KnownFacts-KW] matched {keywords} -> {verdict}: {explanation[:60]}")
            if negated:
                flipped = {"REAL":"FAKE","FAKE":"REAL","PARTIALLY_TRUE":"PARTIALLY_TRUE"}
                return flipped.get(verdict, verdict)
            return verdict

    if negated and positive_claim:
        pos_words = set(re.findall(r"[a-z0-9]+", positive_claim.lower()))
        for keywords, verdict, explanation in KNOWN_FACTS:
            if keywords.issubset(pos_words):
                flipped = {"REAL":"FAKE","FAKE":"REAL","PARTIALLY_TRUE":"PARTIALLY_TRUE"}
                result = flipped.get(verdict, verdict)
                print(f"  [KnownFacts-KW] negated match {keywords} -> {result}")
                return result
    return None

def local_kb_lookup(claim):
    if not _KB: return None
    cl = re.sub(r"[^a-z0-9 ]","",claim.lower().strip())
    cl = re.sub(r"\s{2,}"," ",cl).strip()
    if cl in _KB: return _KB[cl]
    best_key, best_len = None, 0
    for key in _KB:
        if key in cl and len(key) > best_len and len(key) >= 15:
            best_key, best_len = key, len(key)
    if best_key: return _KB[best_key]
    cl_words = set(cl.split()) - {"the","a","an","is","are","was","were","of","in",
                                    "on","that","it","has","have","to","and","or","by"}
    if len(cl_words) >= 4:
        best_match, best_score = None, 0.0
        for key, verdict in _KB.items():
            kw = set(key.split())
            if not kw: continue
            if len(cl_words & kw)/len(cl_words) < 0.85 or len(key) < 12: continue
            ordered_sim  = _ordered_word_similarity(cl, key)
            cl_rev       = _reverse_subject_object(cl)
            reversed_sim = _ordered_word_similarity(cl_rev, key)
            if reversed_sim > ordered_sim and reversed_sim >= 0.75:
                flip = {"REAL":"FAKE","FAKE":"REAL","PARTIALLY_TRUE":"PARTIALLY_TRUE"}
                score = reversed_sim + len(key)*0.001
                if score > best_score:
                    best_score = score; best_match = (key, flip.get(verdict, verdict), "reversed")
            elif ordered_sim >= 0.70:
                score = ordered_sim + len(key)*0.001
                if score > best_score:
                    best_score = score; best_match = (key, verdict, "ordered")
        if best_match:
            key, verdict, mt = best_match
            print(f"  [KB] {mt.capitalize()} ({best_score:.2f}): '{key[:60]}' -> {verdict}")
            return verdict
    return None

# =============================================================
# LRU CACHE
# =============================================================
class TTLCache:
    def __init__(self, max_size=500, ttl=1800):
        self._cache = OrderedDict(); self._lock = Lock()
        self.max_size = max_size; self.ttl = ttl
    def _key(self, c): return hashlib.md5(c.strip().lower().encode()).hexdigest()
    def get(self, c):
        k = self._key(c)
        with self._lock:
            if k not in self._cache: return None
            e, ts = self._cache[k]
            if time.time()-ts > self.ttl: del self._cache[k]; return None
            self._cache.move_to_end(k); return e
    def set(self, c, v):
        k = self._key(c)
        with self._lock:
            if k in self._cache: self._cache.move_to_end(k)
            self._cache[k] = (v, time.time())
            if len(self._cache) > self.max_size: self._cache.popitem(last=False)
    def __len__(self): return len(self._cache)

_cache = TTLCache(max_size=CACHE_MAX_SIZE, ttl=CACHE_TTL)

# =============================================================
# NLI LOADING
# =============================================================
NLI = None; NLI_BACKEND = "keyword"

def _try_load_nli_pipeline(model_name, local_only=False):
    global NLI, NLI_BACKEND
    try:
        from transformers import pipeline as hf_pipeline
        kwargs = dict(task="text-classification", model=model_name,
                      return_all_scores=True,
                      device=0 if torch.cuda.is_available() else -1)
        if local_only: kwargs["local_files_only"] = True
        NLI = hf_pipeline(**kwargs); NLI_BACKEND = "pipeline"
        print(f"  ✓ NLI loaded: {model_name}"); return True
    except Exception as e:
        print(f"  ✗ NLI load failed ({model_name}): {type(e).__name__}"); NLI = None; return False

def load_nli():
    global NLI_BACKEND
    for model in NLI_MODEL_CANDIDATES:
        if _try_load_nli_pipeline(model, local_only=True): return
    for model in NLI_MODEL_CANDIDATES:
        if _try_load_nli_pipeline(model, local_only=False): return
    NLI_BACKEND = "keyword"

load_nli()
print(f"NLI backend: {NLI_BACKEND}\n")

# =============================================================
# HTTP SESSION
# =============================================================
def _make_session():
    import urllib3; urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    s = req_lib.Session()
    a = HTTPAdapter(max_retries=Retry(total=1, backoff_factor=0.3, status_forcelist=[500,502,503,504]))
    s.mount("https://", a); s.mount("http://", a)
    s.verify = False; s.headers.update(HEADERS); return s

_SESSION = _make_session()

def safe_get(url, timeout=FETCH_TIMEOUT):
    try: return _SESSION.get(url, timeout=timeout)
    except Exception as e:
        if url.startswith("https://"):
            try: return _SESSION.get("http://"+url[8:], timeout=timeout)
            except: pass
        raise e

# =============================================================
# TEXT CLEANING
# =============================================================
def _clean_article_text(text):
    if not text or not isinstance(text, str): return ""
    try:
        fixed = text.encode("latin-1").decode("utf-8")
        if fixed.count("â€") < text.count("â€"):
            text = fixed
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    MOJIBAKE = [
        ("\u00e2\u20ac\u009c", '"'), ("\u00e2\u20ac\u009d", '"'),
        ("\u00e2\u20ac\u2122", "'"), ("\u00e2\u20ac\u0098", "'"),
        ("\u00e2\u20ac\u201c", "\u2014"), ("\u00e2\u20ac\u201d", "\u2013"),
        ("\u00e2\u20ac\u00a6", "..."), ("\u00e2\u20ac\u02dc", "'"),
        ("\u00e2\u20ac", '"'),
        ("\u00c3\u00a2\u00e2\u0082\u00ac\u00e2\u0084\u00a2", "'"),
        ("\u00c3\u00a2\u00e2\u0082\u00ac\u00c5\u0093", '"'),
        ("\u00c3\u00a2\u00e2\u0082\u00ac", '"'),
    ]
    for bad, good in MOJIBAKE:
        if bad in text: text = text.replace(bad, good)
    text = re.sub(r'â€[™œ˜\x80-\x9f]', "'", text)
    text = re.sub(r'â€', '"', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'^\s*\(?\s*(Reuters|AP|AFP|PTI|ANI|IANS|UNI|Bloomberg|BBC)\s*\)?\s*[-\u2013\u2014]\s*',
                  '', text, flags=re.IGNORECASE)
    text = re.sub(r'\b[A-Z]{2,10}\s*,\s*[A-Z][a-z]+\s*\((?:Reuters|AP|AFP|PTI)\)\s*[-\u2013\u2014]\s*', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', '', text)
    return re.sub(r'\s{2,}', ' ', text).strip()

def clean_text(t):
    t = re.sub(r"<[^>]+>","",t); t = html.unescape(t)
    return re.sub(r"\s+"," ",t).strip()

# =============================================================
# LABEL NORMALISATION
# =============================================================
_STRING_LABEL_MAP = {
    "FAKE":0, "FALSE":0, "MISINFORMATION":0, "UNRELIABLE":0, "SATIRE":0,
    "BIAS":0, "CONSPIRACY":0, "RUMOR":0, "RUMOUR":0, "PANTS-FIRE":0,
    "PANTS ON FIRE":0, "BARELY-TRUE":0, "BARELY TRUE":0, "MISLEADING":0,
    "INCORRECT":0, "INACCURATE":0, "UNVERIFIED":0, "HALF-TRUE":0,
    "HALF TRUE":0, "MOSTLY FALSE":0, "MOSTLY-FALSE":0, "DISPUTED":0,
    "FABRICATED":0, "MANIPULATED":0, "PROPAGANDA":0, "CLICKBAIT":0,
    "DECEPTIVE":0, "DISTORTED":0, "EXAGGERATED":0, "HOAX":0,
    "REAL":1, "TRUE":1, "RELIABLE":1, "CREDIBLE":1, "VERIFIED":1,
    "FACTUAL":1, "ACCURATE":1, "CORRECT":1, "MOSTLY-TRUE":1, "MOSTLY TRUE":1,
    "LARGELY TRUE":1, "LEGITIMATE":1, "AUTHENTIC":1, "CONFIRMED":1,
    "SUPPORTED":1, "VALID":1,
}
_NUMERIC_LABEL_MAP = {"0":0, "0.0":0, "1":1, "1.0":1}

def _norm_labels(s, flip=False):
    s = s.astype(str).str.strip().str.upper()
    numeric_mask = s.isin(set(_NUMERIC_LABEL_MAP.keys()))
    string_mask  = s.isin(set(_STRING_LABEL_MAP.keys()))
    result = s.map({**_STRING_LABEL_MAP, **_NUMERIC_LABEL_MAP})
    if flip and numeric_mask.any():
        result[numeric_mask] = result[numeric_mask].map({0:1, 1:0})
        print(f"  [LabelNorm] Flipped {numeric_mask.sum()} numeric rows  "
              f"(string={string_mask.sum()} unchanged)")
    else:
        if string_mask.any() and numeric_mask.any():
            print(f"  [LabelNorm] Mixed dataset: "
                  f"{string_mask.sum()} string + {numeric_mask.sum()} numeric rows")
        elif string_mask.any():
            print(f"  [LabelNorm] String-only: {string_mask.sum()} rows")
        else:
            print(f"  [LabelNorm] Numeric-only: {numeric_mask.sum()} rows  flip={flip}")
    unknown = (~numeric_mask & ~string_mask & s.notna() & (s != 'NAN') & (s != ''))
    if unknown.any():
        print(f"  [LabelNorm] WARNING: {unknown.sum()} unrecognised labels dropped: "
              f"{s[unknown].value_counts().head(5).to_dict()}")
    return result

def _detect_csv_columns(df):
    cols = [c.lower().strip() for c in df.columns]
    cm   = {c.lower().strip(): c for c in df.columns}
    LABEL_COLS = ["label","class","target","category","type","tag",
                  "ground_truth","verdict","annotation","output","y",
                  "fake","real","is_fake","is_real","truthfulness","credibility"]
    lc = next((cm[c] for c in LABEL_COLS if c in cols), None)
    tc  = next((cm[c] for c in ["text","content","body","article","news","statement",
                                  "claim","description","full_text","story","post"] if c in cols), None)
    tit = cm.get("title") or cm.get("headline") or cm.get("head")
    sub = next((cm[c] for c in ["subject","category","topic","section","genre"]
                if c in cols and cm.get(c) != lc), None)
    if tc is None and tit: tc = "__combined__"
    return tc, tit, lc, sub

def _detect_label_polarity(df, lc):
    if LABEL_0_MEANS == "FAKE": return False
    if LABEL_0_MEANS == "REAL": return True
    try:
        raw     = df[lc].astype(str).str.strip().str.upper()
        lc_low  = lc.lower().strip()
        if lc_low in ("fake","is_fake","fake_news","is_fake_news"):
            print(f"  [LabelPolarity] Column '{lc}' is a fake-flag -> 0=REAL, 1=FAKE -> FLIP")
            return True
        if lc_low in ("real","is_real","is_credible","credible"):
            print(f"  [LabelPolarity] Column '{lc}' is a real-flag -> 0=FAKE, 1=REAL -> no flip")
            return False
        string_mask  = raw.isin(set(_STRING_LABEL_MAP.keys()))
        numeric_mask = raw.isin({"0","1","0.0","1.0"})
        if string_mask.any():
            str_fake_count = (raw[string_mask].map(_STRING_LABEL_MAP) == 0).sum()
            str_real_count = (raw[string_mask].map(_STRING_LABEL_MAP) == 1).sum()
            print(f"  [LabelPolarity] String rows: FAKE={str_fake_count} REAL={str_real_count}")
            return False
        is_zero = raw == "0"; is_one = raw == "1"
        if not (is_zero.any() and is_one.any()):
            return False
        tc_check = next((c for c in df.columns if c.lower() in
                         ["text","title","content","body","article","news"]), None)
        if tc_check is None: return False
        REAL_SIGNALS = {
            "reuters","bbc","associated press","washington post","new york times",
            "cnn","nbc","abc","cbs","npr","guardian","according to","officials said",
            "confirmed","announced","government","president","minister","cbs news",
        }
        FAKE_SIGNALS = {
            "hoax","satire","conspiracy","clickbait","you won't believe",
            "shocking truth","mainstream media lies","wake up","sheeple",
            "share before deleted","truth exposed","they don't want you",
            "breaking alert","warning viral","exposed secret",
        }
        def _score(texts):
            combined = " ".join(str(t).lower() for t in texts[:100])
            r = sum(1 for w in REAL_SIGNALS if w in combined)
            f = sum(1 for w in FAKE_SIGNALS if w in combined)
            return r, f
        z_rs, z_fs = _score(df.loc[is_zero, tc_check].dropna().head(100).tolist())
        o_rs, o_fs = _score(df.loc[is_one,  tc_check].dropna().head(100).tolist())
        print(f"  [LabelPolarity] Group-0: real_signals={z_rs} fake_signals={z_fs} | "
              f"Group-1: real_signals={o_rs} fake_signals={o_fs}")
        if z_rs > o_rs and z_rs > z_fs and z_rs >= 2:
            print(f"  [LabelPolarity] AUTO: Group-0 looks REAL -> FLIP")
            return True
        if o_fs > z_fs and o_fs > o_rs and o_fs >= 2:
            print(f"  [LabelPolarity] AUTO: Group-1 looks FAKE -> FLIP")
            return True
        print(f"  [LabelPolarity] AUTO: signals unclear -> no flip (default 0=FAKE)")
        return False
    except Exception as e:
        print(f"  [LabelPolarity] Error: {e} -> no flip")
        return False
# =============================================================
# BERT AUTO-TRAINING  — uses _read_csv_safe
# =============================================================
def train_bert_from_csv(csv_path, model_dir):
    print(f"\n{'='*60}\n  BERT AUTO-TRAINING from {csv_path}\n{'='*60}")
    try:
        import pandas as pd
        from sklearn.model_selection import train_test_split
        from torch.utils.data import Dataset, DataLoader
        from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
        from torch.optim import AdamW

        df = _read_csv_safe(csv_path)  # FIX: was pd.read_csv(csv_path, encoding="utf-8", ...)
        print(f"  Loaded {len(df)} rows, cols:{list(df.columns)}")
        tc, tit, lc, sub = _detect_csv_columns(df)
        if lc is None: print("  ERROR: no label col"); return False
        flip = _detect_label_polarity(df, lc)
        df["__label__"] = _norm_labels(df[lc], flip=flip)
        df = df.dropna(subset=["__label__"])
        df = df[df["__label__"].isin([0,1])]
        if tc == "__combined__" or tc is None:
            if tit:
                df["__text__"] = df[tit].fillna("").astype(str).apply(_clean_article_text); tc = "__text__"
            else: print("  ERROR: no text col"); return False
        else:
            df[tc] = df[tc].fillna("").astype(str).apply(_clean_article_text)
            if tit and tit in df.columns:
                df[tit] = df[tit].fillna("").astype(str).apply(_clean_article_text)
            if sub and sub in df.columns:
                df["__text__"] = ("[" + df[sub].fillna("").astype(str).str.strip().str.upper() + "] " +
                                   df[tit].fillna("").astype(str).str.strip() + " " + df[tc].str[:200])
            elif tit and tit in df.columns:
                df["__text__"] = df[tit].fillna("").astype(str).str.strip() + " " + df[tc].str[:250]
            else:
                df["__text__"] = df[tc].str[:350]
            tc = "__text__"
        df[tc] = df[tc].fillna("").astype(str)
        df = df[df[tc].str.len() > 5]
        if len(df) < 10: print("  ERROR: too few rows"); return False
        n_fake = int((df["__label__"]==0).sum()); n_real = int((df["__label__"]==1).sum())
        print(f"  Clean:{len(df)} FAKE:{n_fake} REAL:{n_real}  flip={flip}")
        if len(df) > 24000:
            df = df.groupby("__label__", group_keys=False).apply(
                lambda x: x.sample(min(len(x), 12000), random_state=42))
            n_fake = int((df["__label__"]==0).sum()); n_real = int((df["__label__"]==1).sum())
        total = n_fake + n_real
        w_fake = total/(2.0*max(n_fake,1)); w_real = total/(2.0*max(n_real,1))
        class_weights = torch.tensor([w_fake, w_real], dtype=torch.float)
        print(f"  Class weights: FAKE={w_fake:.3f} REAL={w_real:.3f}")
        tr_t, va_t, tr_l, va_l = train_test_split(
            df[tc].tolist(), df["__label__"].astype(int).tolist(),
            test_size=0.1, random_state=42, stratify=df["__label__"].tolist())
        tok = BertTokenizer.from_pretrained("bert-base-uncased")
        class DS(Dataset):
            def __init__(self, t, l):
                self.e = tok(t, truncation=True, padding=True, max_length=128, return_tensors="pt")
                self.l = torch.tensor(l, dtype=torch.long)
            def __len__(self): return len(self.l)
            def __getitem__(self, i): return {k: v[i] for k, v in self.e.items()}, self.l[i]
        trd = DataLoader(DS(tr_t, tr_l), batch_size=16, shuffle=True)
        vad = DataLoader(DS(va_t, va_l), batch_size=32)
        dev   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        model.to(dev); class_weights = class_weights.to(dev)
        opt    = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        EPOCHS = 3
        sched  = get_linear_schedule_with_warmup(opt, len(trd)//5, len(trd)*EPOCHS)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        for ep in range(EPOCHS):
            model.train(); tl, c, t = 0, 0, 0
            for i, (b, lb) in enumerate(trd):
                b = {k: v.to(dev) for k, v in b.items()}; lb = lb.to(dev)
                logits = model(**b).logits; loss = loss_fn(logits, lb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                tl += loss.item(); p = torch.argmax(logits, dim=1)
                c += (p==lb).sum().item(); t += lb.size(0)
                if (i+1)%50==0:
                    print(f"    Ep{ep+1} Step{i+1}/{len(trd)} Loss={tl/(i+1):.4f} Acc={c/t:.3f}")
            print(f"  Ep{ep+1} Loss={tl/len(trd):.4f} Acc={c/t:.3f}")
            model.eval(); vc, vt = 0, 0
            with torch.no_grad():
                for b, lb in vad:
                    b = {k: v.to(dev) for k, v in b.items()}; lb = lb.to(dev)
                    vc += (torch.argmax(model(**b).logits,dim=1)==lb).sum().item(); vt += lb.size(0)
            print(f"  Val Acc:{vc/vt:.3f}")
        os.makedirs(model_dir, exist_ok=True)
        model.save_pretrained(model_dir); tok.save_pretrained(model_dir)
        print(f"  Model saved to {model_dir}")
        print("  Running threshold calibration on validation set...")
        model.eval(); val_probs, val_labels = [], []
        with torch.no_grad():
            for b, lb in vad:
                b = {k: v.to(dev) for k, v in b.items()}
                probs = torch.softmax(model(**b).logits, dim=1)[:, 1].cpu().tolist()
                val_probs.extend(probs); val_labels.extend(lb.cpu().tolist())
        calibrate_threshold(val_probs, val_labels)
        return True
    except Exception as e:
        print(f"  Training failed:{e}"); traceback.print_exc(); return False

# =============================================================
# DATASET-DRIVEN ENGINE (TF-IDF kNN)  — uses _read_csv_safe
# =============================================================
CSV_FALLBACK_LOADED = False
_tfv = _tfm = _tfl = _tfv2 = _tfm2 = None
OPTIMAL_THRESHOLD  = 0.50
LABEL_PRIOR_REAL   = 0.50
_TFIDF_TITLE_READY = False
_TFIDF_TEXT_READY  = False

def _build_tfidf_input(row, tc, tit, sub):
    parts = []
    if tit and tit in row.index and str(row[tit]).strip():
        parts.append(str(row[tit]).strip())
    if sub and sub in row.index and str(row[sub]).strip():
        parts.insert(0, "[" + str(row[sub]).strip().upper() + "]")
    if tc and tc in row.index and str(row[tc]).strip():
        parts.append(str(row[tc])[:150].strip())
    return " ".join(parts).strip()

def load_csv_fallback(p):
    global CSV_FALLBACK_LOADED, _tfv, _tfm, _tfl, _tfv2, _tfm2
    global LABEL_PRIOR_REAL, _TFIDF_TITLE_READY, _TFIDF_TEXT_READY
    if not os.path.isfile(p): return False
    try:
        import pandas as pd
        from sklearn.feature_extraction.text import TfidfVectorizer
        df = _read_csv_safe(p)  # FIX: was pd.read_csv(p, encoding="utf-8", ...)
        tc, tit, lc, sub = _detect_csv_columns(df)
        if not lc: return False
        flip = _detect_label_polarity(df, lc)
        df["__label__"] = _norm_labels(df[lc], flip=flip)
        df = df.dropna(subset=["__label__"])
        df = df[df["__label__"].isin([0,1])]
        if tc == "__combined__" or tc is None:
            if tit: df["__full__"] = df[tit].fillna("").astype(str).apply(_clean_article_text)
            else: return False
        else:
            df[tc] = df[tc].fillna("").astype(str).apply(_clean_article_text)
            if tit and tit in df.columns:
                df[tit] = df[tit].fillna("").astype(str).apply(_clean_article_text)
            df["__full__"] = df.apply(lambda r: _build_tfidf_input(r, tc, tit, sub), axis=1)
        df["__full__"] = df["__full__"].fillna("").astype(str)
        df = df[df["__full__"].str.len() > 5]
        if len(df) < 10: return False
        if len(df) > 40000:
            df = df.groupby("__label__", group_keys=False).apply(
                lambda x: x.sample(min(len(x), 20000), random_state=42))
        _tfl = df["__label__"].astype(int).values
        n_real = int((_tfl==1).sum()); n_fake = int((_tfl==0).sum())
        LABEL_PRIOR_REAL = n_real / max(len(_tfl), 1)
        print(f"  [Dataset Engine] {len(df)} rows  FAKE:{n_fake} REAL:{n_real}  prior_real={LABEL_PRIOR_REAL:.3f}  flip={flip}")
        title_texts = []
        for _, row in df.iterrows():
            t = ""
            if tit and tit in df.columns: t = str(row.get(tit,"")).strip()
            if sub and sub in df.columns: t = "["+str(row.get(sub,"")).upper()+"] "+t
            title_texts.append(t if t.strip() else str(row["__full__"])[:100])
        v1 = TfidfVectorizer(max_features=60000, ngram_range=(1,2), min_df=1, sublinear_tf=True)
        _tfv = v1; _tfm = v1.fit_transform(title_texts); _TFIDF_TITLE_READY = True
        v2 = TfidfVectorizer(max_features=80000, ngram_range=(1,3), min_df=2, sublinear_tf=True)
        _tfv2 = v2; _tfm2 = v2.fit_transform(df["__full__"].tolist()); _TFIDF_TEXT_READY = True
        print(f"  [Dataset Engine] Title TF-IDF {_tfm.shape}  Full-text TF-IDF {_tfm2.shape}")
        CSV_FALLBACK_LOADED = True; return True
    except Exception as e:
        print(f"  [Dataset Engine] Failed: {e}"); traceback.print_exc(); return False

def csv_fallback_score(claim, top_k=20):
    if not CSV_FALLBACK_LOADED:
        return {"label":"UNCERTAIN","fake_prob":0.5,"real_prob":0.5,"confidence":0.5}
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        clean_claim = _clean_article_text(claim)
        def _knn_score(vec, mat, labels, k=top_k):
            s = cosine_similarity(vec, mat)[0]; idx = np.argsort(s)[-k:][::-1]
            ts = s[idx]; tl = labels[idx]
            if ts.max() < 0.02: return LABEL_PRIOR_REAL, 1-LABEL_PRIOR_REAL, 0.0
            ts_s = ts+1e-6; w = ts_s/ts_s.sum()
            return float((w*(tl==1)).sum()), float((w*(tl==0)).sum()), float(ts.max())
        rp1=fp1=conf1=rp2=fp2=conf2=0.0
        if _TFIDF_TITLE_READY:
            v1 = _tfv.transform([clean_claim]); rp1, fp1, conf1 = _knn_score(v1, _tfm, _tfl)
        if _TFIDF_TEXT_READY:
            v2 = _tfv2.transform([clean_claim]); rp2, fp2, conf2 = _knn_score(v2, _tfm2, _tfl)
        claim_words = len(clean_claim.split())
        w1, w2 = (0.70, 0.30) if claim_words <= 15 else (0.45, 0.55)
        if _TFIDF_TITLE_READY and _TFIDF_TEXT_READY:
            rp = w1*rp1 + w2*rp2; fp = w1*fp1 + w2*fp2
        elif _TFIDF_TITLE_READY: rp, fp = rp1, fp1
        else: rp, fp = rp2, fp2
        label = "REAL" if rp >= OPTIMAL_THRESHOLD else "FAKE"
        return {"label":label,"fake_prob":round(fp,4),"real_prob":round(rp,4),"confidence":round(max(rp,fp),4)}
    except Exception as e:
        print(f"  [Dataset kNN] {e}")
        return {"label":"UNCERTAIN","fake_prob":0.5,"real_prob":0.5,"confidence":0.5}

def calibrate_threshold(bert_probs, true_labels):
    global OPTIMAL_THRESHOLD
    if not bert_probs or not true_labels: return 0.50
    best_f1, best_t = 0.0, 0.50
    for t in [x/100 for x in range(30, 71, 2)]:
        preds = [1 if p>=t else 0 for p in bert_probs]
        tp=sum(1 for a,b in zip(true_labels,preds) if a==1 and b==1)
        tn=sum(1 for a,b in zip(true_labels,preds) if a==0 and b==0)
        fp=sum(1 for a,b in zip(true_labels,preds) if a==0 and b==1)
        fn=sum(1 for a,b in zip(true_labels,preds) if a==1 and b==0)
        pr=tp/(tp+fp) if (tp+fp)>0 else 0; rc=tp/(tp+fn) if (tp+fn)>0 else 0
        f1r=2*pr*rc/(pr+rc) if (pr+rc)>0 else 0
        pf=tn/(tn+fn) if (tn+fn)>0 else 0; rf=tn/(tn+fp) if (tn+fp)>0 else 0
        f1f=2*pf*rf/(pf+rf) if (pf+rf)>0 else 0
        macro=(f1r+f1f)/2
        if macro > best_f1: best_f1, best_t = macro, t
    OPTIMAL_THRESHOLD = best_t
    print(f"  [Calibration] Optimal threshold={best_t:.2f}  val_macro_F1={best_f1:.4f}")
    return best_t

# =============================================================
# BERT MODEL LOADING
# =============================================================
BERT_LOADED = False; BERT_LABEL_MAP = {0:"FAKE",1:"REAL"}
bert_tokenizer = bert_model = None

def _try_load_bert(d):
    global bert_tokenizer, bert_model, BERT_LOADED
    try:
        bert_tokenizer = BertTokenizer.from_pretrained(d)
        bert_model     = BertForSequenceClassification.from_pretrained(d)
        bert_model.eval(); bert_model.to("cpu")
        BERT_LOADED = True; print("BERT loaded."); return True
    except Exception as e:
        print(f"BERT load failed:{e}"); BERT_LOADED = False; return False

if os.path.isdir(BERT_MODEL_DIR):
    _try_load_bert(BERT_MODEL_DIR)
else:
    print(f"WARNING: BERT not at {BERT_MODEL_DIR}")

if not BERT_LOADED:
    if os.path.isfile(CSV_PATH):
        print("CSV found. Auto-training...")
        if train_bert_from_csv(CSV_PATH, BERT_MODEL_DIR):
            _try_load_bert(BERT_MODEL_DIR)
    else:
        print("Trying HuggingFace pre-trained model...")
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            os.makedirs(BERT_MODEL_DIR, exist_ok=True)
            bert_tokenizer = AutoTokenizer.from_pretrained("hamzab/roberta-fake-news-classification")
            bert_model     = AutoModelForSequenceClassification.from_pretrained("hamzab/roberta-fake-news-classification")
            bert_model.eval(); bert_model.to("cpu")
            bert_tokenizer.save_pretrained(BERT_MODEL_DIR)
            bert_model.save_pretrained(BERT_MODEL_DIR)
            BERT_LOADED = True; print("Pre-trained model downloaded.")
        except Exception as e:
            print(f"HuggingFace failed:{e}")

if os.path.isfile(CSV_PATH) and not CSV_FALLBACK_LOADED:
    print("Loading dataset TF-IDF engine...")
    load_csv_fallback(CSV_PATH)

print(f"\nServer ready. BERT={BERT_LOADED} NLI={NLI_BACKEND} CSV_FALLBACK={CSV_FALLBACK_LOADED}\n")

# =============================================================
# QUICK CALIBRATION (startup)  — uses _read_csv_safe
# =============================================================
def quick_calibrate_existing_model(csv_path, n_samples=300):
    global OPTIMAL_THRESHOLD
    if not BERT_LOADED or not os.path.isfile(csv_path):
        print("[QuickCalib] Skipped"); return OPTIMAL_THRESHOLD
    try:
        import pandas as pd
        print(f"[QuickCalib] Running on {n_samples} rows...")
        df = _read_csv_safe(csv_path)  # FIX: was pd.read_csv(csv_path, encoding="utf-8", ...)
        tc, tit, lc, sub = _detect_csv_columns(df)
        if not lc: return OPTIMAL_THRESHOLD
        flip = _detect_label_polarity(df, lc)
        df["__label__"] = _norm_labels(df[lc], flip=flip)
        df = df.dropna(subset=["__label__"]); df = df[df["__label__"].isin([0,1])]
        if tc and tc != "__combined__":
            df[tc] = df[tc].fillna("").astype(str).apply(_clean_article_text)
            if tit and tit in df.columns:
                df[tit] = df[tit].fillna("").astype(str).apply(_clean_article_text)
                df["__cal_text__"] = (df[tit].str.strip()+" "+df[tc].str[:200]).str.strip()
            else:
                df["__cal_text__"] = df[tc].str[:300]
        elif tit and tit in df.columns:
            df["__cal_text__"] = df[tit].fillna("").astype(str).apply(_clean_article_text)
        else:
            return OPTIMAL_THRESHOLD
        df["__cal_text__"] = df["__cal_text__"].fillna("").astype(str)
        df = df[df["__cal_text__"].str.len() > 5]
        n_each  = n_samples//2
        fake_df = df[df["__label__"]==0].sample(min(n_each,(df["__label__"]==0).sum()), random_state=42)
        real_df = df[df["__label__"]==1].sample(min(n_each,(df["__label__"]==1).sum()), random_state=42)
        cal_df  = pd.concat([fake_df,real_df]).sample(frac=1,random_state=42).reset_index(drop=True)
        print(f"[QuickCalib] Sampled {len(cal_df)} rows (FAKE:{len(fake_df)} REAL:{len(real_df)})")
        probs, labels = [], []
        for i, row in cal_df.iterrows():
            text = str(row["__cal_text__"])[:400]; label = int(row["__label__"])
            try:
                bp   = bert_score_claim(text); rp = bp["real_prob"]
                boost = _credible_source_boost(text); rp = min(rp+boost, 0.99)
                probs.append(rp); labels.append(label)
            except: pass
            if (i+1)%50==0: print(f"[QuickCalib]   {i+1}/{len(cal_df)} scored...")
        if not probs: return OPTIMAL_THRESHOLD
        best_f1, best_t = 0.0, 0.50; results = []
        for t_int in range(25, 76, 2):
            t = t_int/100.0; preds = [1 if p>=t else 0 for p in probs]
            tp=sum(1 for a,b in zip(labels,preds) if a==1 and b==1)
            tn=sum(1 for a,b in zip(labels,preds) if a==0 and b==0)
            fp=sum(1 for a,b in zip(labels,preds) if a==0 and b==1)
            fn=sum(1 for a,b in zip(labels,preds) if a==1 and b==0)
            acc=(tp+tn)/len(labels) if labels else 0
            pr_r=tp/(tp+fp) if (tp+fp)>0 else 0; rc_r=tp/(tp+fn) if (tp+fn)>0 else 0
            f1_r=2*pr_r*rc_r/(pr_r+rc_r) if (pr_r+rc_r)>0 else 0
            pr_f=tn/(tn+fn) if (tn+fn)>0 else 0; rc_f=tn/(tn+fp) if (tn+fp)>0 else 0
            f1_f=2*pr_f*rc_f/(pr_f+rc_f) if (pr_f+rc_f)>0 else 0
            macro=(f1_r+f1_f)/2; results.append((macro,acc,t,tp,tn,fp,fn))
            if macro > best_f1: best_f1, best_t = macro, t
        OPTIMAL_THRESHOLD = best_t
        best = [r for r in results if r[2]==best_t][0]
        macro,acc,t,tp,tn,fp,fn = best
        print(f"\n[QuickCalib] {'='*50}")
        print(f"[QuickCalib] Optimal threshold: {best_t:.2f}  Accuracy: {acc:.1%}  Macro F1: {macro:.4f}")
        print(f"[QuickCalib] TP={tp}  TN={tn}  FP={fp}  FN={fn}")
        print(f"[QuickCalib] {'='*50}\n")
        return best_t
    except Exception as e:
        print(f"[QuickCalib] Error: {e}"); traceback.print_exc(); return OPTIMAL_THRESHOLD

if BERT_LOADED and os.path.isfile(CSV_PATH):
    quick_calibrate_existing_model(CSV_PATH, n_samples=300)
else:
    print("[QuickCalib] Skipped (BERT not loaded or CSV missing)")
# =============================================================
# BERT INFERENCE
# =============================================================
def _credible_source_boost(claim):
    cl = claim.lower()
    credible_hits    = sum(1 for src in CREDIBLE_SOURCE_SIGNALS if src in cl)
    sensational_hits = sum(1 for pat in SENSATIONALIST_PATTERNS if re.search(pat, claim, re.IGNORECASE))
    if credible_hits == 0: return 0.0
    if sensational_hits == 0:
        boost = min(0.08 + 0.04 * (credible_hits - 1), 0.20)
        print(f"  [CredibleBoost] credible={credible_hits} (no sensationalism) boost=+{boost:.2f}")
        return boost
    else:
        boost = min(0.06 + 0.02 * (credible_hits - 1), 0.15)
        print(f"  [CredibleBoost] credible={credible_hits} sensational={sensational_hits} boost=+{boost:.2f}")
        return boost

def bert_score_claim(claim):
    cleaned    = _clean_article_text(claim)
    bert_input = cleaned if cleaned.strip() else claim
    is_long    = len(bert_input.split()) > 40
    max_len    = ARTICLE_MAX_LENGTH if is_long else CLAIM_MAX_LENGTH
    bert_result = tfidf_result = None
    if BERT_LOADED:
        try:
            inp = bert_tokenizer(bert_input, return_tensors="pt", truncation=True,
                                  padding=True, max_length=max_len)
            with torch.no_grad():
                logits = bert_model(**inp).logits
            p = torch.softmax(logits, dim=1)[0].tolist()
            fp, rp = (p[0],p[1]) if len(p)==2 else (sum(p[:len(p)//2]),sum(p[len(p)//2:]))
            if len(p) != 2:
                tot = fp+rp; fp, rp = fp/tot, rp/tot
            bert_result = {"real_prob":rp,"fake_prob":fp}
        except Exception as e: print(f"  [BERT] {e}")
    if CSV_FALLBACK_LOADED:
        tfidf_result = csv_fallback_score(claim)
    if bert_result and tfidf_result:
        rp = 0.55*bert_result["real_prob"] + 0.35*tfidf_result["real_prob"] + 0.10*LABEL_PRIOR_REAL
        fp = 1.0-rp
    elif bert_result:
        rp = bert_result["real_prob"]; fp = bert_result["fake_prob"]
    elif tfidf_result:
        rp = 0.85*tfidf_result["real_prob"] + 0.15*LABEL_PRIOR_REAL; fp = 1.0-rp
    else:
        return {"label":"UNCERTAIN","fake_prob":0.5,"real_prob":0.5,"confidence":0.5}
    label = "REAL" if rp >= OPTIMAL_THRESHOLD else "FAKE"
    return {"label":label,"fake_prob":round(fp,4),"real_prob":round(rp,4),"confidence":round(max(rp,fp),4)}

def bert_score_evidence(ev, claim):
    return bert_score_claim(f"{ev[:200]} {claim[:80]}")

# =============================================================
# SEMANTIC OVERLAP
# =============================================================
def _char_bigrams(text):
    t = re.sub(r"\s+"," ",text.lower().strip())
    return {t[i:i+2] for i in range(len(t)-1)} if len(t) >= 2 else set()

def semantic_overlap(claim, evidence):
    cl_tok = set(re.findall(r"[a-z]{3,}",claim.lower()))
    ev_tok = set(re.findall(r"[a-z]{3,}",evidence.lower()))
    stop   = {"the","and","for","are","was","were","has","had","have","been","this","that",
               "with","from","not","but","they","their","its"}
    cl_tok -= stop; ev_tok -= stop
    token_score = len(cl_tok&ev_tok)/len(cl_tok) if cl_tok else 0.0
    cl_bg = _char_bigrams(claim); ev_bg = _char_bigrams(evidence)
    bigram_score = 2*len(cl_bg&ev_bg)/(len(cl_bg)+len(ev_bg)) if (cl_bg and ev_bg) else 0.0
    return round(0.70*token_score + 0.30*bigram_score, 4)

def semantic_relevance_score(evidence, keywords):
    ev_lower = evidence.lower(); ev_bg = _char_bigrams(ev_lower); score = 0.0
    for kw in keywords:
        kw_l = kw.lower()
        if kw_l in ev_lower: score += 1.0
        else:
            kw_bg = _char_bigrams(kw_l)
            if kw_bg and ev_bg:
                dice = 2*len(kw_bg&ev_bg)/(len(kw_bg)+len(ev_bg))
                if dice > 0.35: score += dice
    return score

def is_relevant_semantic(evidence, keywords, threshold=0.5):
    return semantic_relevance_score(evidence, keywords) >= threshold

# =============================================================
# LIVE LOOKUP (Wikipedia / Web)
# =============================================================
_live_cache = {}; _live_cache_time = {}; LIVE_CACHE_TTL = 21600

def _extract_name_from_text(text):
    for pat in NAME_EXTRACT_PATTERNS:
        m = re.search(pat, text)
        if m:
            name = m.group(1).strip()
            if len(name.split()) >= 2 and not re.search(r"\d", name):
                return name.lower()
    return None

def _wikipedia_lookup(query):
    try:
        results = safe_get(
            f"https://en.wikipedia.org/w/api.php?action=query&list=search"
            f"&srsearch={quote_plus(query)}&format=json&srlimit=2", timeout=3
        ).json().get("query",{}).get("search",[])
        for result in results:
            title = result["title"]
            pages = safe_get(
                f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts"
                f"&exintro=true&explaintext=true&titles={quote_plus(title)}&format=json", timeout=3
            ).json().get("query",{}).get("pages",{})
            for p in pages.values():
                text = p.get("extract","")[:1500]
                if not text: continue
                name = _extract_name_from_text(text)
                if name: return name
    except Exception as e: print(f"  [WikiLookup] {e}")
    return None

def _web_search_lookup(query):
    try:
        items = parse_rss(f"https://news.google.com/rss/search?q={quote_plus(query)}&hl=en&gl=US&ceid=US:en","WebSearchLookup",5)
        name  = _extract_name_from_text(" ".join(items))
        if name: return name
    except Exception as e: print(f"  [WebSearchLookup] {e}")
    try:
        data  = _SESSION.get(f"http://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_redirect=1&no_html=1&skip_disambig=1",timeout=3).json()
        ab    = clean_text(data.get("AbstractText",""))
        if ab:
            name = _extract_name_from_text(ab)
            if name: return name
    except Exception as e: print(f"  [DDGLookup] {e}")
    return None

def live_lookup(cache_key, wiki_query, web_query):
    now = time.time()
    if cache_key in _live_cache and now-_live_cache_time.get(cache_key,0) < LIVE_CACHE_TTL:
        return _live_cache[cache_key]
    result = _wikipedia_lookup(wiki_query) or _web_search_lookup(web_query)
    if result: _live_cache[cache_key] = result; _live_cache_time[cache_key] = now
    return result

def _claim_names_person(cl, person):
    if person in cl: return True
    for part in person.split():
        if len(part) >= 4 and part in cl: return True
    return False

def captain_check(claim):
    cl = claim.lower()
    if not any(w in cl for w in ["captain","captian","skipper","captains","led by"]):
        return {"verdict":"UNCERTAIN","reason":"not a captain claim"}
    if not any(w in cl for w in ["cricket","t20","odi","test match","icc","bcci","twenty20","one day","test cricket"]):
        return {"verdict":"UNCERTAIN","reason":"not a cricket claim"}
    fmt = ""
    if any(w in cl for w in ["t20","twenty20","twenty-20","ipl"]):    fmt = "t20"
    elif any(w in cl for w in ["odi","one day","one-day","50 over"]): fmt = "odi"
    elif any(w in cl for w in ["test","test match","test cricket"]):  fmt = "test"
    country = None
    for alias, canon in COUNTRY_ALIAS_MAP.items():
        if alias in cl: country = canon; break
    if not country:
        for c in CRICKET_TEAMS:
            if c in cl: country = c; break
    if not country: return {"verdict":"UNCERTAIN","reason":"country not detected"}
    fmt_label = fmt.upper() if fmt else "cricket"
    cache_key = f"cricket_captain_{country}_{fmt}".replace(" ","_")
    actual    = live_lookup(cache_key,
                            f"{country.title()} cricket team {fmt_label} captain current 2025",
                            f"who is current {country} cricket {fmt_label} captain 2025")
    if not actual: return {"verdict":"UNCERTAIN","reason":f"could not fetch {country} {fmt_label} captain live"}
    if _claim_names_person(cl, actual):
        return {"verdict":"REAL","reason":f"Verified live: {actual.title()} IS the {country.title()} {fmt_label} captain"}
    return {"verdict":"FAKE","reason":f"Live lookup: actual {country.title()} {fmt_label} captain is {actual.title()}"}

def role_check(claim):
    cl = claim.lower()
    DEATH_ALIVE_PHRASES = {"passed away","has not passed","not passed","not died","has not died",
                           "didn't die","did not die","is not dead","is alive","not dead","still alive",
                           "hasn't died","hasn't passed","death","died","dead","obituary","no more",
                           "demise","passes away","breathed his last"}
    if any(ph in cl for ph in DEATH_ALIVE_PHRASES):
        return {"verdict":"UNCERTAIN","reason":"skipped — death/alive claim"}
    matched_role = wiki_tmpl = web_tmpl = None
    for triggers, role_label, wq, webq in ROLE_DETECTORS:
        if any(t in cl for t in triggers):
            matched_role = role_label; wiki_tmpl = wq; web_tmpl = webq; break
    if not matched_role: return {"verdict":"UNCERTAIN","reason":"no known role detected"}
    country = "india"
    for c in POLITICAL_COUNTRIES:
        if c in cl: country = c; break
    country_label = country.title().replace("Usa","USA").replace("Uk","UK").replace("Un","UN")
    cache_key = f"role_{matched_role}_{country}".replace(" ","_").lower()
    actual    = live_lookup(cache_key,
                            wiki_tmpl.format(country=country_label),
                            web_tmpl.format(country=country_label))
    if not actual: return {"verdict":"UNCERTAIN","reason":f"could not fetch {matched_role} of {country_label}"}
    actual_words = actual.strip().split()
    if (len(actual_words)<2 or len(actual)>60
        or any(w in actual for w in ["museum","article","according","breaking","news","search",
                                      "government","parliament","election","party"])
        or not all(w[0].isalpha() for w in actual_words)):
        return {"verdict":"UNCERTAIN","reason":f"web lookup returned invalid name: '{actual}'"}
    if _claim_names_person(cl, actual):
        return {"verdict":"REAL","reason":f"Verified live: {actual.title()} IS the {matched_role} of {country_label}"}
    wiki_result = _wikipedia_lookup(wiki_tmpl.format(country=country_label))
    if wiki_result and wiki_result == actual:
        return {"verdict":"FAKE","reason":f"Wikipedia confirmed: actual {matched_role} of {country_label} is {actual.title()}"}
    return {"verdict":"UNCERTAIN","reason":f"Live lookup inconclusive for {matched_role} of {country_label}"}

# =============================================================
# FACT CHECK / WIKIPEDIA / NEWSAPI / GEMINI
# =============================================================
def fetch_google_factcheck(claim):
    if not FACTCHECK_API_KEY or "YOUR" in FACTCHECK_API_KEY:
        return {"found":False,"verdict":None,"claims":[]}
    qtry = [claim[:200], " ".join(claim.split()[:10])]
    out, tc, fc, seen = [], 0, 0, set()
    for q in qtry:
        try:
            d = safe_get(
                f"https://factchecktools.googleapis.com/v1alpha1/claims:search"
                f"?query={quote_plus(q)}&key={FACTCHECK_API_KEY}&languageCode=en",
                timeout=FETCH_TIMEOUT).json()
            for it in d.get("claims",[])[:6]:
                for rv in it.get("claimReview",[]):
                    u = rv.get("url","")
                    if u in seen: continue
                    seen.add(u); r = rv.get("textualRating","").lower()
                    out.append({"text":it.get("text","")[:200],"claimant":it.get("claimant",""),
                                "rating":r,"url":u,"publisher":rv.get("publisher",{}).get("name","")})
                    if any(w in r for w in ["true","correct","accurate","verified","confirmed","mostly true","largely true"]): tc += 1
                    elif any(w in r for w in ["false","incorrect","inaccurate","fake","misleading","pants on fire","mostly false","fabricated","no evidence","unverified","baseless"]): fc += 1
        except Exception as e: print(f"  [FactCheck] {e}")
    if not out: return {"found":False,"verdict":None,"claims":[]}
    v = "REAL" if tc>fc else ("FAKE" if fc>tc else "UNCERTAIN")
    return {"found":True,"verdict":v,"claims":out}

def fetch_wikipedia(entity):
    if not entity or len(entity.strip()) < 3:
        return {"found":False,"summary":"","alive":None,"url":""}
    try:
        res = safe_get(
            f"https://en.wikipedia.org/w/api.php?action=query&list=search"
            f"&srsearch={quote_plus(entity)}&format=json&srlimit=1",
            timeout=FETCH_TIMEOUT).json().get("query",{}).get("search",[])
        if not res: return {"found":False,"summary":"","alive":None,"url":""}
        title = res[0]["title"]
        pg    = safe_get(
            f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts"
            f"&exintro=true&explaintext=true&titles={quote_plus(title)}&format=json",
            timeout=FETCH_TIMEOUT).json().get("query",{}).get("pages",{})
        s = ""
        for p in pg.values(): s = p.get("extract","")[:800]; break
        if not s: return {"found":False,"summary":"","alive":None,"url":""}
        low   = s.lower()
        alive = not any(re.search(pat, low) for pat in ["died","death","deceased","passed away",r"born \d{4}.*\d{4}"])
        if re.search(r"\bis (a|an|the) [a-z]+ (actor|politician|cricketer|singer|director|businessman|player)",low):
            alive = True
        return {"found":True,"summary":s,"alive":alive,
                "url":f"https://en.wikipedia.org/wiki/{quote_plus(title.replace(' ','_'))}"}
    except Exception as e:
        print(f"  [Wikipedia] {e}"); return {"found":False,"summary":"","alive":None,"url":""}

def fetch_wikipedia_evidence(claim, max_sentences=8):
    try:
        query = re.sub(r"[^\x00-\x7F]"," ",claim)
        query = re.sub(r"\s{2,}"," ",query).strip()
        if len(query) > 120: query = query[:120].rsplit(" ",1)[0]
        search_res = safe_get(
            f"https://en.wikipedia.org/w/api.php?action=query&list=search"
            f"&srsearch={quote_plus(query)}&format=json&srlimit=3",
            timeout=FETCH_TIMEOUT).json().get("query",{}).get("search",[])
        if not search_res: return []
        best_sentences, best_score = [], 0.0
        for result in search_res[:2]:
            title = result["title"]
            pages = safe_get(
                f"https://en.wikipedia.org/w/api.php?action=query&prop=extracts"
                f"&explaintext=true&titles={quote_plus(title)}&format=json",
                timeout=FETCH_TIMEOUT).json().get("query",{}).get("pages",{})
            full_text = ""
            for p in pages.values(): full_text = p.get("extract","")[:3000]; break
            if not full_text: continue
            raw_sents = [s.strip() for s in re.split(r"[.!?\n]",full_text) if len(s.strip())>25]
            scored    = sorted([(semantic_overlap(claim,s),s) for s in raw_sents], reverse=True)
            top       = [s for score,s in scored[:max_sentences] if score>0.05]
            art_score = scored[0][0] if scored else 0.0
            if art_score > best_score:
                best_score = art_score; best_sentences = top
        return best_sentences
    except Exception as e:
        print(f"  [WikiEvidence] {e}"); return []

def wiki_alive_check(claim, entities):
    r = {"is_alive":None,"wiki_summary":"","wiki_url":""}
    candidates = list(entities[:2]); cl = claim.lower(); ncl = normalise_claim(claim).lower()
    for p in sorted(KNOWN_LIVING_PEOPLE, key=len, reverse=True):
        if (p in cl or p in ncl) and p.title() not in candidates:
            candidates.insert(0, p.title()); break
    all_words = set(cl.split())|set(ncl.split())
    for l, f in LAST_NAME_MAP.items():
        if l in all_words and f.title() not in candidates:
            candidates.append(f.title()); break
    for e in candidates[:3]:
        w = fetch_wikipedia(e)
        if w["found"]:
            r.update({"is_alive":w["alive"],"wiki_summary":w["summary"],"wiki_url":w["url"]}); break
    return r

def fetch_newsapi(q):
    if not NEWSAPI_KEY or "YOUR" in NEWSAPI_KEY: return []
    try:
        d = safe_get(
            f"https://newsapi.org/v2/everything?q={quote_plus(q)}"
            f"&language=en&sortBy=relevancy&pageSize=10&apiKey={NEWSAPI_KEY}",
            timeout=FETCH_TIMEOUT).json()
        out = []
        for a in d.get("articles",[]):
            t = clean_text(a.get("title","")); dc = clean_text(a.get("description",""))
            s = dc if len(dc)>len(t) else t
            if s and len(s)>20 and "[Removed]" not in s: out.append(s)
        return out[:10]
    except Exception as e: print(f"    [NewsAPI] {e}"); return []

def _call_gemini(prompt):
    if not GEMINI_SDK or not GEMINI_API_KEY or "YOUR" in GEMINI_API_KEY:
        raise RuntimeError("Gemini API not configured")
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    resp  = model.generate_content(prompt); raw = resp.text.strip()
    raw   = re.sub(r"^```(?:json)?\s*|\s*```$","",raw,flags=re.MULTILINE).strip()
    m     = re.search(r'\{[^{}]+\}',raw,re.DOTALL)
    if m: raw = m.group(0)
    return json.loads(raw)

def call_claude_ai(claim, evs, fc_claims, wiki_sum, bert_prior):
    blk = ""
    if fc_claims:
        blk += "FACT-CHECKS:\n"
        for fc in fc_claims[:3]: blk += f"• [{fc['publisher']}] {fc['text'][:120]}  Rating:{fc['rating']}\n"
        blk += "\n"
    if wiki_sum: blk += f"WIKIPEDIA:\n{wiki_sum[:600]}\n\n"
    if evs:
        blk += "NEWS EVIDENCE:\n"
        for i, e in enumerate(evs[:8],1): blk += f"{i}. {e[:250]}\n"
    if bert_prior and bert_prior.get("label") != "UNCERTAIN":
        blk += f"\nML MODEL: {bert_prior['label']} (confidence {bert_prior['confidence']:.0%})"
    prompt = (
        f'You are a professional fact-checker. Verify the following claim IN ITS ENTIRETY.\n\n'
        f'SCIENTIFIC RULES:\n'
        f'- Earth revolves around the Sun (NOT the other way). "Sun revolves around Earth" is FAKE.\n'
        f'- The Earth is spherical. "Earth is flat" is FAKE.\n'
        f'- Light travels faster than sound.\n'
        f'- Pay attention to SUBJECT vs OBJECT — reversing them changes the meaning entirely.\n\n'
        f'FULL CLAIM:\n"{claim}"\n\nEVIDENCE:\n{blk}\n\n'
        f'Instructions: REAL=fully supported, FAKE=key part wrong, PARTIALLY_TRUE=mixed, UNCERTAIN=insufficient.\n'
        f'Default to UNCERTAIN over FAKE when evidence is sparse.\n\n'
        f'Reply ONLY with valid JSON:\n'
        f'{{"verdict":"REAL or FAKE or PARTIALLY_TRUE or UNCERTAIN","explanation":"1-2 sentences","confidence":"high or medium or low"}}'
    )
    try:
        p = _call_gemini(prompt); v = p.get("verdict","UNCERTAIN").upper()
        if v not in ("REAL","FAKE","UNCERTAIN",PARTIALLY_TRUE): v = "UNCERTAIN"
        return {"verdict":v,"explanation":p.get("explanation",""),"confidence":p.get("confidence","medium"),"used":True}
    except Exception as e:
        print(f"  [Gemini AI] {e}")
        return {"verdict":"UNCERTAIN","explanation":str(e),"confidence":"low","used":False}

def call_claude_ai_knowledge(claim, wiki_sum="", fc_claims=None):
    """Knowledge-based fact check using facts loaded from truthlens_data.json,
    then falls back to Gemini AI."""
    cl = claim.lower()
    claim_words = set(re.findall(r"[a-z0-9]+", cl))

    # Tier 0a: Order-sensitive regex from JSON science_regex key
    for pat, verdict, explanation in SCIENCE_REGEX:
        if re.search(pat, cl):
            print(f"  [KnowledgeBase-Regex] '{pat[:40]}' -> {verdict}: {explanation[:60]}")
            return {"verdict":verdict,"explanation":explanation,"confidence":"high","used":True,"source":"science_regex"}

    # Tier 0b: Keyword-set facts from JSON known_facts key
    matched = []
    for keywords, verdict, explanation in KNOWN_FACTS:
        if keywords.issubset(claim_words): matched.append((keywords, verdict, explanation))
    if matched:
        fake_m = [(k,v,e) for k,v,e in matched if v=="FAKE"]
        k, verdict, explanation = fake_m[0] if fake_m else matched[0]
        print(f"  [KnowledgeBase-KW] -> {verdict}: {explanation[:80]}")
        return {"verdict":verdict,"explanation":explanation,"confidence":"high","used":True,"source":"knowledge_base"}

    # Tier 1: Gemini AI
    context = ""
    if wiki_sum: context += f"Wikipedia context:\n{wiki_sum[:600]}\n\n"
    for fc in (fc_claims or [])[:2]:
        context += f"Fact-check: [{fc['publisher']}] {fc['text'][:120]} -> {fc['rating']}\n"
    prompt = (
        f'You are a fact-checking assistant. Verify the following claim using your knowledge.\n\n'
        f'CRITICAL RULES:\n'
        f'1. Earth revolves around the Sun — not the other way.\n'
        f'2. Pay close attention to SUBJECT and OBJECT. "A revolves around B" != "B revolves around A".\n'
        f'3. For science: apply established consensus confidently.\n'
        f'4. For recent events: say UNCERTAIN if unsure.\n\n'
        f'FULL CLAIM:\n"{claim}"\n\n{context}'
        f'Reply ONLY with valid JSON:\n'
        f'{{"verdict":"REAL","explanation":"one sentence","confidence":"high"}}'
    )
    try:
        p = _call_gemini(prompt); v = p.get("verdict","UNCERTAIN").upper()
        if v not in ("REAL","FAKE","UNCERTAIN"): v = "UNCERTAIN"
        return {"verdict":v,"explanation":p.get("explanation",""),"confidence":p.get("confidence","medium"),"used":True}
    except Exception as e:
        print(f"  [Gemini Knowledge] {e}")
        return {"verdict":"UNCERTAIN","explanation":str(e),"confidence":"low","used":False}

# =============================================================
# TEXT UTILS
# =============================================================
_FILLER = re.compile(
    r"^(indian|bollywood|tollywood|kollywood|hollywood|south\s+indian)?\s*"
    r"(heroin|hero|actor|actress|celebrity|star|superstar|legend|singer|"
    r"director|producer|cricketer|player|politician|leader|officer|chief|"
    r"minister|president|pm)\s+", re.IGNORECASE)

def normalise_claim(claim):
    r, p = claim.strip(), None
    while p != r:
        p = r; r = _FILLER.sub("",r).strip()
        r = re.sub(r"(\band\b|,)\s+(indian|bollywood|tollywood|kollywood|hollywood|south\s+indian)?\s*"
                   r"(heroin|hero|actor|actress|celebrity|star|superstar|legend|singer|director|producer|"
                   r"cricketer|player|politician|leader|officer|chief|minister|president|pm)\s+",
                   r"\1 ",r,flags=re.IGNORECASE)
    return re.sub(r"\s{2,}"," ",r).strip()

def apply_role_map(c):
    for p, cn in ROLE_PATTERNS: c = re.sub(p, cn, c, flags=re.IGNORECASE)
    return c

def expand_celeb_names(entities):
    seen, out = set(), []
    for e in entities:
        f = CELEB_DATA.get(e.lower().strip(), e)
        if f.lower() not in seen: seen.add(f.lower()); out.append(f)
    return out

def extract_keywords(c):
    raw  = re.sub(r"[^\x00-\x7F]"," ",c)
    hyph = re.findall(r"[a-zA-Z][a-zA-Z0-9]*-[a-zA-Z0-9]+",raw)
    kept = []; seen = set()
    for h in hyph:
        hl = h.lower()
        if hl not in seen: seen.add(hl); kept.append(hl)
    for t in re.findall(r"[a-zA-Z0-9]+",raw):
        tl = t.lower()
        if tl in seen: continue
        seen.add(tl)
        if re.fullmatch(r"\d{1,2}",tl): continue
        if tl in PRESERVE_WORDS: kept.append(tl)
        elif tl not in GENERIC_WORDS and len(tl) >= 3: kept.append(tl)
    return kept

def keyword_overlap(c, e): return semantic_overlap(c, e)
def is_relevant(e, kws, threshold=2): return semantic_relevance_score(e, kws) >= threshold
def relevance_score(e, kws): return semantic_relevance_score(e, kws)

def relation_match(claim, evidence):
    cl = claim.lower(); ev = evidence.lower()
    for root, words in RELATION_KEYWORDS.items():
        if root in cl: return any(w in ev for w in words)
    return True

def extract_best_sentence(article, claim):
    sentences = [s.strip() for s in re.split(r"[.!?]",article) if len(s.strip())>20]
    if not sentences: return article
    best, best_score = article, 0.0
    for s in sentences:
        score = semantic_overlap(claim, s)
        if score > best_score: best_score = score; best = s
    return best

def classify_claim_scope(c):
    cl = c.lower()
    for w in GLOBAL_INDICATORS:
        if w in cl: return "global"
    for w in LOCAL_INDICATORS:
        if w in cl: return "local"
    return "global"

def is_celebrity_personal_claim(c):
    cl = c.lower()
    if not any(w in cl for w in _CELEB_ONLY_NAMES): return False
    if not any(w in cl for w in CELEBRITY_PERSONAL_WORDS): return False
    if any(w in cl for w in _POLITICAL_INDICATORS): return False
    return True

def _is_article_input(claim):
    cl = claim.strip()
    if cl.count("â€") >= 2 or cl.count("Ã") >= 2: return True
    BYLINE_PATTERNS = [r"^[A-Z]{2,20}\s*\(Reuters\)", r"^[A-Z]{2,20}\s*\(AP\)",
                       r"^[A-Z]{2,20}\s*\(AFP\)", r"^\(Reuters\)", r"^\(AP\)",
                       r"^Reuters\s*-", r"^By [A-Z][a-z]+ [A-Z][a-z]+"]
    for pat in BYLINE_PATTERNS:
        if re.match(pat, cl): return True
    if len(cl.split()) > 40: return True
    if len(re.findall(r'[.!?]\s+[A-Z]', cl)) >= 3: return True
    ARTICLE_PHRASES = ["according to","told reporters","said in a statement","said on",
                       "said wednesday","said thursday","said friday","said monday",
                       "said tuesday","said saturday","said sunday","the statement said",
                       "officials said","the report said","news agency","news service",
                       "according to cbs","according to nbc","according to abc",
                       "according to reuters","according to ap ","reported by"]
    if sum(1 for p in ARTICLE_PHRASES if p in cl.lower()) >= 2: return True
    return False

def _article_verdict(claim, scope):
    bp    = bert_score_claim(claim)
    rp    = bp["real_prob"]
    boost = _credible_source_boost(claim)
    if boost > 0: rp = min(rp + boost, 0.99)
    fp = 1.0 - rp
    v  = "REAL" if rp >= OPTIMAL_THRESHOLD else "FAKE"
    print(f"  [ArticleFastPath] BERT={bp['label']}({bp['real_prob']:.3f}) boost={boost:.2f} -> {v}({rp:.3f})")
    return {"verdict":v, "supports":1 if v=="REAL" else 0, "refutes":1 if v=="FAKE" else 0,
            "skipped":0, "details":[], "scope":scope,
            "bert_prior":{"label":v,"fake_prob":round(fp,4),"real_prob":round(rp,4),
                          "confidence":round(max(rp,fp),4)},
            "factcheck_verdict":None, "wiki_alive":None, "claude_result":None,
            "source":"article_bert_ensemble", "hybrid_score":round(rp,3), "no_evidence":False}

def is_negated(c):
    if _is_article_input(c): return False
    return any(re.search(p, c.lower()) for p in NEGATION_PATTERNS)

def rewrite_claim_positive(c):
    for p, r in [
        (r"\bhas\s+not\s+been\b",""), (r"\bhave\s+not\s+been\b",""),
        (r"\bhas\s+not\b","has"),     (r"\bhave\s+not\b","have"),
        (r"\bdid\s+not\b",""),        (r"\bwas\s+not\b","was"),
        (r"\bwere\s+not\b","were"),   (r"\bis\s+not\b","is"),
        (r"\bare\s+not\b","are"),     (r"\bhasn'?t\b","has"),
        (r"\bhaven'?t\b","have"),     (r"\bdidn'?t\b",""),
        (r"\bisn'?t\b","is"),         (r"\baren'?t\b","are"),
        (r"\bwasn'?t\b","was"),       (r"\bweren'?t\b","were"),
        (r"\bno\s+longer\b",""),      (r"\bnever\b",""),
        (r"\bnot\b",""),              (r"\bcannot\b","can"),
        (r"\bcan'?t\b","can"),        (r"\bwon'?t\b","will"),
        (r"\bdoesn'?t\b","does"),     (r"\bdon'?t\b","do")]:
        c = re.sub(p, r, c, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}"," ",c).strip()

def detect_topic(c):
    cl = c.lower(); is_article = _is_article_input(c) if len(c)>50 else False
    for t, ps in TOPIC_PHRASES:
        if any(p in cl for p in ps): return t
    if "bill" in cl:
        LEGISLATIVE = {"congress","senate","house","republican","democrat","legislation",
                       "lawmakers","vote","passed","signed","veto","amendment","committee",
                       "parliament","lok sabha","assembly"}
        if any(w in cl for w in LEGISLATIVE): return "policy"
        if not is_article: return "policy"
    for t, ws in TOPIC_SINGLE.items():
        if any(w in cl for w in ws): return t
    return "general"

def extract_entities_spacy(t):
    doc = nlp_ner(t); seen, out = set(), []
    for e in doc.ents:
        if e.label_ in ("PERSON","ORG","GPE","LOC","NORP","FAC","EVENT"):
            n = e.text.strip()
            if len(n)>=3 and n.lower() not in NOT_NAMES and n.lower() not in seen:
                seen.add(n.lower()); out.append(n)
    return out

def extract_entities_regex(t):
    seen, out = set(), []
    for m in re.findall(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b',t):
        if len(m.split())==1 and m.lower() in NOT_NAMES: continue
        if len(m)>=3 and m.lower() not in seen: seen.add(m.lower()); out.append(m)
    return out

def extract_entities(c):
    n = normalise_claim(c); rm = apply_role_map(n)
    raw  = extract_entities_spacy(rm) if USE_SPACY else extract_entities_regex(rm)
    ents = expand_celeb_names(raw); cl = c.lower(); ncl = n.lower()
    seen_lower = {e.lower() for e in ents}
    for p in sorted(KNOWN_LIVING_PEOPLE, key=len, reverse=True):
        if (p in cl or p in ncl) and p not in seen_lower:
            ents.insert(0, p.title()); seen_lower.add(p); break
    all_words = set(cl.split())|set(ncl.split())
    for l, f in LAST_NAME_MAP.items():
        if l in all_words and f not in seen_lower:
            ents.append(f.title()); seen_lower.add(f)
    return ents

def extract_year_from_claim(c):
    m = re.search(r"\b(20\d{2}|19\d{2})\b",c); return m.group(1) if m else None

def extract_tournament_key(c):
    cl = c.lower()
    if "t20 world cup" in cl or ("t20" in cl and "world cup" in cl): return "t20 world cup"
    if "odi world cup" in cl or ("odi" in cl and "world cup" in cl): return "odi world cup"
    if "cricket world cup" in cl:       return "cricket world cup"
    if "world test championship" in cl or "wtc" in cl: return "world test championship"
    if "champions trophy" in cl:        return "champions trophy"
    if "ipl" in cl or "indian premier league" in cl: return "ipl"
    if "fifa world cup" in cl or ("football" in cl and "world cup" in cl): return "fifa world cup"
    if "champions league" in cl or "ucl" in cl: return "champions league"
    return None

def sports_winner_check(claim):
    yr = extract_year_from_claim(claim); trn = extract_tournament_key(claim)
    if not yr or not trn: return {"verdict":"UNCERTAIN","reason":"year or tournament not identified"}
    aw = SPORTS_WINNERS.get((trn, yr))
    if aw is None: return {"verdict":"UNCERTAIN","reason":f"no data for {trn} {yr}"}
    cl    = claim.lower()
    all_t = set(v for (t,y),v in SPORTS_WINNERS.items() if t==trn)
    wrong = [t for t in all_t if t!=aw and t in cl]
    if wrong and any(w in cl for w in ["won","win","wins","winner","champion","lifted","victory"]):
        return {"verdict":"FAKE","reason":f"Ground truth:{aw} won {trn} {yr}, not {wrong[0]}"}
    if aw in cl and any(w in cl for w in ["won","win","wins","winner","champion","lifted","victory"]):
        return {"verdict":"REAL","reason":f"Ground truth confirms:{aw} won {trn} {yr}"}
    return {"verdict":"UNCERTAIN","reason":f"winner={aw}, claim unclear"}

def extract_death_subject(c):
    cl = c.lower(); ncl = normalise_claim(c).lower()
    for p in sorted(KNOWN_LIVING_PEOPLE, key=len, reverse=True):
        if p in cl or p in ncl: return p
    all_words = set(cl.split())|set(ncl.split())
    for l, f in LAST_NAME_MAP.items():
        if l in all_words: return f
    if USE_SPACY:
        try:
            doc = nlp_ner(c)
            for ent in doc.ents:
                if ent.label_=="PERSON" and len(ent.text.split()) >= 1:
                    return ent.text.lower().strip()
        except: pass
    names  = re.findall(r'\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})+)\b',c)
    if names: return max(names, key=len).lower()
    SKIP_W = {"India","China","Russia","America","England","Pakistan","Australia",
              "January","February","March","April","June","July","August","September",
              "October","November","December","Sunday","Monday","Tuesday","Wednesday",
              "Thursday","Friday","Saturday","Breaking","According","Sources","Official",
              "Government","Report"}
    for w in re.findall(r'\b([A-Z][a-z]{4,})\b',c):
        if w not in SKIP_W: return w.lower()
    return ""

def person_present_in_evidence(e, p):
    ev = e.lower()
    if p in ev: return True
    last = p.split()[-1]; first = p.split()[0]
    if len(last)>=5 and last in ev: return True
    if len(first)>=6 and first in ev: return True
    return False

def evidence_names_person_dying(e, p):
    if not p or not person_present_in_evidence(e, p): return False
    ev = e.lower()
    if any(ph in ev for ph in DEATH_HOAX_PHRASES): return False
    if any(ph in ev for ph in TRIBUTE_PHRASES): return False
    act = any(w in ev for w in {"attends","attended","posts","posted","says","said","speaks",
                                  "announces","launches","appears","appeared","celebrates"})
    d   = any(w in ev for w in {"passed away","died","death","dead","passes away","no more",
                                  "demise","succumbs","breathed his last","breathed her last",
                                  "no longer with us","funeral","cremation","last rites","mourning"})
    return d and not act

def evidence_confirms_person_alive(e, p):
    if not p or not person_present_in_evidence(e, p): return False
    ev = e.lower()
    if any(w in ev for w in {"alive","alive and well","not dead","is alive","still alive",
                               "denies death","dismisses death report","clarifies is alive","healthy"}):
        return True
    return any(w in ev for w in {"releases","announces","signs","attends","launches","celebrates",
        "posts","shares","returns","shoots","wraps","completes","starring","upcoming film",
        "upcoming movie","new film","new movie","interviewed","speaks at","appeared at","birthday",
        "felicitated","awarded","receives award","new song","new album","spotted","seen at",
        "inaugurates","inaugurated","pre-wedding","wedding rituals","congratulates","cheers"})

def evidence_is_other_persons_death(e, p):
    ev = e.lower()
    d = any(w in ev for w in {"passed away","died","death","dead","passes away","no more",
                                "demise","succumbs","funeral","mourning","cremation","last rites",
                                "condolences","obituary"})
    return d and not person_present_in_evidence(e, p)
# =============================================================
# SOURCE SELECTION & FETCHING
# =============================================================
def parse_rss(url, name, limit=6):
    try:
        r = safe_get(url, timeout=FETCH_TIMEOUT)
        root = ElementTree.fromstring(r.content); out = []
        for it in root.iter("item"):
            t = clean_text(it.findtext("title") or ""); d = clean_text(it.findtext("description") or "")
            s = d if len(d)>len(t) else t
            if s and len(s)>20: out.append(s)
        print(f"    [{name}] -> {len(out)}"); return out[:limit]
    except Exception as e: print(f"    [{name}] Error:{e}"); return []

def fetch_google_rss(q): return parse_rss(f"https://news.google.com/rss/search?q={quote_plus(q)}&hl=en&gl=US&ceid=US:en",f"Google",6)
def fetch_bing(q):        return parse_rss(f"https://www.bing.com/news/search?q={quote_plus(q)}&format=rss","Bing",6)
def fetch_reuters(q):     return parse_rss("https://feeds.reuters.com/reuters/topNews","Reuters",5)
def fetch_ap(q):          return parse_rss("https://feeds.apnews.com/rss/apf-topnews","AP",5)
def fetch_bbc_world(q):   return parse_rss("http://feeds.bbci.co.uk/news/world/rss.xml","BBC-World",5)
def fetch_thehindu(q):    return parse_rss("https://www.thehindu.com/news/feeder/default.rss","TheHindu",5)
def fetch_ndtv(q):        return parse_rss("https://feeds.feedburner.com/ndtvnews-top-stories","NDTV",5)
def fetch_aljazeera(q):   return parse_rss("https://www.aljazeera.com/xml/rss/all.xml","AlJazeera",5)
def fetch_skynews(q):     return parse_rss("https://feeds.skynews.com/feeds/rss/world.xml","SkyNews",4)
def fetch_cricinfo(q):    return parse_rss("https://www.espncricinfo.com/rss/content/story/feeds/0.xml","Cricinfo",5)
def fetch_pinkvilla(q):   return parse_rss("https://www.pinkvilla.com/rss.xml","Pinkvilla",6)
def fetch_bh(q):          return parse_rss("https://www.bollywoodhungama.com/rss/news.xml","BH",6)
def fetch_google_celeb(q):return parse_rss(f"https://news.google.com/rss/search?q={quote_plus(q+' wedding OR married OR marriage')}&hl=en&gl=IN&ceid=IN:en","GoogleCeleb",6)
def fetch_ddg(q):
    try:
        d = _SESSION.get(f"http://api.duckduckgo.com/?q={quote_plus(q)}&format=json&no_redirect=1&no_html=1&skip_disambig=1",timeout=FETCH_TIMEOUT).json()
        out = []; ab = clean_text(d.get("AbstractText",""))
        if ab and len(ab)>20: out.append(ab)
        for r in d.get("RelatedTopics",[]):
            t = clean_text(r.get("Text",""))
            if t and len(t)>20: out.append(t)
        return out[:5]
    except Exception as e: print(f"    [DDG] {e}"); return []

def _build_smart_query(claim, topic):
    cl_raw    = claim.strip(); cl = cl_raw.lower()
    entities  = extract_entities(claim)
    countries = [w.capitalize() for w in re.findall(r"[a-zA-Z]+",cl) if w in COUNTRY_NAMES]
    named     = list(dict.fromkeys(entities+countries))[:3]
    q1 = re.sub(r"[^\x00-\x7F]"," ",cl_raw); q1 = re.sub(r"\s{2,}"," ",q1).strip()
    if len(q1) > 200: q1 = q1[:200].rsplit(" ",1)[0]
    all_kws   = extract_keywords(claim)
    ent_words = set(w.lower() for e in named for w in e.split())
    high_kws  = [w for w in all_kws if (w in PRESERVE_WORDS or len(w)>=5) and w not in ent_words]
    parts2    = (([" ".join(named[:2])] if named else []) +
                  sorted(high_kws, key=len, reverse=True)[:5])
    q2 = " ".join(parts2).strip()
    topic_hint = {"death":"death died","marriage":"wedding married","election":"election result",
                  "war":"attack military","arrest":"arrested","sports":"champion winner",
                  "disaster":"earthquake flood","policy":"government law rule",
                  "health":"health hospital","economy":"economy inflation",
                  "science":"discovered research space","general":""}.get(topic,"")
    q3 = f"{q2} {topic_hint}".strip() if topic_hint and q2 else q2
    queries = []
    for q in [q1,q2,q3]:
        q = re.sub(r"\s{2,}"," ",q).strip()
        if q and q not in queries: queries.append(q)
    return queries[:3], named

def select_sources(claim, topic):
    cl = claim.lower(); queries, entities = _build_smart_query(claim, topic)
    q1 = queries[0]; q2 = queries[1] if len(queries)>1 else q1
    tasks = ([(fetch_newsapi,q1)] + [(fetch_google_rss,q1),(fetch_google_rss,q2)] +
             [(fetch_bing,q1)] + [(fetch_ddg,q1),(fetch_ddg,q2)])
    if topic in ("war","disaster") or any(w in cl for w in ["ukraine","russia","missile"]):
        tasks += [(fetch_aljazeera,q1),(fetch_skynews,q1)]
    elif topic=="science" or any(w in cl for w in ["isro","nasa","chandrayaan","satellite","spacecraft",
                                                     "moon","mars","space","orbit","revolves","orbits"]):
        tasks += [(fetch_thehindu,q1),(fetch_ndtv,q1),(fetch_bbc_world,q1),(fetch_reuters,q1)]
    elif topic in ("death","election","policy","arrest") or any(w in cl for w in ["india","modi","rbi"]):
        tasks += [(fetch_thehindu,q1),(fetch_ndtv,q1)]
    elif topic=="sports" or any(w in cl for w in ["cricket","ipl","bcci","icc","t20","odi","captain","skipper"]):
        tasks += [(fetch_cricinfo,q2),(fetch_bbc_world,q1),(fetch_thehindu,q1)]
    elif is_celebrity_personal_claim(claim) or any(w in cl for w in CELEB_KEYWORDS):
        subject = " ".join(entities[:2]) if entities else q2
        tasks  += [(fetch_pinkvilla,q1),(fetch_bh,q1),(fetch_google_celeb,subject)]
    else:
        tasks += [(fetch_reuters,q1),(fetch_ap,q1)]
    return tasks, queries, entities

def get_evidence_fast(claim):
    topic = detect_topic(claim); tasks, queries, entities = select_sources(claim, topic)
    STOPWORDS = {"the","a","an","and","or","in","on","at","to","for","of","with","by","from","as",
                 "is","are","was","were","be","been","have","has","had","will","would","that","this",
                 "it","its","he","she","they","who","any","their","also","just","says","said","not","been"}
    raw_cl   = re.sub(r"[^\x00-\x7F]"," ",claim).lower()
    hyph     = [h.lower() for h in re.findall(r"[a-zA-Z][a-zA-Z0-9]*-[a-zA-Z0-9]+",claim)]
    full_words       = [w for w in re.findall(r"[a-z]{3,}",raw_cl) if w not in STOPWORDS]
    kws_from_extract = extract_keywords(claim)
    nw               = [w.lower() for n in entities for w in n.split() if len(w)>=3]
    all_kw           = list(dict.fromkeys(hyph+full_words+kws_from_extract+nw))
    all_texts = []; start_time = time.time()
    executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
    try:
        future_map = {executor.submit(fn, q): fn.__name__ for fn, q in tasks}
        try:
            for f in as_completed(future_map, timeout=EVIDENCE_TIMEOUT):
                try: all_texts.extend(f.result(timeout=0.5))
                except: pass
                if time.time()-start_time >= EVIDENCE_TIMEOUT: break
        except FuturesTimeout: pass
        for f in future_map:
            if f.done():
                try: all_texts.extend(f.result(timeout=0))
                except: pass
    finally: executor.shutdown(wait=False)
    seen, unique = set(), []
    for t in all_texts:
        k = t[:80].lower()
        if k not in seen: seen.add(k); unique.append(t)
    if all_kw and unique:
        rel = [e for e in unique if semantic_relevance_score(e,all_kw) >= 0.45]
        if not rel: rel = [e for e in unique if semantic_relevance_score(e,all_kw) >= 0.30]
        if not rel: rel = unique
    else: rel = unique
    rel.sort(key=lambda e: semantic_relevance_score(e,all_kw), reverse=True)
    top = rel[:MAX_EVIDENCE]; top = [extract_best_sentence(e,claim) for e in top]
    seen_sent, unique_top = set(), []
    for s in top:
        k = s[:60].lower()
        if k not in seen_sent: seen_sent.add(k); unique_top.append(s)
    print(f"  After filter: {len(unique_top)} evidence pieces")
    return unique_top, queries, entities

# =============================================================
# NLI
# =============================================================
TIME_RE = re.compile(
    r"\b(today|yesterday|tomorrow|tonight|this\s+morning|this\s+evening|right\s+now|currently|"
    r"at\s+present|on\s+\w+day|this\s+week|this\s+month|this\s+year|january|february|march|april|"
    r"may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|"
    r"oct|nov|dec)\b", re.IGNORECASE)

def normalize_for_nli(t): return re.sub(TIME_RE,"",t)

def _nli_keyword_fallback(evidence, claim):
    ev = evidence.lower().replace("deepfake","deepfake_ai").replace("fake news","fakenews")
    cl = claim.lower()
    sem       = semantic_overlap(cl, ev)
    STRONG_DENY = {"denied","denies","false","hoax","rumor","baseless","incorrect",
                   "no truth","not true","wrong","disputed","misinformation","debunked"}
    STRONG_CONF = {"confirmed","verified","true","official","announced","took place",
                   "happened","occurred","reports","sources"}
    deny_score = sum(1 for w in STRONG_DENY if w in ev)
    conf_score = sum(1 for w in STRONG_CONF if w in ev)
    if deny_score>=2 and sem>=0.08: return "CONTRADICTION", min(0.55+deny_score*0.05,0.90)
    if conf_score>=1 and sem>=0.12: return "ENTAILMENT",   min(0.50+sem*0.5,0.85)
    if sem>=0.35: return "ENTAILMENT", 0.55
    if sem>=0.18: return "NEUTRAL", 0.50
    return "NEUTRAL", 0.40

def run_nli(evidence, claim):
    cn = normalize_for_nli(normalise_claim(claim)).strip()[:100]
    t  = clean_text(evidence)[:NLI_MAX_CHARS]
    if NLI_BACKEND=="pipeline" and NLI is not None:
        try:
            raw   = NLI(t+" </s></s> "+cn)
            items = raw[0] if (raw and isinstance(raw[0],list)) else raw
            scores= {i["label"].upper(): i["score"] for i in items if "label" in i}
            e = scores.get("ENTAILMENT",scores.get("LABEL_2",0.0))
            c = scores.get("CONTRADICTION",scores.get("LABEL_0",0.0))
            n = scores.get("NEUTRAL",scores.get("LABEL_1",0.0))
            if e>0.30: return "ENTAILMENT",   e
            if c>0.55: return "CONTRADICTION", c
            return "NEUTRAL", n
        except Exception as ex: print(f"  [NLI pipeline] {ex}")
    return _nli_keyword_fallback(evidence, claim)

def get_confirm_words(t):
    return TOPIC_CONFIRM_WORDS.get(t, TOPIC_CONFIRM_WORDS.get("general", set()))

def get_deny_words(t):
    return TOPIC_DENY_WORDS.get(t, TOPIC_DENY_WORDS.get("general", set()))

def fast_keyword_score(evidence, negated, topic, claim=""):
    ev = evidence.lower().replace("deepfake","deepfake_ai").replace("fake news","fakenews")
    ch = sum(1 for w in get_confirm_words(topic) if w in ev)
    dh = sum(1 for w in get_deny_words(topic)    if w in ev)
    if topic=="sports" and claim:
        cy = extract_year_from_claim(claim); ey = re.findall(r"\b(20\d{2}|19\d{2})\b",ev)
        if cy and ey and cy not in ey: return "SKIP"
    if not negated:
        if ch>=2 and dh==0: return "SUPPORTS"
        if dh>=2 and ch==0: return "REFUTES"
    else:
        if ch>=2 and dh==0: return "REFUTES"
        if dh>=2 and ch==0: return "SUPPORTS"
    return "SKIP"

def score_piece_full(evidence, overlap, nli_label, negated, topic, claim="", bert_ev=None):
    ev = evidence.lower().replace("deepfake","deepfake_ai").replace("fake news","fakenews")
    if overlap < 0.12: return "SKIP"
    if claim:
        claim_entities = extract_entities(claim)
        if claim_entities:
            if not any(ent.lower() in ev for ent in claim_entities) and overlap < 0.12:
                return "SKIP"
    if claim and not relation_match(claim, ev): return "SKIP"
    ch = sum(1 for w in get_confirm_words(topic) if w in ev)
    dh = sum(1 for w in get_deny_words(topic)    if w in ev)
    br = bert_ev["real_prob"] if bert_ev else 0.5
    bf = bert_ev["fake_prob"] if bert_ev else 0.5
    rb = br>0.70; fb = bf>0.70
    if topic=="science" and overlap < 0.20: return "SKIP"
    if topic=="death" and claim:
        ds = extract_death_subject(claim)
        if ds:
            if negated:
                if evidence_confirms_person_alive(evidence,ds): return "SUPPORTS"
                if evidence_names_person_dying(evidence,ds):    return "REFUTES"
                if evidence_is_other_persons_death(evidence,ds):return "SKIP"
                if person_present_in_evidence(evidence,ds):
                    ev2 = evidence.lower()
                    if any(ph in ev2 for ph in DEATH_HOAX_PHRASES): return "SUPPORTS"
                    if any(w in ev2 for w in {"announces","launches","attends","posts","says",
                                               "speaks","appears","celebrated","releases","shoots",
                                               "starring","upcoming","inaugurates","awarded"}):
                        return "SUPPORTS"
                return "SKIP"
            else:
                if evidence_names_person_dying(evidence,ds):    return "SUPPORTS"
                if evidence_confirms_person_alive(evidence,ds): return "REFUTES"
                if evidence_is_other_persons_death(evidence,ds):return "SKIP"
                if person_present_in_evidence(evidence,ds):     return "SKIP"
                if ds in KNOWN_LIVING_PEOPLE:
                    return "SUPPORTS" if (ch>=2 and dh==0 and overlap>=0.22) else "SKIP"
    if topic=="sports" and not negated and claim:
        cy  = extract_year_from_claim(claim); ey = re.findall(r"\b(20\d{2}|19\d{2})\b",ev)
        cl  = claim.lower()
        ct  = [t for t in CRICKET_TEAMS+["france","argentina","germany","brazil","spain",
               "mumbai indians","chennai super kings","kolkata knight riders","royal challengers",
               "sunrisers","rajasthan royals","delhi capitals","punjab kings",
               "lucknow","gujarat titans","deccan chargers"] if t in cl]
        if ct and cy:
            pt = ct[0]
            if ey and cy not in ey: return "SKIP"
            etc = pt in ev and any(w in ev for w in ["won","champion","winner","title","trophy","lifted","victory"])
            tk  = extract_tournament_key(claim)
            eho = False
            if tk:
                ot = [t for t in CRICKET_TEAMS if t!=pt and t in ev and
                      any(w in ev for w in ["won","champion","winner","title","trophy","lifted"])]
                eho = bool(ot)
            if eho: return "REFUTES"
            if etc: return "SUPPORTS"
    if topic=="election" and not negated and claim:
        cl = claim.lower()
        cj = any(w in cl for w in ["bjp","narendra modi","nda won","nda wins","modi sworn"])
        cb = any(w in cl for w in ["congress party","rahul gandhi","inc won","congress won"])
        ca = any(w in cl for w in ["aam aadmi party","aap won","aap wins","kejriwal sworn"])
        eb = any(w in ev for w in ["bjp wins","bjp won","nda wins","modi wins","modi sworn","bjp majority"])
        ec = any(w in ev for w in ["congress wins","congress won","rahul gandhi sworn","congress majority"])
        ea = any(w in ev for w in ["aap wins","aap won","kejriwal sworn","aap majority"])
        if cj and eb: return "SUPPORTS"
        if cb and ec: return "SUPPORTS"
        if ca and ea: return "SUPPORTS"
        if cb and eb: return "REFUTES"
        if cj and ec: return "REFUTES"
    if not negated:
        if ch>=2 and dh==0:                              return "SUPPORTS"
        if ch>=1 and dh==0 and overlap>=0.12:           return "SUPPORTS"
        if nli_label=="ENTAILMENT" and overlap>=0.18:   return "SUPPORTS"
        if rb and ch>=1 and dh==0:                       return "SUPPORTS"
        if rb and nli_label=="ENTAILMENT" and overlap>=0.07: return "SUPPORTS"
        if dh>=2 and ch==0:                              return "REFUTES"
        if dh>=1 and ch==0 and nli_label=="CONTRADICTION": return "REFUTES"
        if nli_label=="CONTRADICTION" and overlap>=0.17: return "REFUTES"
        if fb and dh>=1 and ch==0:                       return "REFUTES"
        if fb and nli_label=="CONTRADICTION" and overlap>=0.12: return "REFUTES"
        if overlap>=0.38 and nli_label=="ENTAILMENT":   return "SUPPORTS"
        if overlap>=0.38 and nli_label=="CONTRADICTION":return "REFUTES"
        return "SKIP"
    else:
        if ch>=2 and dh==0:                              return "REFUTES"
        if ch>=1 and nli_label=="ENTAILMENT":            return "REFUTES"
        if dh>=2 and ch==0:                              return "SUPPORTS"
        if nli_label=="ENTAILMENT" and overlap>=0.12:   return "REFUTES"
        if nli_label=="CONTRADICTION" and overlap>=0.12: return "SUPPORTS"
        if rb and ch>=1:                                  return "REFUTES"
        if fb and dh>=1:                                  return "SUPPORTS"
        if overlap>=0.27: return "REFUTES" if nli_label=="ENTAILMENT" else "SUPPORTS"
        return "SKIP"

CELEB_CONFIRM = {"married","wedding","tie the knot","tied the knot","confirms","confirmed",
    "officially","announced","ceremony","wedded","got married","engagement","engaged",
    "nuptials","husband","wife","spouse"}
CELEB_DENY    = {"rumor","rumour","denies","denied","not married","just friends","baseless",
    "clarifies","false","no truth","speculation","not true"}

def score_piece_celeb(e, overlap, nli_label):
    ev = e.lower()
    ch = sum(1 for w in CELEB_CONFIRM if w in ev)
    dh = sum(1 for w in CELEB_DENY   if w in ev)
    if ch>=2 and dh==0:                             return "SUPPORTS"
    if ch>=1 and dh==0 and overlap>=0.12:           return "SUPPORTS"
    if dh>=2 and ch==0:                             return "REFUTES"
    if dh>=1 and ch==0 and nli_label=="CONTRADICTION": return "REFUTES"
    if overlap>=0.27: return "REFUTES" if nli_label=="CONTRADICTION" else "SUPPORTS"
    if nli_label=="ENTAILMENT" and overlap>=0.25:   return "SUPPORTS"
    return "SKIP"

def evidence_majority_verdict(supports, refutes):
    total = supports+refutes
    if total==0: return None
    if supports>=2 and refutes==0: return "REAL"
    if refutes>=2  and supports==0: return "FAKE"
    if supports>=3 and supports>refutes*2.5: return "REAL"
    if refutes>=4  and refutes>supports*3:  return "FAKE"
    if supports-refutes>=2: return "REAL"
    if supports>=1 and refutes>=1: return PARTIALLY_TRUE
    return None

def _make_verdict(v, **kw):
    base = {"verdict":v,"supports":0,"refutes":0,"skipped":0,"details":[],
            "bert_prior":None,"factcheck_verdict":None,"wiki_alive":None,
            "claude_result":None,"hybrid_score":None,"no_evidence":False,
            "source":"unknown","sports_reason":""}
    base.update(kw)
    return base

def compute_verdict(claim, evidences, scope, factcheck_result, wiki_result):
    if scope == "local":
        return _make_verdict("UNCERTAIN", scope=scope, source="local_scope")
    if _is_article_input(claim):
        print(f"  [Pipeline] Article ({len(claim.split())} words) -> BERT fast-path")
        return _article_verdict(claim, scope)
    negated = is_negated(claim)
    nc      = normalise_claim(claim)
    nli_c   = rewrite_claim_positive(nc) if negated else nc
    icp     = is_celebrity_personal_claim(claim)
    topic   = detect_topic(claim)
    ws      = wiki_result.get("wiki_summary", "")
    wa      = wiki_result.get("is_alive")
    fc_v    = factcheck_result.get("verdict")
    print(f"\n  [Pipeline] topic={topic} negated={negated} icp={icp}")
    kf_verdict = _check_known_facts(claim, negated, nli_c)
    if kf_verdict:
        print(f"  [Step1-KnownFacts] HIT -> {kf_verdict}")
        return _make_verdict(kf_verdict, scope=scope,
                             supports=1 if kf_verdict=="REAL" else 0,
                             refutes =1 if kf_verdict=="FAKE" else 0,
                             source="known_facts_json")
    kb_verdict = local_kb_lookup(claim)
    if kb_verdict is None and negated:
        kbp = local_kb_lookup(nli_c)
        if kbp:
            kb_verdict = {"REAL":"FAKE","FAKE":"REAL"}.get(kbp, kbp)
    if kb_verdict in ("REAL","FAKE","PARTIALLY_TRUE"):
        print(f"  [Step2-KnowledgeBase] HIT -> {kb_verdict}")
        return _make_verdict(kb_verdict, scope=scope,
                             supports=1 if kb_verdict=="REAL" else 0,
                             refutes =1 if kb_verdict=="FAKE" else 0,
                             source="knowledge_base_json")
    if topic == "sports":
        sw = sports_winner_check(claim)
        if sw["verdict"] in ("REAL","FAKE"):
            print(f"  [Step3-SportsKB] HIT -> {sw['verdict']}")
            return _make_verdict(sw["verdict"], scope=scope,
                                 source="sports_kb", sports_reason=sw["reason"])
    cc = captain_check(claim)
    if cc["verdict"] in ("REAL","FAKE"):
        print(f"  [Step3-CaptainKB] HIT -> {cc['verdict']}")
        return _make_verdict(cc["verdict"], scope=scope,
                             supports=1 if cc["verdict"]=="REAL" else 0,
                             refutes =1 if cc["verdict"]=="FAKE" else 0,
                             source="captain_kb", sports_reason=cc["reason"])
    rc = role_check(claim)
    if rc["verdict"] in ("REAL","FAKE"):
        print(f"  [Step3-RoleKB] HIT -> {rc['verdict']}")
        return _make_verdict(rc["verdict"], scope=scope,
                             supports=1 if rc["verdict"]=="REAL" else 0,
                             refutes =1 if rc["verdict"]=="FAKE" else 0,
                             source="role_kb", sports_reason=rc["reason"])
    if topic == "death" and wa is True:
        ds = extract_death_subject(claim)
        if ds:
            v = "REAL" if negated else "FAKE"
            print(f"  [Step3-WikiAlive] {ds} is alive -> {v}")
            return _make_verdict(v, scope=scope,
                                 supports=1 if v=="REAL" else 0,
                                 refutes =1 if v=="FAKE" else 0,
                                 wiki_alive=True, source="wikipedia_alive_check")
    print(f"  [Step4] Not in KB — fetching live evidence via APIs...")
    if fc_v in ("REAL","FAKE"):
        print(f"  [Step4-FactCheckAPI] -> {fc_v}")
        return _make_verdict(fc_v, scope=scope,
                             factcheck_verdict=fc_v, wiki_alive=wa,
                             source="factcheck_api")
    if not evidences or len(evidences) < 2:
        wiki_ev = fetch_wikipedia_evidence(claim)
        if wiki_ev:
            print(f"  [Step4-Wikipedia] +{len(wiki_ev)} sentences")
            evidences = wiki_ev + list(evidences)
    if topic == "science":
        cr_s = call_claude_ai_knowledge(claim, ws, factcheck_result.get("claims",[]))
        if cr_s["used"] and cr_s["verdict"] in ("REAL","FAKE"):
            print(f"  [Step4-ScienceKnowledge] -> {cr_s['verdict']}")
            return _make_verdict(cr_s["verdict"], scope=scope,
                                 claude_result=cr_s, source="science_knowledge")
    if not evidences:
        print(f"  [Step4] No evidence found -> Gemini fallback")
        cr = call_claude_ai_knowledge(claim, ws, factcheck_result.get("claims",[]))
        if cr["used"] and cr["verdict"] in ("REAL","FAKE", PARTIALLY_TRUE):
            return _make_verdict(cr["verdict"], scope=scope,
                                 claude_result=cr, source="gemini_no_evidence")
        return _make_verdict("UNCERTAIN", scope=scope,
                             claude_result=cr, source="no_evidence", no_evidence=True)
    print(f"  [Step5] Scoring {len(evidences)} evidence pieces...")
    bp  = bert_score_claim(claim)
    br  = bp["real_prob"]
    ml  = BERT_LOADED or CSV_FALLBACK_LOADED
    nkw = extract_keywords(claim)
    ev_sorted = sorted(evidences, key=lambda e: semantic_relevance_score(e, nkw), reverse=True)
    top_n  = ev_sorted[:NLI_BERT_TOP_N]
    rest   = ev_sorted[NLI_BERT_TOP_N:]
    supports = refutes = skipped = 0
    details  = []
    for ev in top_n:
        ov       = semantic_overlap(nli_c, ev)
        lbl, conf= run_nli(ev, nli_c)
        be       = bert_score_evidence(ev, nli_c)
        dec      = (score_piece_celeb(ev, ov, lbl) if icp else
                    score_piece_full(ev, ov, lbl, negated, topic, nli_c, bert_ev=be))
        details.append({"text":ev[:180],"overlap":round(ov,3),"nli":lbl,
                        "conf":round(conf,3),"bert_label":be["label"],
                        "bert_real":be["real_prob"],"bert_fake":be["fake_prob"],
                        "decision":dec,"method":"nli+bert"})
        if   dec=="SUPPORTS": supports += 1
        elif dec=="REFUTES":  refutes  += 1
        else:                 skipped  += 1
        print(f"    [NLI+BERT] overlap={ov:.3f} nli={lbl:14s} bert={be['label']:4s} -> {dec}")
    for ev in rest:
        dec = fast_keyword_score(ev, negated, topic, nli_c)
        details.append({"text":ev[:180],"overlap":0,"nli":"FAST","conf":0,
                        "bert_label":"N/A","bert_real":0,"bert_fake":0,
                        "decision":dec,"method":"keyword"})
        if   dec=="SUPPORTS": supports += 1
        elif dec=="REFUTES":  refutes  += 1
        else:                 skipped  += 1
    total    = supports + refutes
    ev_ratio = (supports / total) if total > 0 else 0.5
    hs       = (EVIDENCE_WEIGHT*ev_ratio + BERT_WEIGHT*br) if ml else ev_ratio
    print(f"  [Step6-Hybrid] S={supports} R={refutes} SK={skipped} "
          f"ev_ratio={ev_ratio:.3f} bert={br:.3f} score={hs:.3f}")
    if supports == 0 and refutes == 0:
        print(f"  [Step6] All evidence skipped -> Gemini fallback")
        cr = call_claude_ai(claim, [d["text"] for d in details[:8]],
                            factcheck_result.get("claims",[]), ws, bp)
        fv = cr["verdict"] if (cr["used"] and
                               cr["verdict"] in ("REAL","FAKE","UNCERTAIN",PARTIALLY_TRUE)) \
             else "UNCERTAIN"
        return _make_verdict(fv, scope=scope, supports=0, refutes=0, skipped=skipped,
                             details=details, bert_prior=bp, factcheck_verdict=fc_v,
                             wiki_alive=wa, claude_result=cr,
                             source="gemini_all_skipped", hybrid_score=None)
    mv = evidence_majority_verdict(supports, refutes)
    if mv in ("REAL","FAKE", PARTIALLY_TRUE):
        final = mv
    else:
        if   hs >= 0.65: final = "REAL"
        elif hs <= 0.25: final = "FAKE"
        else:            final = "UNCERTAIN"
    cr = None
    if final == "UNCERTAIN":
        print(f"  [Step7-Gemini] Hybrid inconclusive -> asking Gemini...")
        cr = call_claude_ai(claim, [d["text"] for d in details[:8]],
                            factcheck_result.get("claims",[]), ws, bp)
        if cr["used"] and cr["verdict"] in ("REAL","FAKE", PARTIALLY_TRUE):
            final = cr["verdict"]
            print(f"  [Step7-Gemini] -> {final} ({cr.get('confidence','?')})")
    return _make_verdict(final, scope=scope, supports=supports, refutes=refutes,
                         skipped=skipped, details=details, bert_prior=bp,
                         factcheck_verdict=fc_v, wiki_alive=wa, claude_result=cr,
                         source="hybrid_nli_bert_pipeline", hybrid_score=round(hs,3))

def _build_warning(icp, topic, verdict, no_evidence=False, source=""):
    if verdict==PARTIALLY_TRUE:
        return "Evidence both supports and contradicts this claim. Verify specific details from trusted sources."
    if source=="local_knowledge_base": return None
    if source=="science_knowledge_check":
        return "Verdict based on established scientific consensus and AI training knowledge."
    if no_evidence and verdict=="UNCERTAIN":
        return "No live news evidence found. Please verify this claim manually from trusted sources."
    if no_evidence and verdict=="FAKE":
        return "No supporting evidence found. Marked FAKE by AI knowledge — verify from trusted sources."
    if source=="claude_knowledge":
        return "No live news found — verdict based on AI training knowledge. Verify recent events manually."
    if "wikipedia" in source:
        return "No recent news found — verdict based on Wikipedia. Accurate for established facts."
    fok = FACTCHECK_API_KEY and "YOUR" not in FACTCHECK_API_KEY
    nok = NEWSAPI_KEY       and "YOUR" not in NEWSAPI_KEY
    cok = GEMINI_SDK and GEMINI_API_KEY and "YOUR" not in GEMINI_API_KEY
    miss = [n for n, ok in [("FactCheck",fok),("NewsAPI",nok),("Gemini AI",cok)] if not ok]
    if miss: return f"Missing API keys:{', '.join(miss)}. Add to app.py for full accuracy."
    if not BERT_LOADED and not CSV_FALLBACK_LOADED:
        return "No ML model loaded. Place merged_news.csv in app directory to auto-train."
    if NLI_BACKEND=="keyword":
        return "NLI model not loaded. Using keyword fallback — accuracy may be lower."
    if icp and verdict=="UNCERTAIN":
        return "Celebrity personal claim — insufficient public evidence. Verify from official sources."
    if topic in UNCERTAIN_DEFAULT_TOPICS and verdict=="UNCERTAIN":
        return "Evidence was mixed or insufficient. Verify from official sources."
    return None
# =============================================================
# ENDPOINTS
# =============================================================
@app.route("/verify", methods=["POST","OPTIONS"])
def verify():
    if request.method=="OPTIONS": return "", 200
    t0 = time.time()
    try:
        body  = request.get_json(silent=True) or {}; claim = body.get("claim","").strip()
        if not claim: return jsonify({"error":"No claim provided"}), 400
        cached = _cache.get(claim)
        if cached: cached["cached"] = True; return jsonify(cached)
        topic    = detect_topic(claim); negated  = is_negated(claim)
        scope    = classify_claim_scope(claim);   icp = is_celebrity_personal_claim(claim)
        entities = extract_entities(claim)
        print(f"\n{'='*60}\n[Claim] {claim[:75]}\n[Topic] {topic} [Neg] {negated} [Scope] {scope}")
        if scope=="local":
            resp = {"verdict":"UNCERTAIN","supports":0,"refutes":0,"skipped":0,"topic":topic,
                    "names":entities,"negated":negated,"queries":[],"evidences":[],"scope":"local",
                    "bert_loaded":BERT_LOADED,"csv_fallback":CSV_FALLBACK_LOADED,"cached":False,
                    "elapsed":round(time.time()-t0,2),"warning":"Local/institutional claim -- verify manually."}
            return jsonify(resp)
        with ThreadPoolExecutor(max_workers=3) as ex:
            fc_f = ex.submit(fetch_google_factcheck, claim)
            wk_f = ex.submit(wiki_alive_check, claim, entities)
            ev_f = ex.submit(get_evidence_fast, claim)
            try: fcr = fc_f.result(timeout=20)
            except: fcr = {"found":False,"verdict":None,"claims":[]}
            try: wkr = wk_f.result(timeout=20)
            except: wkr = {"is_alive":None,"wiki_summary":"","wiki_url":""}
            try: evs, queries, names = ev_f.result(timeout=20)
            except: evs, queries, names = [], [], entities
        result  = compute_verdict(claim, evs, scope, fcr, wkr)
        v       = result["verdict"]; cr = result.get("claude_result")
        no_ev   = result.get("no_evidence",False)
        elapsed = round(time.time()-t0,2)
        warn    = _build_warning(icp, topic, v, no_ev, result.get("source",""))
        resp = {"verdict":v,"supports":result["supports"],"refutes":result["refutes"],
                "skipped":result["skipped"],"topic":topic,"names":names,"negated":negated,
                "queries":queries,"evidences":result["details"],"scope":scope,
                "bert_loaded":BERT_LOADED,"csv_fallback":CSV_FALLBACK_LOADED,"nli_backend":NLI_BACKEND,
                "bert_prior":result.get("bert_prior"),"factcheck_verdict":result.get("factcheck_verdict"),
                "factcheck_claims":fcr.get("claims",[]),"wiki_alive":result.get("wiki_alive"),
                "wiki_url":wkr.get("wiki_url",""),
                "claude_ai_used":bool(cr and cr.get("used")),
                "claude_explanation":cr.get("explanation","") if cr else "",
                "hybrid_score":result.get("hybrid_score"),
                "pipeline_source":result.get("source","hybrid_pipeline"),
                "sports_reason":result.get("sports_reason",""),
                "no_evidence":no_ev,"elapsed":elapsed,"cached":False,
                **({"warning":warn} if warn else {})}
        _cache.set(claim, resp); return jsonify(resp)
    except Exception as e:
        traceback.print_exc(); return jsonify({"error":str(e)}), 500

@app.route("/verify/stream", methods=["POST","OPTIONS"])
def verify_stream():
    if request.method=="OPTIONS": return "", 200
    body = request.get_json(silent=True) or {}; claim = body.get("claim","").strip()
    if not claim: return jsonify({"error":"No claim provided"}), 400
    def generate():
        t0 = time.time()
        def send(ev, d):
            pl = json.dumps({**d,"elapsed":round(time.time()-t0,2)})
            yield f"event: {ev}\ndata: {pl}\n\n"
        cached = _cache.get(claim)
        if cached: cached["cached"] = True; yield from send("result",cached); return
        topic    = detect_topic(claim); negated = is_negated(claim)
        scope    = classify_claim_scope(claim); entities = extract_entities(claim)
        yield from send("progress",{"stage":"Starting verification...","pct":5})
        if scope=="local":
            yield from send("result",{"verdict":"UNCERTAIN","topic":topic,"names":entities,
                                       "cached":False,"warning":"Local/institutional claim -- verify manually."}); return
        yield from send("progress",{"stage":"Checking fact databases & Wikipedia...","pct":15})
        with ThreadPoolExecutor(max_workers=2) as ex:
            fc_f = ex.submit(fetch_google_factcheck, claim)
            wk_f = ex.submit(wiki_alive_check, claim, entities)
            try: fcr = fc_f.result(timeout=15)
            except: fcr = {"found":False,"verdict":None,"claims":[]}
            try: wkr = wk_f.result(timeout=15)
            except: wkr = {"is_alive":None,"wiki_summary":"","wiki_url":""}
        fc_v = fcr.get("verdict")
        yield from send("progress",{"stage":f"Fact databases done. FactCheck={fc_v or 'no match'}","pct":35})
        if fc_v in ("REAL","FAKE"):
            resp = {"verdict":fc_v,"supports":0,"refutes":0,"skipped":0,"topic":topic,
                    "names":entities,"negated":negated,"queries":[],"evidences":[],"scope":scope,
                    "bert_loaded":BERT_LOADED,"csv_fallback":CSV_FALLBACK_LOADED,
                    "factcheck_verdict":fc_v,"factcheck_claims":fcr.get("claims",[]),
                    "wiki_alive":wkr.get("is_alive"),"claude_ai_used":False,
                    "claude_explanation":"","hybrid_score":None,
                    "pipeline_source":"factcheck_api","cached":False}
            _cache.set(claim,resp); yield from send("result",resp); return
        yield from send("progress",{"stage":"Searching news sources...","pct":40})
        evs, queries, names = get_evidence_fast(claim)
        yield from send("progress",{"stage":f"Found {len(evs)} articles. Running analysis...","pct":65})
        result = compute_verdict(claim, evs, scope, fcr, wkr); v = result["verdict"]
        yield from send("progress",{"stage":f"Analysis done. Verdict:{v}","pct":90 if v!="UNCERTAIN" else 75})
        if result.get("claude_result") and result["claude_result"].get("used"):
            yield from send("progress",{"stage":"Gemini AI reasoning...","pct":95})
        elapsed = round(time.time()-t0,2); icp = is_celebrity_personal_claim(claim)
        no_ev = result.get("no_evidence",False)
        warn = _build_warning(icp,topic,v,no_ev,result.get("source","")); cr = result.get("claude_result")
        resp = {"verdict":v,"supports":result["supports"],"refutes":result["refutes"],
                "skipped":result["skipped"],"topic":topic,"names":names,"negated":negated,
                "queries":queries,"evidences":result["details"],"scope":scope,
                "bert_loaded":BERT_LOADED,"csv_fallback":CSV_FALLBACK_LOADED,"nli_backend":NLI_BACKEND,
                "bert_prior":result.get("bert_prior"),"factcheck_verdict":result.get("factcheck_verdict"),
                "factcheck_claims":fcr.get("claims",[]),"wiki_alive":result.get("wiki_alive"),
                "wiki_url":wkr.get("wiki_url",""),
                "claude_ai_used":bool(cr and cr.get("used")),
                "claude_explanation":cr.get("explanation","") if cr else "",
                "hybrid_score":result.get("hybrid_score"),
                "pipeline_source":result.get("source","hybrid_pipeline"),
                "no_evidence":no_ev,"elapsed":elapsed,"cached":False,
                **({"warning":warn} if warn else {})}
        _cache.set(claim,resp); yield from send("result",resp)
    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

@app.route("/cache/clear", methods=["POST"])
def cache_clear():
    with _cache._lock: _cache._cache.clear()
    return jsonify({"status":"cleared"})

@app.route("/cache/stats", methods=["GET"])
def cache_stats():
    return jsonify({"size":len(_cache),"max":CACHE_MAX_SIZE,"ttl_seconds":CACHE_TTL,
                    "bert_loaded":BERT_LOADED,"csv_fallback":CSV_FALLBACK_LOADED,
                    "nli_backend":NLI_BACKEND})

@app.route("/retrain", methods=["POST"])
def retrain():
    global BERT_LOADED, bert_tokenizer, bert_model
    if not os.path.isfile(CSV_PATH):
        return jsonify({"error":f"merged_news.csv not found at {CSV_PATH}"}), 400
    try:
        trained = train_bert_from_csv(CSV_PATH, BERT_MODEL_DIR)
        if trained: return jsonify({"status":"retrained","bert_loaded":_try_load_bert(BERT_MODEL_DIR)})
        return jsonify({"status":"training_failed"}), 500
    except Exception as e:
        return jsonify({"error":str(e)}), 500

# =============================================================
# /evaluate — BATCH DATASET EVALUATION  — uses _read_csv_safe
# =============================================================
_eval_results = {}

def _bert_only_predict(claim):
    if _is_article_input(claim):
        bp    = bert_score_claim(claim); rp = bp["real_prob"]
        boost = _credible_source_boost(claim)
        if boost > 0: rp = min(rp+boost, 0.99)
        return "REAL" if rp >= OPTIMAL_THRESHOLD else "FAKE"
    topic   = detect_topic(claim); negated = is_negated(claim)
    nc      = normalise_claim(claim)
    kb = local_kb_lookup(claim)
    if kb in ("REAL","FAKE"): return kb
    if kb is None and negated:
        kbp = local_kb_lookup(rewrite_claim_positive(nc))
        if kbp in ("REAL","FAKE"):
            return "FAKE" if kbp=="REAL" else "REAL"
    if topic=="sports":
        sw = sports_winner_check(claim)
        if sw["verdict"] in ("REAL","FAKE"): return sw["verdict"]
    if topic=="science":
        cr = call_claude_ai_knowledge(claim)
        if cr.get("verdict") in ("REAL","FAKE"): return cr["verdict"]
    bp = bert_score_claim(claim); hs = bp["real_prob"]
    if hs >= OPTIMAL_THRESHOLD+0.05: return "REAL"
    if hs <= OPTIMAL_THRESHOLD-0.05: return "FAKE"
    try:
        cr = call_claude_ai_knowledge(claim); v = cr.get("verdict","UNCERTAIN")
        if v in ("REAL","FAKE"): return v
    except: pass
    return "REAL" if hs >= OPTIMAL_THRESHOLD else "FAKE"

@app.route("/evaluate", methods=["POST","OPTIONS"])
def evaluate():
    if request.method=="OPTIONS": return "", 200
    global _eval_results
    t0 = time.time()
    try:
        import pandas as pd
        body     = request.get_json(silent=True) or {}
        csv_path = body.get("csv_path", CSV_PATH)
        max_rows = int(body.get("max_rows", 200))
        req_tc   = body.get("text_col"); req_lc = body.get("label_col")
        if not os.path.isfile(csv_path):
            return jsonify({"error": f"CSV not found: {csv_path}"}), 400
        df = _read_csv_safe(csv_path)  # FIX: was pd.read_csv(csv_path, encoding="utf-8", ...)
        tc, tit, lc, sub = _detect_csv_columns(df)
        if req_tc and req_tc in df.columns: tc = req_tc
        if req_lc and req_lc in df.columns: lc = req_lc
        if not tc or tc=="__combined__":
            if tit:
                df["__text__"] = df[tit].fillna("")+" "+df.get("text",pd.Series(dtype=str)).fillna("")
                tc = "__text__"
            else: return jsonify({"error":"Could not detect text column"}), 400
        if not lc: return jsonify({"error":"Could not detect label column"}), 400
        flip = _detect_label_polarity(df, lc)
        df["__label__"] = _norm_labels(df[lc], flip=flip)
        df = df.dropna(subset=["__label__"]); df = df[df["__label__"].isin([0,1])]
        df[tc] = df[tc].fillna("").astype(str); df = df[df[tc].str.len()>5]
        n_each  = max_rows//2
        fake_df = df[df["__label__"]==0].sample(min(n_each,(df["__label__"]==0).sum()), random_state=42)
        real_df = df[df["__label__"]==1].sample(min(n_each,(df["__label__"]==1).sum()), random_state=42)
        eval_df = pd.concat([fake_df,real_df]).sample(frac=1,random_state=42).reset_index(drop=True)
        print(f"\n[EVALUATE] {len(eval_df)} rows (FAKE:{len(fake_df)} REAL:{len(real_df)})  flip={flip}")
        y_true, y_pred, errors = [], [], []
        for idx, row in eval_df.iterrows():
            claim    = str(row[tc])[:300]; true_lbl = int(row["__label__"])
            try:
                pred_str = _bert_only_predict(claim); pred_lbl = 1 if pred_str=="REAL" else 0
            except Exception as e:
                print(f"  [Eval] Row {idx} error: {e}"); pred_lbl = 1 if true_lbl==0 else 0
            y_true.append(true_lbl); y_pred.append(pred_lbl)
            if pred_lbl != true_lbl:
                errors.append({"claim":claim[:120],"true":"REAL" if true_lbl==1 else "FAKE",
                                "pred":"REAL" if pred_lbl==1 else "FAKE"})
            if (idx+1)%20==0:
                running_acc = sum(a==b for a,b in zip(y_true,y_pred))/len(y_true)
                print(f"  [Eval] {idx+1}/{len(eval_df)} running_acc={running_acc:.3f}")
        total = len(y_true); correct = sum(a==b for a,b in zip(y_true,y_pred))
        tp=sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==1)
        tn=sum(1 for t,p in zip(y_true,y_pred) if t==0 and p==0)
        fp=sum(1 for t,p in zip(y_true,y_pred) if t==0 and p==1)
        fn=sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==0)
        prec_r=tp/(tp+fp) if (tp+fp)>0 else 0; rec_r=tp/(tp+fn) if (tp+fn)>0 else 0
        f1_r=2*prec_r*rec_r/(prec_r+rec_r) if (prec_r+rec_r)>0 else 0
        prec_f=tn/(tn+fn) if (tn+fn)>0 else 0; rec_f=tn/(tn+fp) if (tn+fp)>0 else 0
        f1_f=2*prec_f*rec_f/(prec_f+rec_f) if (prec_f+rec_f)>0 else 0
        result = {"accuracy":round(correct/total,4),"macro_f1":round((f1_r+f1_f)/2,4),
                  "f1_real":round(f1_r,4),"f1_fake":round(f1_f,4),
                  "precision_real":round(prec_r,4),"recall_real":round(rec_r,4),
                  "precision_fake":round(prec_f,4),"recall_fake":round(rec_f,4),
                  "total":total,"correct":correct,"wrong":total-correct,
                  "confusion":{"TP":tp,"TN":tn,"FP":fp,"FN":fn},"errors":errors[:20],
                  "elapsed":round(time.time()-t0,2),"label_flip_applied":flip}
        _eval_results = result
        print(f"\n[EVALUATE DONE] acc={correct/total:.3f} macro_f1={(f1_r+f1_f)/2:.3f} flip={flip}")
        return jsonify(result)
    except Exception as e:
        traceback.print_exc(); return jsonify({"error":str(e)}), 500

@app.route("/evaluate/results", methods=["GET"])
def evaluate_results():
    if not _eval_results:
        return jsonify({"error":"No evaluation run yet. POST to /evaluate first."}), 404
    return jsonify(_eval_results)

@app.route("/evaluate/stream", methods=["POST","OPTIONS"])
def evaluate_stream():
    if request.method=="OPTIONS": return "", 200
    body     = request.get_json(silent=True) or {}
    csv_path = body.get("csv_path", CSV_PATH); max_rows = int(body.get("max_rows", 200))
    def generate():
        global _eval_results
        t0 = time.time()
        def send(ev, d):
            pl = json.dumps({**d,"elapsed":round(time.time()-t0,2)})
            yield f"event: {ev}\ndata: {pl}\n\n"
        try:
            import pandas as pd
            if not os.path.isfile(csv_path):
                yield from send("error",{"message":f"CSV not found: {csv_path}"}); return
            df = _read_csv_safe(csv_path)  # FIX: was pd.read_csv(csv_path, encoding="utf-8", ...)
            tc, tit, lc, sub = _detect_csv_columns(df)
            if not tc or tc=="__combined__":
                if tit:
                    df["__text__"] = df[tit].fillna("")+" "+df.get("text",pd.Series(dtype=str)).fillna("")
                    tc = "__text__"
                else: yield from send("error",{"message":"Could not detect text column"}); return
            if not lc: yield from send("error",{"message":"Could not detect label column"}); return
            flip = _detect_label_polarity(df, lc)
            df["__label__"] = _norm_labels(df[lc], flip=flip)
            df = df.dropna(subset=["__label__"]); df = df[df["__label__"].isin([0,1])]
            df[tc] = df[tc].fillna("").astype(str); df = df[df[tc].str.len()>5]
            n_each  = max_rows//2
            fake_df = df[df["__label__"]==0].sample(min(n_each,(df["__label__"]==0).sum()), random_state=42)
            real_df = df[df["__label__"]==1].sample(min(n_each,(df["__label__"]==1).sum()), random_state=42)
            eval_df = pd.concat([fake_df,real_df]).sample(frac=1,random_state=42).reset_index(drop=True)
            yield from send("progress",{"stage":f"Loaded {len(eval_df)} rows (flip={flip})","total":len(eval_df),"done":0})
            y_true, y_pred, errors = [], [], []
            for i, (_, row) in enumerate(eval_df.iterrows()):
                claim    = str(row[tc])[:300]; true_lbl = int(row["__label__"])
                try:
                    pred_str = _bert_only_predict(claim); pred_lbl = 1 if pred_str=="REAL" else 0
                except: pred_lbl = 1 if true_lbl==0 else 0
                y_true.append(true_lbl); y_pred.append(pred_lbl)
                if pred_lbl != true_lbl:
                    errors.append({"claim":claim[:100],"true":"REAL" if true_lbl==1 else "FAKE",
                                   "pred":"REAL" if pred_lbl==1 else "FAKE"})
                if (i+1)%10==0:
                    running_acc = sum(a==b for a,b in zip(y_true,y_pred))/len(y_true)
                    yield from send("progress",{"stage":f"Row {i+1}/{len(eval_df)}","done":i+1,
                                                "total":len(eval_df),"running_acc":round(running_acc,3)})
            total = len(y_true); correct = sum(a==b for a,b in zip(y_true,y_pred))
            tp=sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==1)
            tn=sum(1 for t,p in zip(y_true,y_pred) if t==0 and p==0)
            fp=sum(1 for t,p in zip(y_true,y_pred) if t==0 and p==1)
            fn=sum(1 for t,p in zip(y_true,y_pred) if t==1 and p==0)
            prec_r=tp/(tp+fp) if (tp+fp)>0 else 0; rec_r=tp/(tp+fn) if (tp+fn)>0 else 0
            f1_r=2*prec_r*rec_r/(prec_r+rec_r) if (prec_r+rec_r)>0 else 0
            prec_f=tn/(tn+fn) if (tn+fn)>0 else 0; rec_f=tn/(tn+fp) if (tn+fp)>0 else 0
            f1_f=2*prec_f*rec_f/(prec_f+rec_f) if (prec_f+rec_f)>0 else 0  # FIX: was rc_f (typo)
            result = {"accuracy":round(correct/total,4),"macro_f1":round((f1_r+f1_f)/2,4),
                      "f1_real":round(f1_r,4),"f1_fake":round(f1_f,4),
                      "precision_real":round(prec_r,4),"recall_real":round(rec_r,4),
                      "precision_fake":round(prec_f,4),"recall_fake":round(rec_f,4),
                      "total":total,"correct":correct,"wrong":total-correct,
                      "confusion":{"TP":tp,"TN":tn,"FP":fp,"FN":fn},"errors":errors[:20],
                      "label_flip_applied":flip}
            _eval_results = result; yield from send("result",result)
        except Exception as e:
            traceback.print_exc(); yield from send("error",{"message":str(e)})
    return Response(stream_with_context(generate()), mimetype="text/event-stream",
                    headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


if __name__ == "__main__":
    with _cache._lock: _cache._cache.clear()
    _live_cache.clear(); _live_cache_time.clear()
    print("[Startup] Caches cleared.")
    fok = FACTCHECK_API_KEY and "YOUR" not in FACTCHECK_API_KEY
    nok = NEWSAPI_KEY       and "YOUR" not in NEWSAPI_KEY
    cok = GEMINI_SDK and GEMINI_API_KEY and "YOUR" not in GEMINI_API_KEY
    ms  = ("✓ BERT LOADED" if BERT_LOADED else
           ("✓ CSV TF-IDF FALLBACK" if CSV_FALLBACK_LOADED else "✗ No model"))
    ni  = (f"✓ {NLI_BACKEND.upper()} NLI" if NLI_BACKEND=="pipeline" else "⚠ KEYWORD FALLBACK")
    print("\n" + "="*60)
    print("  TruthLens PRO v37 — all data in truthlens_data.json, no hardcoded tables")
    print(f"  URL              : http://127.0.0.1:5000")
    print(f"  Google FactCheck : {'✓ ENABLED' if fok else '✗ Add FACTCHECK_API_KEY'}")
    print(f"  NewsAPI          : {'✓ ENABLED' if nok else '✗ Add NEWSAPI_KEY'}")
    print(f"  Gemini AI        : {'✓ ENABLED' if cok else '✗ Add GEMINI_API_KEY'}")
    print(f"  ML Model         : {ms}")
    print(f"  NLI Backend      : {ni}")
    print(f"  Label Encoding   : LABEL_0_MEANS={LABEL_0_MEANS}")
    print(f"  Data Folder      : {DATA_DIR}")
    print(f"  Celebrities      : {len(CELEB_DATA)} aliases | Living people: {len(KNOWN_LIVING_PEOPLE)}")
    print(f"  Sports Winners   : {len(SPORTS_WINNERS)} tournament-year entries")
    print(f"  Known Facts      : {len(KNOWN_FACTS)} science/event facts")
    print(f"  Local KB         : {'✓ '+str(len(_KB))+' facts' if _KB else '✗ knowledge_base.json not found'}")
    print(f"  Calibrated Thr.  : OPTIMAL_THRESHOLD={OPTIMAL_THRESHOLD:.2f}")
    print(f"  Label Prior      : P(REAL)={LABEL_PRIOR_REAL:.3f}")
    print(f"  CSV Path         : {CSV_PATH}")
    print(f"  BERT Save Dir    : {BERT_MODEL_DIR}")
    print("="*60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)