"""
CreatorPulse — NLP preprocessing for comment classification.

Adds three columns next to the untouched original `text`:

    text_norm  normalised text (one string, readable)
    tokens     space-separated tokens of text_norm
    lemmas     space-separated lemmas (English words only; see below)

Pipeline
--------
1. Normalisation
   - Unicode NFKC, curly quotes -> straight, lowercase (no effect on Devanagari).
   - URLs -> <url>, @mentions -> <user>, video timestamps (12:03) -> <time>.
   - Emoji kept (they carry intent: ❤ / 🙏 / 😭); variation selectors and skin
     tones dropped so ❤️ == ❤.
   - Elongations squeezed to two characters: "sooooo" -> "soo", "plzzz" -> "plzz"
     (digits are never squeezed). Upper-case "IT" -> info_tech.
   - English contractions expanded so negation survives as its own token:
     "didn't" -> "did not", "can't" -> "can not", "dont" -> "do not".
   - Common spelling variants unified (SPELLING_VARIANTS): pls/plz -> please,
     nhi/nahin -> nahi, kese -> kaise, vedio -> video, u -> you ...

2. Tokenisation
   Words are runs of letters, digits and combining marks, so Devanagari
   (बढ़िया) and other scripts stay whole. "?" and "!" are kept as tokens
   because they signal questions and requests. Each emoji is a token. Other
   punctuation (including the danda ।) is dropped.

3. Lemmatisation (optional, on by default)
   NLTK WordNet (verb and noun base forms) applied only to
   Latin-script tokens longer than two letters that are NOT protected.
   Protected = negations, question words, request words, and a Hinglish
   lexicon; without this WordNet turns Hinglish "us" (that) into "u". Words
   that are already WordNet base words stay as written, the longest candidate
   lemma wins, and intent-bearing adjectives ("confused", "amazing") are kept
   (KEEP_AS_IS). See _lemma for details. Devanagari and unknown words pass through
   unchanged. WordNet data is downloaded on first run if missing.

4. Stopword removal (optional, off by default)
   A small English stopword list; negations, question words, "please" and all
   Hinglish words are never removed.

Usage:
    python -m src.preprocess
    python -m src.preprocess --no-lemmatize
    python -m src.preprocess --remove-stopwords
"""

import argparse
import re
import unicodedata
from pathlib import Path
from typing import List

import pandas as pd

from src.config import PROCESSED_DATA_DIR

INPUT = PROCESSED_DATA_DIR / "training_comments.csv"
OUTPUT = PROCESSED_DATA_DIR / "preprocessed_comments.csv"
CSV_ENCODING = "utf-8-sig"

# --------------------------------------------------------------------------
# Word lists
# --------------------------------------------------------------------------

NEGATIONS = {
    # English
    "not", "no", "never", "nor", "none", "nothing", "nobody", "nowhere",
    "neither", "without", "cannot",
    # Hinglish / Hindi (romanised and Devanagari)
    "nahi", "na", "mat", "bina", "nahi", "नहीं", "ना", "मत", "बिना",
}

QUESTION_WORDS = {
    "what", "why", "how", "when", "where", "which", "who", "whom", "whose",
    "can", "could", "should", "would", "is", "are", "do", "does", "did", "will",
    "kya", "kyun", "kaise", "kab", "kahan", "kaun", "kon", "kitna", "kitne",
    "kitni", "kaisa", "kaisi", "konsa", "kaunsa",
    "क्या", "क्यों", "कैसे", "कब", "कहाँ", "कौन", "कितना",
}

REQUEST_WORDS = {
    "please", "kindly", "request", "make", "upload", "share", "bana", "banao",
    "banaiye", "lao", "laao", "laiye", "dedo", "de", "do", "dijiye", "karo",
    "kariye", "sikhao", "chahiye",
}

# Romanised Hindi/Urdu words common in Hinglish comments. Several look like
# English words (me, to, wo, ye, hi, do, us) and must not be treated as such.
# "the" (= were) is deliberately left out: in this data it is almost always
# the English article.
HINGLISH = {
    "hai", "he", "hain", "h", "ho", "hu", "hun", "hoon", "tha", "thi", "hoga", "nai", "ni",
    "hogi", "honge", "raha", "rahi", "rahe", "rha", "rhi", "rhe",
    "ka", "ki", "ke", "ko", "se", "me", "mein", "mai", "main", "par", "pe",
    "tak", "liye", "lie", "wala", "wali", "wale", "bhi", "aur", "ya", "to",
    "toh", "hi", "sab", "kuch", "koi", "ek", "do", "teen",
    "ye", "yeh", "wo", "woh", "vo", "us", "unka", "iska", "uska", "is", "isme",
    "aap", "ap", "apka", "aapka", "apki", "aapki", "apke", "aapke", "hum",
    "hamein", "humein", "mujhe", "mera", "meri", "mere", "tum", "tera",
    "bhai", "bhaiya", "bhaijaan", "didi", "di", "dada", "sir", "mam", "maam",
    "ji", "yaar", "bro", "guru",
    "kar", "karo", "karna", "karne", "karke", "kiya", "kr", "krna", "krke",
    "de", "dedo", "dena", "diya", "le", "lo", "lena", "liya", "aa", "aao",
    "aaya", "aayi", "aaye", "ja", "jao", "jana", "gaya", "gayi", "gaye",
    "bana", "banao", "banana", "banaya", "lao", "laao", "laiye", "dijiye",
    "samajh", "samjha", "samjhao", "padh", "padhai", "padhna", "sikh",
    "sikhna", "sikho", "sikhao", "dekh", "dekho", "dekha", "chahiye",
    "chahta", "chahti", "sakta", "sakti", "sakte", "milega", "milegi",
    "accha", "acha", "achha", "bahut", "bohot", "bhot", "badhiya", "badiya",
    "zabardast", "mast", "shukriya", "dhanyavad", "dhanyawad",
    "abhi", "ab", "phir", "fir", "jab", "tab", "agar", "lekin", "kyunki",
    "matlab", "sahi", "galat", "naya", "nayi", "pura", "puri", "jaldi",
    "kal", "aaj", "din", "saal", "log", "baat", "kaam", "paisa", "naukri",
    "nahi", "na", "mat", "bina",
    "bas", "kis", "jis", "bade", "bada", "badi", "paise", "wahi", "yahi", "unke", "inke",
    "saath", "sath", "wahan", "yahan", "chal", "chalo", "bhaiyon", "dosto",
    "kya", "kyun", "kaise", "kab", "kahan", "kaun", "kon", "kitna", "kitne",
}

# Variant -> canonical form, applied to whole tokens after elongation squeeze.
# Checked against the corpus. Deliberately NOT mapped because they are
# ambiguous in Hinglish: "nai" (= nahi, but also = ne: "aap nai ..."),
# "r" (variable names, J.A.R.V.I.S.), "hn" (hain / haan), "ty", "sm".
SPELLING_VARIANTS = {
    # please
    "pls": "please", "plz": "please", "plzz": "please", "plss": "please",
    "pleas": "please", "plese": "please", "pleasee": "please", "plx": "please",
    "pllz": "please", "pleez": "please",
    # thanks
    "thx": "thanks", "thnx": "thanks", "thanx": "thanks",
    "thanku": "thanks", "thankuu": "thanks", "thanq": "thanks", "thankyou": "thanks",
    # English chat shorthand
    "u": "you", "ur": "your", "pz": "please", "vid": "video",
    "vids": "videos", "vedio": "video", "vidio": "video", "vdo": "video",
    "vedios": "videos", "bcz": "because", "bcoz": "because", "coz": "because",
    "cuz": "because", "becoz": "because", "plzz": "please",
    "idk": "i do not know", "dunno": "do not know",
    # Hinglish
    "nhi": "nahi", "nahin": "nahi", "nahii": "nahi", "ni": "nahi",
    "kese": "kaise", "kaese": "kaise", "kayse": "kaise", "kyu": "kyun",
    "kyon": "kyun", "kiu": "kyun", "kiun": "kyun", "kyo": "kyun", "kia": "kya", "kyaa": "kya",
    "smjh": "samajh", "samaj": "samajh", "smj": "samajh",
    "bhaiyya": "bhaiya", "bhaiyaa": "bhaiya", "bhaiyya": "bhaiya", "bhaiyaji": "bhaiya",
    "bhot": "bahut", "bohot": "bahut", "bahot": "bahut", "boht": "bahut",
    "acha": "accha", "achha": "accha", "achaa": "accha",
    "krna": "karna", "krke": "karke", "kr": "kar", "rha": "raha", "rhi": "rahi",
    "rhe": "rahe", "krte": "karte", "krenge": "karenge",
    "bhut": "bahut", "bhaut": "bahut", "chaiye": "chahiye", "chahie": "chahiye",
    "lakin": "lekin", "phele": "pehle", "pahle": "pehle",
    "mam": "maam", "ma'am": "maam", "madam": "maam",
}

CONTRACTIONS = [
    (r"\bcan'?t\b", "can not"),
    (r"\bwon'?t\b", "will not"),
    (r"\bshan'?t\b", "shall not"),
    (r"\bain'?t\b", "is not"),
    (r"\b(do|does|did|is|are|was|were|has|have|had|could|should|would|must|need|might)n'?t\b",
     r"\1 not"),
    (r"\b(\w+)n't\b", r"\1 not"),
    (r"\bi'?m\b", "i am"),
    (r"\b(\w+)'re\b", r"\1 are"),
    (r"\b(\w+)'ve\b", r"\1 have"),
    (r"\b(\w+)'ll\b", r"\1 will"),
    (r"\b(\w+)'d\b", r"\1 would"),
    (r"\b(it|that|what|there|here|who|how|where|he|she|this)'s\b", r"\1 is"),
    (r"\blet's\b", "let us"),
    (r"\b(\w+)'s\b", r"\1"),  # possessive: "harry's" -> "harry"
]

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "of", "at", "by",
    "for", "with", "about", "into", "through", "during", "before", "after",
    "above", "below", "from", "up", "down", "in", "out", "on", "off", "over",
    "under", "again", "further", "once", "here", "there", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "only", "own",
    "same", "than", "too", "very", "just", "i", "my", "myself", "we", "our",
    "you", "your", "he", "him", "his", "she", "her", "it", "its", "they",
    "them", "their", "this", "that", "these", "those", "am", "be", "been",
    "being", "have", "has", "had", "having", "was", "were", "also",
}

# Never lemmatised; never removed as stopwords.
PROTECTED = NEGATIONS | QUESTION_WORDS | REQUEST_WORDS | HINGLISH | {
    "please", "thanks", "you", "your",
}
STOPWORDS = STOPWORDS - PROTECTED

# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------

_URL = re.compile(r"(https?://|www\.)\S+")
_MENTION = re.compile(r"(?<!\w)@[\w.\-]+")
_TIMESTAMP = re.compile(r"(?<![\d:])\d{1,2}:\d{2}(?::\d{2})?(?![\d:])")
# Squeeze runs of 3+ identical letters, symbols or emoji, but never digits
# ("10000000" and "111" are meaningful in maths comments).
_REPEAT = re.compile(r"([^\d\s])\1{2,}")
# Upper-case "IT" (information technology) would otherwise become the pronoun
# "it" after lower-casing. Only applied when the comment is not shouted.
_IT_ACRONYM = re.compile(r"\bIT\b")
_SPACE = re.compile(r"\s+")
_CONTRACTIONS = [(re.compile(p), r) for p, r in CONTRACTIONS]
_DROP_CHARS = {"︎", "️", "‍"} | {chr(c) for c in range(0x1F3FB, 0x1F400)}
_PLACEHOLDERS = {"<url>", "<user>", "<time>"}


def normalize(text: str) -> str:
    """Return the normalised form of one comment (original is not changed)."""
    t = unicodedata.normalize("NFKC", text or "")
    t = t.replace("’", "'").replace("‘", "'").replace("`", "'")
    t = "".join(ch for ch in t if ch not in _DROP_CHARS)
    t = _URL.sub(" <url> ", t)
    t = _MENTION.sub(" <user> ", t)
    t = _TIMESTAMP.sub(" <time> ", t)
    letters = [c for c in t if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) < 0.5:
        t = _IT_ACRONYM.sub("info_tech", t)
    t = t.lower()
    t = _REPEAT.sub(r"\1\1", t)
    for pat, rep in _CONTRACTIONS:
        t = pat.sub(rep, t)
    words = [SPELLING_VARIANTS.get(w, w) for w in t.split()]
    return _SPACE.sub(" ", " ".join(words)).strip()


# --------------------------------------------------------------------------
# Tokenisation
# --------------------------------------------------------------------------

def _is_word_char(ch: str) -> bool:
    cat = unicodedata.category(ch)
    return cat[0] in ("L", "N", "M") or ch == "_"


def _is_emoji(ch: str) -> bool:
    return unicodedata.category(ch) == "So" or 0x1F000 <= ord(ch) <= 0x1FAFF


def tokenize(norm: str) -> List[str]:
    """Split normalised text into word, emoji, '?' and '!' tokens."""
    tokens: List[str] = []
    i, n = 0, len(norm)
    while i < n:
        ch = norm[i]
        if ch == "<":
            end = norm.find(">", i)
            if end != -1 and norm[i:end + 1] in _PLACEHOLDERS:
                tokens.append(norm[i:end + 1])
                i = end + 1
                continue
        if _is_word_char(ch):
            j = i + 1
            while j < n and (_is_word_char(norm[j]) or
                             (norm[j] == "'" and j + 1 < n and _is_word_char(norm[j + 1]))):
                j += 1
            word = norm[i:j]
            tokens.extend(SPELLING_VARIANTS.get(word, word).split())
            i = j
            continue
        if ch in "?!" or _is_emoji(ch):
            tokens.append(ch)
        i += 1
    # A run like "????" or "!!!" becomes at most two tokens.
    squeezed: List[str] = []
    for tok in tokens:
        if tok in "?!" and len(squeezed) >= 2 and squeezed[-1] == squeezed[-2] == tok:
            continue
        squeezed.append(tok)
    return squeezed


# --------------------------------------------------------------------------
# Lemmatisation and stopwords
# --------------------------------------------------------------------------

def _is_latin_word(tok: str) -> bool:
    return tok.isascii() and tok.isalpha()


# Words kept exactly as written. Emotional -ing/-ed adjectives carry the
# label signal ("confused" -> Question, "amazing" -> Praise), and some domain
# words are mangled by WordNet's suffix rules (pandas -> panda).
KEEP_AS_IS = {
    "confused", "confusing", "stuck", "amazing", "interesting", "boring",
    "exciting", "excited", "inspiring", "inspired", "outstanding", "detailed",
    "advanced", "interested", "tired", "worried", "missing", "pending",
    "fascinating", "disappointing", "disappointed", "surprising", "surprised",
    "embarrassing", "satisfying", "promising", "refreshing", "encouraging",
    "misleading", "thrilled", "astonished", "terrified", "exhausted", "exhausting",
    "frustrated", "frustrating", "scared", "blessed",
    "thanks", "pandas", "numpy", "media", "data", "lets", "mains",
}

_wordnet = None


def _get_wordnet():
    """Load NLTK's WordNet corpus reader, downloading the data on first use."""
    global _wordnet
    if _wordnet is None:
        import nltk

        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            if not nltk.download("wordnet", quiet=True):
                raise RuntimeError(
                    "WordNet data is missing. Run:  python -m nltk.downloader wordnet omw-1.4"
                )
        from nltk.corpus import wordnet

        wordnet.ensure_loaded()
        _wordnet = wordnet
    return _wordnet


def _candidates(wn, tok: str, pos: str) -> List[str]:
    """All WordNet base forms of tok for one part of speech."""
    try:
        forms = wn._morphy(tok, pos, check_exceptions=True)  # every candidate
    except Exception:
        first = wn.morphy(tok, pos)
        forms = [first] if first else []
    return [f for f in forms if f != tok]


def _lemma(tok: str) -> str:
    """WordNet lemma of an English token, from its verb and noun base forms.

    - A word that is already a WordNet base word stays as written, so real
      words are not reinterpreted: "bit" (not "bite"), "physics", "pass",
      "boss", "feed", "learning" (the noun).
    - With several candidates (verb and noun) the longest wins, verb on a tie:
      "hoping" -> "hope" not "hop", "coded" -> "code", "nodes" -> "node",
      "dies" -> "die", "was" -> "be".
    - Three-letter words only take irregular lemmas ("was" -> "be",
      "got" -> "get"), never a stripped "s" ("css", "tcs", Hinglish "kis").
    - Adjectives are not lemmatised ("latest" stays "latest").
    """
    if tok in KEEP_AS_IS:
        return tok
    wn = _get_wordnet()
    if any(lem.name().lower() == tok for syn in wn.synsets(tok) for lem in syn.lemmas()):
        return tok  # already a base word
    # (length, is_verb): longest candidate wins, verb preferred on a tie.
    forms = [(f, 1) for f in _candidates(wn, tok, "v")] + [(f, 0) for f in _candidates(wn, tok, "n")]
    if len(tok) <= 3:
        forms = [(f, v) for f, v in forms if not tok.startswith(f)]
    if not forms:
        return tok
    return max(forms, key=lambda fv: (len(fv[0]), fv[1]))[0]


def lemmatize(tokens: List[str]) -> List[str]:
    """Lemmatise English tokens; protected, non-Latin and short tokens pass through."""
    return [
        tok if (tok in PROTECTED or not _is_latin_word(tok) or len(tok) <= 2) else _lemma(tok)
        for tok in tokens
    ]


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [t for t in tokens if t not in STOPWORDS]


def preprocess(text: str, do_lemmatize: bool = True, drop_stopwords: bool = False) -> dict:
    norm = normalize(text)
    tokens = tokenize(norm)
    lemmas = lemmatize(tokens) if do_lemmatize else []
    if drop_stopwords:
        tokens = remove_stopwords(tokens)
        lemmas = remove_stopwords(lemmas)
    return {"text_norm": norm, "tokens": " ".join(tokens), "lemmas": " ".join(lemmas)}


def preprocess_frame(df: pd.DataFrame, do_lemmatize: bool = True,
                     drop_stopwords: bool = False) -> pd.DataFrame:
    """Return a copy of df with text_norm, tokens and lemmas added. `text` is unchanged."""
    out = df.copy()
    rows = [preprocess(t, do_lemmatize, drop_stopwords) for t in out["text"]]
    for col in ("text_norm", "tokens", "lemmas"):
        out[col] = [r[col] for r in rows]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess comments for NLP classification.")
    parser.add_argument("--input", default=str(INPUT))
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--no-lemmatize", action="store_true", help="Skip lemmatisation.")
    parser.add_argument("--force", action="store_true", help="Overwrite the output file if it exists.")
    parser.add_argument("--remove-stopwords", action="store_true",
                        help="Drop English stopwords (negations, question words and Hinglish kept).")
    args = parser.parse_args()

    if Path(args.output).exists() and not args.force:
        raise SystemExit(f"[ERROR] {args.output} already exists. Use --output NEW_PATH or --force.")
    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    out = preprocess_frame(df, not args.no_lemmatize, args.remove_stopwords)
    out.to_csv(args.output, index=False, encoding=CSV_ENCODING)

    n_empty = (out["tokens"] == "").sum()
    print(f"Preprocessed {len(out)} comments -> {args.output}")
    print(f"  lemmatize={not args.no_lemmatize}  remove_stopwords={args.remove_stopwords}")
    print(f"  comments with no tokens: {n_empty}")


if __name__ == "__main__":
    main()
