"""
CreatorPulse — Stage 2: near-duplicate detection and topic discovery.

Groups semantically similar audience comments (by default the actionable
intents Question, Request and Confusion/Problem) into topics.

Steps
-----
1. Select comments from data/processed/comments_predicted.csv by predicted
   category. Spam is always excluded; Praise/Other only with --include.
2. Near-duplicates. Character 3-5-gram TF-IDF on the normalised text; pairs
   with cosine >= 0.85 are linked and linked comments form one duplicate group
   (union-find). Clustering, topic counts and naming use ONE comment per group
   (the earliest), so copy-pasted or flooded comments are not double-counted.
   The group size is kept so raw frequency is still reported.
3. Topic features. Intent and filler words ("please", "make", "video", "sir",
   "bhai", "kya", "how" ...) are removed first — otherwise every request would
   cluster with every other request because they all say "please make a video".
   The remaining topic words are represented two ways and combined:
     * word 1-2 gram TF-IDF  (weight 0.5) — phrases like "dynamic programming"
     * char 3-5 gram TF-IDF  (weight 0.5) — tolerant of spelling variants and
       shorthand (pyhton/python, bnao/banao, devops/dev ops)
   Each row is L2-normalised so cosine similarity is a dot product.
4. Clustering. Agglomerative clustering, average linkage, cosine distance,
   cut at similarity 0.30 (distance 0.70). No fixed number of topics, no
   randomness. Average linkage needs a comment to be similar to the group as a
   whole, not just to one member, which prevents long "chains".
5. Quality gates (comments that fail go to topic -1 "Unclustered"):
     * a topic needs >= 2 unique comments (after duplicate collapsing);
     * each member's similarity to its topic centroid must be >= 0.20;
     * keyword-only check: if the members only share ONE content word
       (similarity drops to ~0 without it) and they mix requests with
       questions/problems, the group is split by intent family — "please make a
       Python course" and "Python indentation error" share a word, not a need.
6. Naming. Class-based TF-IDF over topics picks each topic's most distinctive
   terms (a bigram when one is strong enough). The representative comment is
   the medoid: the unique comment most similar to the rest of its topic.

Multilingual sentence embeddings (sentence-transformers) are supported with
--features embeddings when the package and model are installed; see
src/topic_features_eval.py for the comparison that chose the default.

Outputs
    data/processed/topic_clusters.csv  one row per selected comment (duplicates included)
    data/processed/topic_summary.csv   one row per topic, plus the Unclustered row

Usage:
    python -m src.topics
    python -m src.topics --include Praise
    python -m src.topics --threshold 0.35 --features word+char
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DATA_DIR  # noqa: E402
from src.preprocess import HINGLISH, normalize, tokenize  # noqa: E402

CSV_ENCODING = "utf-8-sig"
IN_PREDICTED = PROCESSED_DATA_DIR / "comments_predicted.csv"
OUT_CLUSTERS = PROCESSED_DATA_DIR / "topic_clusters.csv"
OUT_SUMMARY = PROCESSED_DATA_DIR / "topic_summary.csv"

DEFAULT_CATEGORIES = ["Question", "Request", "Confusion/Problem"]
OPTIONAL_CATEGORIES = ["Praise", "Other"]
NEAR_DUP_THRESHOLD = 0.85
CLUSTER_THRESHOLD = 0.30      # min average cosine similarity inside a topic
MIN_MEMBER_SIM = 0.20         # min similarity of a member to its topic centroid
MIN_TOPIC_SIZE = 2            # unique comments
WORD_WEIGHT, CHAR_WEIGHT = 0.5, 0.5   # chosen with src.topic_features_eval
UNCLUSTERED_ID = -1
UNCLUSTERED_NAME = "Unclustered"
INTENT_FAMILY = {"Request": "request", "Question": "ask", "Confusion/Problem": "ask",
                 "Praise": "other", "Other": "other"}

# Words that signal intent, politeness or format rather than WHAT the comment is
# about. Removed only for topic features (the classifier still sees them).
INTENT_WORDS = {
    "please", "kindly", "request", "make", "made", "making", "create", "upload", "share", "start",
    "launch", "release", "bring", "provide", "give", "post", "cover", "explain", "teach", "tell",
    "guide", "help", "want", "wanted", "need", "needed", "like", "love", "wish", "hope", "waiting",
    "video", "videos", "vid", "course", "courses", "series", "playlist", "tutorial", "tutorials",
    "lecture", "lectures", "class", "classes", "session", "content", "part", "full", "complete",
    "detailed", "basic", "basics", "advanced", "advance", "beginner", "beginners", "next", "new",
    "also", "more", "one", "ek", "thank", "thanks", "thankyou", "great", "good", "nice", "best",
    "amazing", "awesome", "sir", "mam", "maam", "ma'am", "madam", "bhai", "bhaiya", "bro", "didi",
    "di", "dada", "ji", "guys", "team", "everyone", "anyone", "someone", "harry", "brother", "sister",
    "how", "what", "why", "when", "where", "which", "who", "can", "could", "would", "should", "will",
    "shall", "may", "might", "must", "do", "does", "did", "done", "doing", "is", "are", "was", "were",
    "be", "been", "am", "get", "got", "getting", "go", "going", "know", "think", "possible", "really",
    "bana", "banao", "banaiye", "bnao", "bnado", "banado", "bana do", "lao", "laao", "laiye",
    "leke", "lekar", "aao", "dalo", "daalo", "dijiye", "dedo", "do", "karo", "kro", "kariye", "krdo",
    "kardo", "sikhao", "sikha", "batao", "btao", "btaiye", "bataiye", "chahiye", "chahta", "chahti",
    "kya", "kaise", "kese", "kyun", "kab", "kahan", "kaun", "kon", "konsa", "kaunsa", "please",
    "yeh", "ye", "wo", "vo", "plz", "pls", "u", "ur", "you", "your", "i", "im", "me", "my", "we", "us",
    "our", "aap", "ap", "apki", "aapki", "apka", "aapka", "apke", "aapke", "hum", "hame", "hamien",
    "mujhe", "mai", "main", "mera", "meri", "mere", "hu", "hoon", "hun", "ho", "hai", "h", "hain",
    "he", "tha", "thi", "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "at",
    "by", "from", "about", "this", "that", "these", "those", "it", "its", "so", "very", "much",
    "some", "any", "all", "just", "only", "now", "then", "there", "here", "if", "but", "not", "no",
    "<url>", "<user>", "<time>", "?", "!", "ka", "ki", "ke", "ko", "se", "par", "pe", "bhi", "aur",
    "toh", "to", "hi", "sab", "kuch", "koi", "liye", "wala", "wali", "wale", "abhi", "ab", "jo",
    "kar", "kr", "krna", "karna", "karne", "dena", "lena", "de", "le", "sakte", "sakta", "sakti",
    "vote", "votes", "voting", "voted", "petition",
    "milega", "milegi", "hoga", "hogi", "raha", "rahi", "rahe", "rha", "rhi", "rhe", "gaya", "gayi",
}
# Hinglish request / filler verbs and their common spellings.
HINGLISH_FILLER = {
    "lai", "laye", "laiye", "layiye", "laayei", "laayiye", "aoo", "aaoo", "leke", "lekar", "laake",
    "bare", "baare", "barein", "sikhna", "shikhna", "seekhna", "sikhni", "sikhana", "sikhaye", "sikhaiye",
    "dikhao", "dikhaiye", "samjhaiye", "samjhao", "padhaiye", "padhao", "karwa", "karwao", "karvao",
    "krwa", "krwao", "dedo", "dijie", "kijiye", "kijie", "kare", "karein", "kariye", "krein", "kre",
    "chahte", "chahiye", "chaiye", "chahie", "hona", "hoti", "hota", "hote", "wali", "vala", "wale",
    "na", "haa", "haan", "han", "yaar", "yar", "pls", "plz", "plzz", "plss", "sirr", "mamm",
    "highly", "needed", "learn", "watch", "watching", "watched", "upcoming", "soon", "asap", "today",
    "yes", "ok", "okay", "hello", "hi", "hey", "dear", "padha", "padhna", "padhaya", "padhte", "padho", "congratulations", "congrats", "congratulation", "dekhna", "dekhni", "dekhne", "dekho", "dekh",
}
# The project's own small English stopword list (src.preprocess.STOPWORDS is the
# curated, negation-safe one) plus pronoun/aux words. scikit-learn's list is NOT
# used: it drops topic words such as "system", "interest" and "part".
from src.preprocess import STOPWORDS as PROJECT_STOPWORDS  # noqa: E402
TOPIC_STOPWORDS = set(INTENT_WORDS) | set(PROJECT_STOPWORDS) | set(HINGLISH) | HINGLISH_FILLER
# Subject words that also appear in the Hinglish list but carry topic meaning.
TOPIC_STOPWORDS -= {"padhai", "naukri", "paisa", "paise", "kaam", "galat", "samajh", "samjha"}

ACRONYMS = {"ai", "ml", "dsa", "it", "api", "mcp", "gpu", "pw", "jee", "neet", "llm", "sql", "css",
            "html", "js", "ui", "ux", "cse", "bca", "mca", "btech", "msc", "bsc", "iit", "nit", "dba",
            "aws", "gcp", "seo", "n8n", "cpu", "ram", "pdf", "ssc", "upsc", "gate", "imo", "jp", "ece",
            "cs", "agi", "rag", "nlp", "dbms", "os", "oops", "ios", "vlsi", "awt", "gui", "bsc", "ba", "aiml"}

_EMOJI_OR_PUNCT = re.compile(r"^[^\w]+$", re.UNICODE)


# --------------------------------------------------------------------------
# Text for topic features
# --------------------------------------------------------------------------

# Words that say HOW someone feels about the content (didn't understand, explain,
# doubt) rather than WHAT it is about. They may glue comments together, but they
# never name a topic, and a group sharing only these words is not a topic.
GENERIC_TERMS = {
    "understand", "understood", "understanding", "samajh", "samjha", "samjh", "smj", "smjh", "aata", "aaya",
    "aya", "ata", "nahi", "nhi", "nai", "explanation", "explained", "explaining", "doubt", "doubts", "question",
    "questions", "query", "problem", "problems", "issue", "issues", "confused", "confusion", "confusing",
    "clear", "clarify", "term", "terms", "thing", "things", "way", "don", "didn", "didnt", "dont", "doesn",
    "cause", "many", "topic", "topics", "work", "working", "works", "time", "point", "example", "concept",
    "right", "wrong", "correct", "answer", "please", "better", "well", "lot", "able", "use", "using",
    "anything", "something", "everything", "people", "student", "students", "bhai", "sir", "mam", "pata",
    "important", "necessary", "difference", "different", "best", "good", "possible", "required", "useful",
    "worth", "enough", "late", "now", "today", "teaching", "taught", "starting", "started", "starts", "start",
    "doing", "watch", "watching", "everyone", "guys", "anyone", "someone", "still", "complete",
}

# Maths written many ways in comments: k^2, k², k square, ksquare, k ki power 2, K2, k+k.
_MATH = [
    (re.compile(r"\b([a-z])\s*(?:\^|\*\*)\s*\(?2\)?(?![\d.])", re.I), r" \1sqr "),
    (re.compile(r"\b([a-z])\s*²", re.I), r" \1sqr "),
    (re.compile(r"\b([a-z])\s*(?:square|squared|sqr|sq)\b", re.I), r" \1sqr "),
    (re.compile(r"\b([a-z])\s+(?:ki\s+|to\s+the\s+)?(?:power|paower|pow)\s*(?:of\s+)?2\b", re.I), r" \1sqr "),
    (re.compile(r"\b([a-z])2\b", re.I), r" \1sqr "),
    (re.compile(r"\b([a-z])\s*\+\s*\1\b", re.I), r" 2\1 "),
]
_ELONGATED = re.compile(r"([a-z])(\1{2,})\b", re.I)   # a word stretched at its end: "stringssss", "pleaseeee"
_VOWELS = set("aeiouAEIOU")


def _unstretch(m) -> str:
    """Undo a stretched word ending: stringssss -> strings, pleaseeee -> please, linkkkk -> link,
    but dressss -> dress, alll -> all (s after a vowel, and l/f/z, are usually doubled in real words)."""
    ch = m.group(1)
    before = m.string[m.start() - 1] if m.start() > 0 else ""
    if ch.lower() in "lfz" or (ch.lower() == "s" and before in _VOWELS):
        return ch * 2
    return ch
_SQR = re.compile(r"^([a-z])sqr$")


# Different words for the same audience need. Topic features only: each word is
# replaced by one concept token, so "is there a text book?" and "any resources?"
# land in the same topic. The label is what the topic is called.
CONCEPTS = {
    "cmaterial": ("Study materials",
                  {"book", "books", "textbook", "textbooks", "ebook", "notes", "note", "resource", "resources",
                   "material", "materials", "pdf", "pdfs", "slide", "slides", "reference", "references",
                   "handout", "handouts", "reading"}),
    "crelevance": ("Is the course still relevant?",
                   {"relevant", "relevance", "irrelevant", "outdated", "obsolete", "uptodate", "updated", "update",
                    "old"}),
    "cpeer": ("Study partners and classmates",
               {"classmate", "classmates", "partner", "partners", "buddy", "buddies", "studygroup", "discord"}),
    "cnoteapp": ("Note-taking app", {"noteapp"}),
    "ccheckin": ("Viewer check-ins (who's watching when)", set()),
}
CONCEPT_OF = {w: c for c, (_, words) in CONCEPTS.items() for w in words}
CHECKIN_TOKEN = "ccheckin"
_CHECKIN = re.compile(
    r"(\b(who|whos|who's|anyone|anybody|any one|any1|kon|kaun)\b.{0,30}\b(start\w*|watch\w*|here|learn\w*|doing|"
    r"taking|join\w*|in)\b.{0,30}\b(19|20)\d\d\b)"
    r"|(^\W*(start(ed|ing)?|watching|day\s*\d+|attendance|present|here)\b.{0,40}"
    r"(\b(19|20)\d\d\b|\d{1,2}[/.-]\d{1,2}([/.-]\d{2,4})?|today|now|jan\w*|feb\w*|mar\w*|apr\w*|may|jun\w*|"
    r"jul\w*|aug\w*|sep\w*|oct\w*|nov\w*|dec\w*))", re.I)


def is_checkin(text: str) -> bool:
    """A roll-call comment ("Who is starting in 2026?", "Started 17/4/2026"): social, not a content need.
    Long comments that also ask something real are not treated as check-ins."""
    t = str(text).strip()
    return bool(_CHECKIN.search(t)) and len(t.split()) <= 10


def _canon_text(text: str) -> str:
    t = _ELONGATED.sub(_unstretch, str(text))
    t = re.sub(r"\bnotes?[\s-]*(?:taking[\s-]+)?apps?\b", " noteapp ", t, flags=re.I)   # a note-taking app
    t = re.sub(r"\bai\s*(?:/|&|and|-|\+)?\s*ml\b", " aiml ", t, flags=re.I)                 # AI/ML, ai and ml, aiml
    for pat, rep in _MATH:
        t = pat.sub(rep, t)
    return t


def _stem(t: str) -> str:
    """Plural to singular for topic matching only (strings -> string, notes -> note)."""
    if (len(t) > 4 and t.endswith("s") and not t.endswith(("ss", "us", "is", "ies", "os"))
            and t not in ACRONYMS and t.isalpha()):
        return t[:-1]
    return t


def topic_tokens(text: str) -> List[str]:
    """Content tokens only: normalised, tokenised, intent/filler words and emoji removed.
    Check-in comments become a single token; words for the same need become one concept token."""
    if is_checkin(text):
        return [CHECKIN_TOKEN]
    toks = tokenize(normalize(_canon_text(text)))
    toks = [CONCEPT_OF.get(t, CONCEPT_OF.get(_stem(t), t)) for t in toks]
    out = [_stem(t) for t in toks if t not in TOPIC_STOPWORDS and not _EMOJI_OR_PUNCT.match(t)
           and not t.isdigit() and len(t) > 1]
    return [t for t in out if t not in TOPIC_STOPWORDS]          # "playlists" -> "playlist" (an intent word)


def topic_text(text: str) -> str:
    return " ".join(topic_tokens(text))


# --------------------------------------------------------------------------
# Near-duplicates
# --------------------------------------------------------------------------

class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def near_duplicate_groups(texts: Sequence[str], threshold: float = NEAR_DUP_THRESHOLD) -> np.ndarray:
    """Group id per text; texts with char-TF-IDF cosine >= threshold share a group.

    Group ids are the index of the group's first member, so they are stable for
    a given input order. Empty normalised texts each get their own group.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    n = len(texts)
    if n == 0:
        return np.array([], dtype=int)
    norm = [normalize(t) for t in texts]
    uf = _UnionFind(n)
    # Exact normalised matches first (covers very short texts like "python").
    first = {}
    for i, t in enumerate(norm):
        if t:
            if t in first:
                uf.union(first[t], i)
            else:
                first[t] = i
    nonempty = [i for i, t in enumerate(norm) if t]
    if len(nonempty) > 1:
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=False)
        try:
            X = vec.fit_transform([norm[i] for i in nonempty])
            S = (X @ X.T).tocoo()
            for a, b, v in zip(S.row, S.col, S.data):
                if a < b and v >= threshold:
                    uf.union(nonempty[a], nonempty[b])
        except ValueError:  # vocabulary empty
            pass
    return np.array([uf.find(i) for i in range(n)])


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------

def _l2(X):
    from sklearn.preprocessing import normalize as l2norm
    if X.shape[1] == 0:
        return sparse.csr_matrix(X)
    return l2norm(X, norm="l2")


def build_features(texts: Sequence[str], kind: str = "word+char", topic_texts: Optional[Sequence[str]] = None):
    """Row-normalised feature matrix for topic similarity (fitted on these texts only).

    Returns (X, word_vectorizer). word_vectorizer is used for naming topics and
    for the keyword-only check (None for embeddings).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    tt = list(topic_texts) if topic_texts is not None else [topic_text(t) for t in texts]
    word_vec = TfidfVectorizer(analyzer=lambda s: _word_ngrams(s), sublinear_tf=True)
    if kind == "embeddings":
        X = sparse.csr_matrix(embed(texts))
        try:
            word_vec.fit(tt)
        except ValueError:
            word_vec = None
        return _l2(X), word_vec
    parts = []
    if kind in ("word", "word+char"):
        try:
            W = word_vec.fit_transform(tt)
        except ValueError:
            W = sparse.csr_matrix((len(tt), 0))
            word_vec = None
        parts.append(_l2(W) * (WORD_WEIGHT if kind == "word+char" else 1.0))
    if kind in ("char", "word+char"):
        char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        try:
            C = char_vec.fit_transform(tt)
        except ValueError:
            C = sparse.csr_matrix((len(tt), 0))
        parts.append(_l2(C) * (CHAR_WEIGHT if kind == "word+char" else 1.0))
        if kind == "char":
            try:
                word_vec.fit(tt)
            except ValueError:
                word_vec = None
    if not parts:
        raise ValueError(f"unknown feature kind {kind!r}")
    X = sparse.hstack(parts).tocsr()
    return _l2(X), word_vec


def _word_ngrams(s: str) -> List[str]:
    toks = s.split()
    return toks + [f"{a} {b}" for a, b in zip(toks, toks[1:])]


_EMBEDDER = {}


def embed(texts: Sequence[str], model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> np.ndarray:
    """Multilingual sentence embeddings (optional dependency)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("--features embeddings needs `pip install sentence-transformers`") from exc
    if model_name not in _EMBEDDER:
        _EMBEDDER[model_name] = SentenceTransformer(model_name)
    return _EMBEDDER[model_name].encode(list(texts), normalize_embeddings=True, show_progress_bar=False)


# --------------------------------------------------------------------------
# Clustering
# --------------------------------------------------------------------------

def _agglomerate(X, threshold: float) -> np.ndarray:
    n = X.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0])
    from sklearn.cluster import AgglomerativeClustering
    D = 1.0 - (X @ X.T).toarray()
    np.fill_diagonal(D, 0.0)
    D = np.clip(D, 0.0, 2.0)
    model = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average",
                                    distance_threshold=1.0 - threshold)
    return model.fit_predict(D)


def _centroid_sims(X, idx: np.ndarray) -> np.ndarray:
    c = np.asarray(X[idx].mean(axis=0)).ravel()
    norm = np.linalg.norm(c)
    if norm == 0:
        return np.zeros(len(idx))
    return np.asarray(X[idx] @ (c / norm)).ravel()


def _shared_terms(word_tokens: List[List[str]], idx: Iterable[int]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for i in idx:
        for t in set(word_tokens[i]):
            counts[t] = counts.get(t, 0) + 1
    return counts


def _keyword_only(word_tokens: List[List[str]], idx: List[int]) -> Optional[str]:
    """The single word holding a group together, if nothing else is shared.

    True when one content word is in every member but no other word is shared
    by more than one member.
    """
    counts = _shared_terms(word_tokens, idx)
    if not counts:
        return None
    everywhere = [t for t, k in counts.items() if k == len(idx)]
    others = [t for t, k in counts.items() if k > 1 and t not in everywhere]
    if len(everywhere) == 1 and not others:
        return everywhere[0]
    return None


def _split_without_dominant_term(texts, word_tokens, members, threshold, min_size, features):
    """Re-cluster a group with its most shared word removed.

    If the group falls apart into >= 2 sub-groups of >= min_size, the shared
    word was the only glue (e.g. "dynamic programming lectures" vs "which
    programming language is best"), so the sub-groups are returned. Returns
    None when the group is held together by more than that one word.
    """
    if len(members) < 2 * min_size:
        return None
    counts = _shared_terms(word_tokens, members)
    if not counts:
        return None
    term, k = max(counts.items(), key=lambda kv: (kv[1], kv[0]))
    if k < 0.6 * len(members):
        return None
    reduced = [" ".join(t for t in word_tokens[i] if t != term) for i in range(len(texts))]
    X2, _ = build_features(texts, features if features != "embeddings" else "word+char", topic_texts=reduced)
    sub = np.array(members)
    has = np.array([X2[i].nnz > 0 for i in sub])
    if has.sum() < 2 * min_size:
        return None
    labels = _agglomerate(X2[sub[has]], threshold)
    parts = [list(sub[has][labels == g]) for g in sorted(set(labels))]
    parts = [p for p in parts if len(p) >= min_size]
    return (term, parts) if len(parts) >= 2 else None


def cluster_unique(texts: Sequence[str], intents: Sequence[str], threshold: float = CLUSTER_THRESHOLD,
                   min_size: int = MIN_TOPIC_SIZE, min_member_sim: float = MIN_MEMBER_SIM,
                   features: str = "word+char") -> dict:
    """Cluster unique comments. Returns labels (-1 = Unclustered) and diagnostics."""
    n = len(texts)
    labels = np.full(n, UNCLUSTERED_ID)
    sim_to_centroid = np.zeros(n)
    notes: Dict[int, str] = {}
    if n == 0:
        return {"labels": labels, "sim": sim_to_centroid, "X": None, "word_vec": None, "notes": notes}
    X, word_vec = build_features(texts, features)
    word_tokens = [topic_tokens(t) for t in texts]
    has_content = np.array([X[i].nnz > 0 for i in range(n)])
    idx_all = np.where(has_content)[0]
    raw = np.full(n, -1)
    if len(idx_all):
        raw[idx_all] = _agglomerate(X[idx_all], threshold)

    groups = []
    for g in sorted(set(raw[idx_all])):
        members = list(np.where(raw == g)[0])
        kw = _keyword_only(word_tokens, members) if len(members) > 1 else None
        fams = {INTENT_FAMILY.get(intents[i], "other") for i in members}
        if kw and len(fams) > 1:
            for fam in sorted(fams):
                sub = [i for i in members if INTENT_FAMILY.get(intents[i], "other") == fam]
                groups.append((sub, f"split by intent: only shared word '{kw}'"))
            continue
        split = _split_without_dominant_term(texts, word_tokens, members, threshold, min_size, features)
        if split:
            term, parts = split
            groups.extend((p, f"split: sub-groups shared only the word '{term}'") for p in parts)
        else:
            groups.append((members, ""))

    next_id = 0
    # Deterministic topic ids: larger topics first, ties by first member index.
    for members, note in sorted(groups, key=lambda gm: (-len(gm[0]), min(gm[0]))):
        members = np.array(members)
        if len(members) < min_size:
            continue
        sims = _centroid_sims(X, members)
        keep = members[sims >= min_member_sim]
        if len(keep) < min_size:
            continue
        sims = _centroid_sims(X, keep)
        labels[keep] = next_id
        sim_to_centroid[keep] = sims
        if note:
            notes[next_id] = note
        next_id += 1
    for i in np.where(labels == UNCLUSTERED_ID)[0]:
        sim_to_centroid[i] = 0.0
    return {"labels": labels, "sim": sim_to_centroid, "X": X, "word_vec": word_vec, "notes": notes}


# --------------------------------------------------------------------------
# Naming
# --------------------------------------------------------------------------

def _pretty_word(w: str) -> str:
    if w in CONCEPTS:
        return CONCEPTS[w][0]
    m = _SQR.match(w)
    if m:
        return f"{m.group(1)}\u00b2"                      # ksqr -> k²
    return w.upper() if w in ACRONYMS else (w[:1].upper() + w[1:])


def _pretty(term: str) -> str:
    return " ".join(_pretty_word(w) for w in term.split())


def is_generic(term: str) -> bool:
    """True when every word of the term is a generic word (understand, doubt, explain...)."""
    return all(w in GENERIC_TERMS for w in term.split())


PHRASES: Dict[int, str] = {}
COVERAGE: Dict[int, Dict[str, float]] = {}     # topic -> term -> share of its comments containing it


def topic_terms(word_tokens: List[List[str]], labels: np.ndarray, top_k: int = 5) -> Dict[int, List[str]]:
    """Class-based TF-IDF: terms frequent in a topic but rare in other topics."""
    topics = sorted(t for t in set(labels) if t != UNCLUSTERED_ID)
    if not topics:
        return {}
    docs = {t: [] for t in topics}
    for toks, lab in zip(word_tokens, labels):
        if lab != UNCLUSTERED_ID:
            docs[lab].append(toks)
    df_count: Dict[str, int] = {}
    tf: Dict[int, Dict[str, int]] = {}
    for t, rows in docs.items():
        c: Dict[str, int] = {}
        for toks in rows:
            for term in set(_word_ngrams(" ".join(toks))):   # count each comment once
                c[term] = c.get(term, 0) + 1
        tf[t] = c
        for term in c:
            df_count[term] = df_count.get(term, 0) + 1
    out = {}
    PHRASES.clear()
    COVERAGE.clear()
    for t in topics:
        size = len(docs[t])
        COVERAGE[t] = {term: k / size for term, k in tf[t].items()}
        bigrams = [(k, term) for term, k in tf[t].items() if " " in term and k >= max(2, size / 2)
                   and not is_generic(term)]
        if bigrams:
            PHRASES[t] = sorted(bigrams, key=lambda kv: (-kv[0], kv[1]))[0][1]
        scored = []
        for term, k in tf[t].items():
            if k < 2 and size > 1:
                continue           # a term must recur within the topic
            idf = np.log(1 + len(topics) / df_count[term])
            bonus = 1.15 if " " in term else 1.0
            if is_generic(term):
                bonus *= 0.3           # "understand", "doubt": never the name if a subject word exists
            scored.append(((k / size) * idf * bonus, term))
        scored.sort(key=lambda s: (-s[0], s[1]))
        out[t] = [term for _, term in scored[:top_k]]
    return out


def _case_forms(texts: List[str]) -> Dict[str, Dict[str, int]]:
    """Spellings seen in the comments, keyed by lower-case word and by its singular form
    (so the topic term "note" is shown as the commenters' "Notes")."""
    forms: Dict[str, Dict[str, int]] = {}
    for text in texts:
        for t in _raw_tokens(_ELONGATED.sub(_unstretch, text)):
            for key in dict.fromkeys([t.lower(), _stem(t.lower())]):
                forms.setdefault(key, {}).setdefault(t, 0)
                forms[key][t] += 1
    return forms


def topic_name(terms: List[str], phrase: Optional[str] = None, forms=None, coverage=None) -> str:
    """Readable name from the top terms; a phrase shared by >= half the topic wins.

    `forms` (word -> spellings seen in the comments) keeps the comments' own
    spelling, e.g. "ls" stays lower-case and "DevOps" keeps its capitals.
    """
    def pretty(term: str) -> str:
        if not forms:
            return _pretty(term)
        return " ".join(_display_case(forms[w]) if w in forms and not _SQR.match(w) and w not in CONCEPTS
                        else _pretty_word(w) for w in term.split())
    def trim(term: str) -> str:              # "necessary python" -> "python"
        words = term.split()
        while len(words) > 1 and words[0] in GENERIC_TERMS:
            words = words[1:]
        while len(words) > 1 and words[-1] in GENERIC_TERMS:
            words = words[:-1]
        return " ".join(words)
    if phrase:
        return pretty(trim(phrase))
    terms = [trim(t) for t in terms if not is_generic(t)] or terms
    if not terms:
        return "Mixed topic"
    concept = next((t for t in terms[:2] if t in CONCEPTS), None)
    if concept:                               # "Study materials: Operating System"
        other = next((t for t in terms[:3] if t != concept and not any(w in CONCEPTS for w in t.split())), None)
        return f"{CONCEPTS[concept][0]}: {pretty(other)}" if other else CONCEPTS[concept][0]
    first = terms[0]
    if " " in first:
        return pretty(first)
    # Combine with a second subject word that is not part of the first.
    for t in terms[1:]:
        if " " not in t and t != first and not is_generic(t) and (coverage is None or coverage.get(t, 0) >= 0.4):
            sep = " vs " if _SQR.match(first) or _SQR.match(t) else " / "
            return f"{pretty(first)}{sep}{pretty(t)}"
    return pretty(first)


def medoid(X, idx: np.ndarray) -> int:
    if len(idx) == 1:
        return int(idx[0])
    S = (X[idx] @ X[idx].T).toarray()
    return int(idx[int(np.argmax(S.sum(axis=1)))])



# --------------------------------------------------------------------------
# Merging topics about the same thing, and naming from real phrases
# --------------------------------------------------------------------------

MERGE_MIN_SIM = 0.20      # centroid cosine needed (on top of sharing the leading term)


def _family(intents, idx) -> str:
    fams = pd.Series([INTENT_FAMILY.get(intents[i], "other") for i in idx])
    return fams.mode().iloc[0] if len(fams) else "other"


def merge_same_subject(labels: np.ndarray, X, word_tokens, intents) -> tuple:
    """Merge topics that are about the same subject.

    Two topics merge when their most distinctive term is the same, most of their
    comments have the same kind of intent (asking vs requesting), and their
    centroids are similar (>= MERGE_MIN_SIM). Example: "ls -a is not working" and
    "use ls -Force instead" are one problem seen from two sides.
    Returns (new_labels, merged_pairs).
    """
    topics = sorted(t for t in set(labels) if t != UNCLUSTERED_ID)
    if len(topics) < 2:
        return labels, []
    terms = topic_terms(word_tokens, labels, top_k=1)
    cents = {}
    for t in topics:
        c = np.asarray(X[np.where(labels == t)[0]].mean(axis=0)).ravel()
        n = np.linalg.norm(c)
        cents[t] = c / n if n else c
    parent = {t: t for t in topics}

    def find(t):
        while parent[t] != t:
            t = parent[t]
        return t
    merged = []
    core = _core_terms(labels, word_tokens, topics)
    for i, a in enumerate(topics):
        for b in topics[i + 1:]:
            ta, tb = terms.get(a, []), terms.get(b, [])
            same_lead = bool(ta and tb and ta[0].split()[0] == tb[0].split()[0])
            if not same_lead and not (core[a] & core[b]):
                continue
            ia, ib = np.where(labels == a)[0], np.where(labels == b)[0]
            if same_lead:
                if _family(intents, ia) != _family(intents, ib):
                    continue
            elif not ({INTENT_FAMILY.get(intents[i], "other") for i in ia}
                      & {INTENT_FAMILY.get(intents[i], "other") for i in ib}):
                continue
            if float(cents[a] @ cents[b]) < MERGE_MIN_SIM:
                continue
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
                merged.append((a, b))
    if not merged:
        return labels, []
    new = labels.copy()
    for t in topics:
        new[labels == t] = find(t)
    return renumber(new), merged


def renumber(labels: np.ndarray) -> np.ndarray:
    """Topic ids 0..n-1: larger topics first, ties by first member."""
    order = sorted({t for t in set(labels) if t != UNCLUSTERED_ID},
                   key=lambda t: (-(labels == t).sum(), int(np.where(labels == t)[0][0])))
    remap = {old: i for i, old in enumerate(order)}
    return np.array([remap.get(t, UNCLUSTERED_ID) for t in labels])


CORE_SHARE = 0.6          # a core term is in >= 60% of a topic's comments...
CORE_MAX_DF = 0.4         # ...but in <= 40% of all the comments (not the video's general subject)


def _core_terms(labels, word_tokens, topics) -> Dict[int, set]:
    """Subject words most of a topic's comments contain ("string" in "string playlist please")."""
    df: Dict[str, int] = {}
    for i in range(len(word_tokens)):
        for w in set(word_tokens[i]):
            df[w] = df.get(w, 0) + 1
    out = {}
    for t in topics:
        idx = np.where(labels == t)[0]
        counts = _shared_terms(word_tokens, idx)
        out[t] = {w for w, k in counts.items() if k >= max(2, CORE_SHARE * len(idx)) and w not in GENERIC_TERMS
                  and df.get(w, 0) <= CORE_MAX_DF * len(word_tokens)}
    return out


def drop_subjectless(labels: np.ndarray, word_tokens) -> tuple:
    """Ungroup topics whose comments share no subject word, only generic ones
    ("didn't understand", "nice explanation"): there is nothing meaningful to name them by."""
    out = labels.copy()
    dropped = []
    for t in sorted(set(labels) - {UNCLUSTERED_ID}):
        idx = np.where(labels == t)[0]
        counts = _shared_terms(word_tokens, idx)
        if not any(k >= 2 and w not in GENERIC_TERMS for w, k in counts.items()):
            out[idx] = UNCLUSTERED_ID
            dropped.append(int(t))
    return (renumber(out) if dropped else out), dropped


_RAW_TOKEN = re.compile(r"[\w.+#-]*\w[\w.+#-]*", re.UNICODE)


def _raw_tokens(text: str) -> List[str]:
    """Tokens as written, keeping command flags and dotted names (ls -a, .git, node.js)."""
    toks = []
    for t in _RAW_TOKEN.findall(normalize_spaces(text)):
        t = t.strip(".").rstrip("-") if not t.startswith(("-", ".")) else t.rstrip(".")
        if t and re.search(r"[^\W\d_]", t):           # must contain a letter
            toks.append(t)
    return toks


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text))


def _display_case(forms: Dict[str, int]) -> str:
    """Most common spelling; mixed-case forms (GitDemo, DevOps) win ties; long plain words get a capital."""
    best = sorted(forms.items(), key=lambda kv: (-kv[1], kv[0] == kv[0].lower(), kv[0]))[0][0]
    if best.lower() in ACRONYMS:
        return best.upper()
    if best == best.lower() and best.isalpha() and len(best) > 2:
        return best[:1].upper() + best[1:]
    return best


_HONORIFIC = r"(sir|mam|maam|ma'am|madam|didi|di|bhaiya|bhai|bro|hello|hi|hey|dear|please|pls|plz|plzz|guys|everyone)"
_LEAD = re.compile(rf"^(\W*\b({_HONORIFIC[1:-1]}|vote for|petition to \w+)\b[\s,.!:-]*)+", re.I)
_TRAIL = re.compile(rf"([\s,.!]*\b{_HONORIFIC}\b\W*)+$", re.I)
_KEEP = re.compile(r"[^\w\s?.,'+&/#-]", re.UNICODE)


def clean_comment_title(text: str, max_words: int = 9) -> Optional[str]:
    """A short real comment cleaned up to serve as a topic name ("Can I join it now?"), or None."""
    t = re.sub(r"(?<![\d:])\d{1,2}[:.]\d{2}(?::\d{2})?(?![\d:])", " ", _ELONGATED.sub(_unstretch, str(text)))
    t = _KEEP.sub(" ", t)
    t = re.sub(r"[.?!]*\?[.?!]*", "?", t)
    t = re.sub(r"([.!,])\1+", r"\1", t)
    t = re.sub(r"\s+([?.,])(?=\s|$)", r"\1", re.sub(r"\s+", " ", t)).strip(" ,.-")
    t = _TRAIL.sub("", _LEAD.sub("", t)).strip(" ,.-")
    words = t.split()
    if not (2 <= len(words) <= max_words) or len(t) > 60:
        return None
    letters = [c for c in t if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.6:
        t = t.lower()                                             # DO I GET RECORDED LECTURES
    t = re.sub(r"\bi\b", "I", t)
    return t[:1].upper() + t[1:]


# Words that already name a subject well on their own (no need to borrow a comment).
SUBJECT_WORDS = {"git", "github", "python", "java", "javascript", "linux", "docker", "react", "excel", "string",
                 "strings", "heap", "array", "arrays", "english", "hindi", "math", "maths", "mathematics", "physics",
                 "chemistry", "biology", "statistics", "sql", "kotlin", "flutter", "android", "django", "flask",
                 "kubernetes", "aws", "azure", "pandas", "numpy", "tensorflow", "pytorch", "golang", "rust", "php",
                 "laptop", "price", "placement", "placements", "internship", "salary", "utilisation", "utilization"}


def name_from_comments(name: str, texts: List[str], sims: List[float]) -> str:
    """A one-word name ("Join", "Recorded") says little. Use the most typical short comment that
    contains the word instead ("Can I join it now?"). Acronyms and capitalised names are kept."""
    if (" " in name or "/" in name or name.isupper() or any(c.isupper() for c in name[1:]) or not name.isalpha()
            or name.lower() in SUBJECT_WORDS):
        return name
    word = name.lower()
    for _, text in sorted(zip(sims, texts), key=lambda st: -st[0]):
        title = clean_comment_title(text)
        if not title or not re.search(rf"\b{re.escape(word)}", title.lower()) or _REQUEST_FILLER.search(title):
            continue
        extra = [w for w in topic_tokens(title) if w != _stem(word) and w != word and w not in GENERIC_TERMS]
        if title.endswith("?") or _QUESTION_WORD.match(title) or extra:
            return title
    return name


_REQUEST_FILLER = re.compile(r"\b(make|video|videos|please|pls|plz|plzz|bana\w*|la+o+|le aoo|lao|dijiye|chahiye|"
                             r"padha\w*|karo|kardo|kar do|kr do|launch|upload)\b", re.I)
_QUESTION_WORD = re.compile(r"^(is|are|can|could|do|does|did|how|what|why|which|when|where|should|will|who)\b", re.I)


def with_core_word(name: str, member_tokens: List[List[str]], forms) -> str:
    """Add the word nearly every comment shares when the name misses it:
    "Operating System" + notes in 6 of 6 comments -> "Operating System Notes"."""
    n = len(member_tokens)
    if n < 3:
        return name
    counts = _shared_terms(member_tokens, range(n))
    cands = [(k, w) for w, k in counts.items() if k >= 0.8 * n and w not in GENERIC_TERMS]
    if not cands:
        return name
    k, w = max(cands, key=lambda kw: (kw[0], kw[1]))
    shown = _display_case(forms[w]) if w in forms and not _SQR.match(w) else _pretty_word(w)
    have = {x.lower() for x in re.findall(r"[\w²+#.-]+", name)} | {_stem(x.lower()) for x in re.findall(r"\w+", name)}
    if (w in have or shown.lower() in have or "/" in name or " vs " in name or ":" in name or len(name.split()) > 3
            or any(name.startswith(lbl) for lbl, _ in CONCEPTS.values())):
        return name
    return f"{name} {shown}"


def phrase_name(raw_texts: List[str], top_terms: List[str]) -> Optional[str]:
    """A 2-3 word phrase that most of the topic's comments actually contain (e.g. "ls -a", "GitDemo folder")."""
    n = len(raw_texts)
    if n < 2 or not top_terms:
        return None
    keys = {w for t in top_terms[:3] for w in t.split() if w not in GENERIC_TERMS}
    counts: Dict[tuple, int] = {}
    forms: Dict[str, Dict[str, int]] = {}
    for text in raw_texts:
        toks = _raw_tokens(text)
        for t in toks:
            forms.setdefault(t.lower(), {}).setdefault(t, 0)
            forms[t.lower()][t] += 1
        seen = set()
        low = [t.lower() for t in toks]
        for size in (2, 3):
            for i in range(len(low) - size + 1):
                g = tuple(low[i:i + size])
                if g in seen or not ({w for w in g} | {_stem(w) for w in g}) & keys:
                    continue
                edge_ok = all(not ((w in TOPIC_STOPWORDS or w in GENERIC_TERMS) and not w.startswith("-"))
                              for w in (g[0], g[-1]))
                if edge_ok:
                    seen.add(g)
                    counts[g] = counts.get(g, 0) + 1
    need = max(2, int(np.ceil(n / 2)))
    good = [(c, len(g), g) for g, c in counts.items() if c >= need]
    if not good:
        return None
    c, size, g = sorted(good, key=lambda x: (-x[0], -x[1], x[2]))[0]
    words = []
    for i, w in enumerate(g):
        nxt = g[i + 1] if i + 1 < len(g) else ""
        if nxt.startswith("-"):                      # a command followed by a flag keeps its spelling: ls -a
            words.append(max(forms.get(w, {w: 1}).items(), key=lambda kv: (kv[1], kv[0] == kv[0].lower()))[0])
        else:
            words.append(_display_case(forms.get(w, {w: 1})))
    return " ".join(words)

# --------------------------------------------------------------------------
# Stage runner
# --------------------------------------------------------------------------

def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding=CSV_ENCODING)


def select_comments(pred: pd.DataFrame, categories: Sequence[str], allow_spam: bool = False) -> pd.DataFrame:
    cats = [c for c in categories if c != "Spam" or allow_spam]
    sel = pred[pred["predicted_category"].isin(cats)].copy()
    sel["_ts"] = pd.to_datetime(sel["published_at"], utc=True, errors="coerce")
    # Stable order: oldest first so a duplicate group's canonical row is its first appearance.
    sel = sel.sort_values(["_ts", "comment_id"], na_position="last", kind="stable").reset_index(drop=True)
    return sel


def discover(pred: pd.DataFrame, categories: Sequence[str] = DEFAULT_CATEGORIES,
             threshold: float = CLUSTER_THRESHOLD, min_size: int = MIN_TOPIC_SIZE,
             features: str = "word+char", near_dup_threshold: float = NEAR_DUP_THRESHOLD,
             allow_spam: bool = False):
    """Return (clusters_df, summary_df, stats). Spam is dropped unless allow_spam=True."""
    sel = select_comments(pred, categories, allow_spam)
    if sel.empty:
        cols = ["topic_id", "topic_name"]
        return (sel.assign(topic_id=pd.Series(dtype=int), topic_name=pd.Series(dtype=str)),
                pd.DataFrame(columns=cols), {"selected_comments": 0, "topics": 0})
    groups = near_duplicate_groups(sel["text"].tolist(), near_dup_threshold)
    sel["near_dup_group"] = groups
    sel["near_dup_group_size"] = sel.groupby("near_dup_group")["comment_id"].transform("size")
    sel["is_canonical"] = ~sel.duplicated("near_dup_group", keep="first")
    canon = sel[sel["is_canonical"]].reset_index()          # 'index' -> row in sel
    res = cluster_unique(canon["text"].tolist(), canon["predicted_category"].tolist(),
                         threshold=threshold, min_size=min_size, features=features)
    word_tokens = [topic_tokens(t) for t in canon["text"]]
    if res["X"] is not None:
        labels, merged = merge_same_subject(res["labels"], res["X"], word_tokens,
                                            canon["predicted_category"].tolist())
        labels, dropped = drop_subjectless(labels, word_tokens)
        if merged or dropped:
            res["labels"] = labels
            res["notes"] = {}
            for t in set(labels) - {UNCLUSTERED_ID}:
                idx = np.where(labels == t)[0]
                res["sim"][idx] = _centroid_sims(res["X"], idx)
    canon["topic_id"] = res["labels"]
    canon["similarity_to_topic"] = np.round(res["sim"], 3)

    # Propagate the canonical comment's topic to its duplicates.
    gmap = canon.set_index("near_dup_group")
    sel["topic_id"] = sel["near_dup_group"].map(gmap["topic_id"]).astype(int)
    sel["similarity_to_topic"] = sel["near_dup_group"].map(gmap["similarity_to_topic"])

    terms = topic_terms(word_tokens, res["labels"])
    names = {}
    for t, v in terms.items():
        texts = canon.loc[res["labels"] == t, "text"].tolist()
        forms = _case_forms(texts)
        names[t] = phrase_name(texts, v) or topic_name(v, PHRASES.get(t), forms, COVERAGE.get(t))
        idx = np.where(res["labels"] == t)[0]
        names[t] = with_core_word(names[t], [word_tokens[i] for i in idx], forms)
        if sum(word_tokens[i] == [CHECKIN_TOKEN] for i in idx) * 2 >= len(idx):
            names[t] = CONCEPTS[CHECKIN_TOKEN][0]
        else:
            names[t] = name_from_comments(names[t], texts, [float(res["sim"][i]) for i in idx])
    # Disambiguate identical names.
    seen: Dict[str, int] = {}
    for t in sorted(names):
        base = names[t]
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            names[t] = f"{base} ({seen[base]})"
    names[UNCLUSTERED_ID] = UNCLUSTERED_NAME
    sel["topic_name"] = sel["topic_id"].map(names)

    # Representative = medoid of the topic's unique comments.
    rep_rows = {}
    X = res["X"]
    for t in terms:
        idx = np.where(res["labels"] == t)[0]
        rep_rows[t] = int(canon.loc[medoid(X, idx), "index"])
    sel["is_representative"] = False
    for t, r in rep_rows.items():
        sel.loc[r, "is_representative"] = True
    sel["topic_note"] = sel["topic_id"].map(res["notes"]).fillna("")
    sel["topic_text"] = sel["text"].map(topic_text)

    summary = summarise_topics(sel, terms, names, rep_rows)
    stats = {
        "selected_comments": int(len(sel)),
        "categories": list(categories),
        "unique_comments_after_near_dup": int(sel["is_canonical"].sum()),
        "near_duplicate_groups_with_repeats": int((sel.groupby("near_dup_group").size() > 1).sum()),
        "comments_in_duplicate_groups": int((sel["near_dup_group_size"] > 1).sum()),
        "topics": int(len(terms)),
        "clustered_comments": int((sel["topic_id"] != UNCLUSTERED_ID).sum()),
        "unclustered_comments": int((sel["topic_id"] == UNCLUSTERED_ID).sum()),
        "features": features, "threshold": threshold, "min_topic_size": min_size,
        "near_dup_threshold": near_dup_threshold,
    }
    return sel.drop(columns=["_ts"]), summary, stats


def _join_unique(values, limit=None) -> str:
    vals = [v for v in dict.fromkeys(values) if v]
    return " | ".join(vals[:limit] if limit else vals)


def summarise_topics(sel: pd.DataFrame, terms, names, rep_rows) -> pd.DataFrame:
    rows = []
    has_author = (sel["author_channel_id"] != "").any()
    for t in sorted(set(sel["topic_id"]), key=lambda x: (x == UNCLUSTERED_ID, x)):
        g = sel[sel["topic_id"] == t]
        u = g[g["is_canonical"]]
        rep = sel.loc[rep_rows[t]] if t in rep_rows else None
        intents = u["predicted_category"].value_counts()
        ts = pd.to_datetime(g["published_at"], utc=True, errors="coerce")
        rows.append({
            "topic_id": int(t),
            "topic_name": names[t],
            "top_terms": "; ".join(terms.get(t, [])),
            "n_comments_raw": int(len(g)),
            "n_unique_comments": int(len(u)),
            "n_unique_commenters": int(g.loc[g["author_channel_id"] != "", "author_channel_id"].nunique())
            if has_author else "",
            "n_videos": int(g["video_id"].nunique()),
            "n_channels": int(g.loc[g["channel_title"] != "unknown", "channel_title"].nunique()),
            "n_question": int(intents.get("Question", 0)),
            "n_request": int(intents.get("Request", 0)),
            "n_confusion_problem": int(intents.get("Confusion/Problem", 0)),
            "n_other_categories": int(len(u) - intents.reindex(DEFAULT_CATEGORIES).fillna(0).sum()),
            "dominant_intent": intents.index[0] if len(intents) else "",
            "mean_similarity_to_topic": round(float(u["similarity_to_topic"].astype(float).mean()), 3)
            if t != UNCLUSTERED_ID else "",
            "representative_comment_id": rep["comment_id"] if rep is not None else "",
            "representative_comment": rep["text"] if rep is not None else "",
            "video_ids": _join_unique(g["video_id"]),
            "video_titles": _join_unique(g["video_title"]),
            "channels": _join_unique(g["channel_title"]),
            "first_comment_at": ts.min().isoformat() if ts.notna().any() else "",
            "last_comment_at": ts.max().isoformat() if ts.notna().any() else "",
            "topic_kind": ("check-in" if (u["text"].map(is_checkin).mean() >= 0.5 if len(u) else False) else "need")
            if t != UNCLUSTERED_ID else "",
            "note": g["topic_note"].iloc[0] if t != UNCLUSTERED_ID else
            "comments that did not form a group of >= 2 similar unique comments",
        })
    return pd.DataFrame(rows)


CLUSTER_COLUMNS = ["comment_id", "topic_id", "topic_name", "is_representative", "similarity_to_topic",
                   "predicted_category", "text", "topic_text", "near_dup_group", "near_dup_group_size",
                   "is_canonical", "video_id", "video_title", "channel_title", "channel_id",
                   "author_channel_id", "published_at", "creator_replied", "reply_count", "like_count",
                   "seen_in_training", "low_confidence"]


def run(in_path: Path = IN_PREDICTED, out_dir: Path = PROCESSED_DATA_DIR, categories=None,
        threshold: float = CLUSTER_THRESHOLD, min_size: int = MIN_TOPIC_SIZE, features: str = "word+char",
        allow_spam: bool = False) -> dict:
    pred = read_csv(Path(in_path))
    need = {"comment_id", "text", "predicted_category"}
    if not need <= set(pred.columns):
        raise SystemExit(f"[ERROR] {in_path} is missing {sorted(need - set(pred.columns))}; run src.batch_predict first")
    for c in CLUSTER_COLUMNS:
        if c not in pred.columns and c not in ("topic_id", "topic_name", "is_representative",
                                               "similarity_to_topic", "topic_text", "near_dup_group",
                                               "near_dup_group_size", "is_canonical"):
            pred[c] = ""
    cats = categories or DEFAULT_CATEGORIES
    clusters, summary, stats = discover(pred, cats, threshold, min_size, features, allow_spam=allow_spam)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clusters = clusters.sort_values(["topic_id", "is_representative", "similarity_to_topic"],
                                    ascending=[True, False, False], key=None)
    clusters = pd.concat([clusters[clusters["topic_id"] != UNCLUSTERED_ID],
                          clusters[clusters["topic_id"] == UNCLUSTERED_ID]])
    clusters[[c for c in CLUSTER_COLUMNS if c in clusters.columns]].to_csv(
        out_dir / OUT_CLUSTERS.name, index=False, encoding=CSV_ENCODING)
    summary.to_csv(out_dir / OUT_SUMMARY.name, index=False, encoding=CSV_ENCODING)
    return stats


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    p = argparse.ArgumentParser(description="Group similar audience comments into topics.")
    p.add_argument("--input", default=str(IN_PREDICTED))
    p.add_argument("--out-dir", default=str(PROCESSED_DATA_DIR))
    p.add_argument("--include", nargs="*", default=[], choices=OPTIONAL_CATEGORIES,
                   help="also cluster these categories (Spam is never clustered)")
    p.add_argument("--threshold", type=float, default=CLUSTER_THRESHOLD)
    p.add_argument("--min-size", type=int, default=MIN_TOPIC_SIZE)
    p.add_argument("--features", default="word+char", choices=["word", "char", "word+char", "embeddings"])
    a = p.parse_args(argv)
    stats = run(Path(a.input), Path(a.out_dir), DEFAULT_CATEGORIES + a.include, a.threshold, a.min_size, a.features)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
