"""
CreatorPulse — synthetic comment generator (template + slot filling).

Builds a class-balanced set of synthetic YouTube comments for the six
`label_6class` intents (Question, Request, Confusion/Problem, Praise, Spam,
Other), written in the same mix of English, Hinglish and Devanagari Hindi as
the real data, for the same kinds of channels (coding, maths, JEE/NEET).

IMPORTANT — how to use this data
--------------------------------
* Synthetic rows are for TRAINING ONLY. Never put them in a test set: a model
  scores ~100% on text made by the same templates, which says nothing about
  real comments. All evaluation stays on the real held-out splits.
* Every row's label is the intent its template was written for, so labels are
  correct by construction for the template — but the templates are one
  person's idea of each class, not real viewer behaviour.
* Rows whose character n-gram TF-IDF cosine similarity to ANY real labelled
  comment is >= 0.70 (the leakage threshold used by src.make_splits) are
  dropped, so no synthetic row is a near-copy of a real test comment.

Labelling rules follow data/labelled/LABELLING_GUIDE.md (6-class version):
tie-break Spam > Confusion/Problem > Question > Request > Praise > Other, e.g.
"great video but my code throws X" -> Confusion/Problem, "amazing! please make
one on graphs" -> Request, "can you make ...?" -> Request, joke "area 51 next"
-> Other, "who is watching in 2026?" -> Other, pin/reply/job asks -> Other.
Confusion/Problem (stuck, an error, something broken, did not understand) is
kept separate from Question (an answerable information or advice question).

Output: data/synthetic/synthetic_comments.csv
    comment_id, video_id, channel_title, text, label, label_6class,
    language, template_id, source

The output is deterministic for a given --seed and --per-class.

Usage:
    python -m src.generate_synthetic                 # 1667 per class -> 10,002 rows
    python -m src.generate_synthetic --per-class 500
"""

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, LABELLED_DATA_DIR
from src.preprocess import normalize

CLASSES = ["Question", "Request", "Confusion/Problem", "Praise", "Spam", "Other"]
SEED = 42
PER_CLASS = 1667
LEAKAGE_THRESHOLD = 0.70
SOURCE = "synthetic_template_v1"
OUTPUT = DATA_DIR / "synthetic" / "synthetic_comments.csv"
REAL_LABELLED = LABELLED_DATA_DIR / "comments_labelled.csv"
CSV_ENCODING = "utf-8-sig"

# --------------------------------------------------------------------------
# Slot vocabularies
# --------------------------------------------------------------------------

DOMAINS = {
    "coding": {
        "channel": "synthetic_coding",
        "topics": [
            "python", "java", "c++", "c language", "javascript", "react", "node.js", "express",
            "django", "flask", "fastapi", "DSA", "sql", "mysql", "mongodb", "postgresql", "git",
            "github", "linux", "docker", "kubernetes", "machine learning", "deep learning",
            "data science", "next.js", "typescript", "html", "css", "tailwind css", "bootstrap",
            "spring boot", "kotlin", "android development", "flutter", "system design", "aws",
            "devops", "cyber security", "ethical hacking", "pandas", "numpy", "langchain",
            "generative AI", "MCP servers", "rust", "golang", "recursion", "dynamic programming",
            "graphs", "linked list", "binary search", "OOPs", "pointers", "arrays",
            "sorting algorithms", "REST API", "web scraping", "excel", "power bi", "redux",
            "vue.js", "angular", "php", "laravel", "selenium", "data structures", "trees",
            "heap", "hashing", "sliding window", "backtracking", "jwt authentication",
            "microservices", "prompt engineering", "RAG", "computer networks", "DBMS",
            "operating systems", "competitive programming", "web development", "app development",
            "blockchain", "unity game development", "tkinter", "opencv", "streamlit",
        ],
        "concepts": [
            "useEffect", "closures", "the event loop", "promises", "async await", "virtual DOM",
            "middleware", "decorators", "generators", "list comprehension", "the self keyword",
            "constructors", "inheritance", "polymorphism", "interfaces", "abstract classes",
            "the static keyword", "garbage collection", "time complexity", "space complexity",
            "memoization", "the two pointer method", "foreign keys", "joins", "indexes",
            "normalisation", "the this keyword", "hoisting", "props and state", "context API",
            "git rebase", "docker volumes", "virtual environments", "lambda functions",
            "the main method", "the try except block", "flexbox", "css grid", "media queries",
        ],
        "errors": [
            "ModuleNotFoundError: No module named 'numpy'", "SyntaxError: invalid syntax",
            "IndentationError: unexpected indent", "NullPointerException",
            "segmentation fault (core dumped)", "npm ERR! code ERESOLVE", "CORS error",
            "404 not found", "port 3000 is already in use",
            "'python' is not recognized as an internal or external command",
            "TypeError: Cannot read properties of undefined", "ImportError",
            "permission denied", "connection refused", "IndexError: list index out of range",
            "stack overflow error", "Error: Cannot find module 'express'",
            "'pip' is not recognized", "error: cannot find symbol", "git push rejected",
            "merge conflict", "kernel died", "ECONNREFUSED 127.0.0.1:27017",
            "ReferenceError: x is not defined", "java.lang.ArrayIndexOutOfBoundsException",
            "KeyError: 'name'", "TemplateDoesNotExist", "no such table", "Access denied for user 'root'",
            "Module not found: Can't resolve './App'", "fatal: not a git repository",
            "ValueError: could not convert string to float", "undefined reference to main",
            "Unhandled Runtime Error", "500 internal server error", "token expired",
            "CUDA out of memory", "AttributeError: 'NoneType' object has no attribute",
        ],
        "tools": ["vs code", "pycharm", "node", "npm", "python", "anaconda", "jupyter notebook",
                  "android studio", "intellij", "mysql workbench", "xampp", "docker desktop",
                  "git bash", "postman", "mongodb compass", "eclipse", "turbo c", "codeblocks"],
        "formats": ["video", "playlist", "one shot", "full course", "project", "series",
                    "crash course", "tutorial", "roadmap", "cheat sheet", "notes", "live session",
                    "interview questions video", "mini project", "complete course"],
        "backgrounds": ["BCA", "BCom", "BSc", "mechanical", "civil", "non-IT", "arts", "12th pass",
                        "diploma", "hindi medium", "ECE", "MBA"],
        "goals": ["a job", "placements", "freelancing", "an internship", "FAANG", "a remote job",
                  "web development", "data science", "startups", "GSoC"],
        "companies": ["TCS", "Infosys", "Wipro", "Accenture", "Amazon", "Google", "Microsoft",
                      "Flipkart", "Zoho", "Capgemini", "Cognizant", "Deloitte", "a startup"],
    },
    "math": {
        "channel": "synthetic_math",
        "topics": [
            "linear algebra", "eigenvectors", "the fourier transform", "calculus", "derivatives",
            "integrals", "probability", "bayes theorem", "topology", "group theory",
            "complex numbers", "differential equations", "prime numbers", "matrices",
            "determinants", "taylor series", "neural networks", "convolution",
            "the riemann hypothesis", "vectors", "statistics", "quaternions", "graph theory",
            "number theory", "combinatorics", "the central limit theorem", "information theory",
            "manifolds", "real analysis", "the laplace transform", "markov chains",
            "game theory", "cryptography", "knot theory", "fractals", "set theory",
            "the gamma function", "hilbert spaces", "the zeta function", "transformers",
        ],
        "concepts": [
            "the determinant as area", "the chain rule", "the dot product", "the cross product",
            "the basis change", "the eigenbasis", "the jacobian", "the limit definition",
            "the epsilon delta proof", "the pigeonhole argument", "the induction step",
            "the complex exponential", "the gradient", "the null space", "the rank",
            "the last lemma", "the visual proof", "the inverse matrix", "the winding number",
            "the moment generating function",
        ],
        "errors": [],
        "tools": ["manim", "geogebra", "desmos", "python", "matlab", "wolfram alpha"],
        "formats": ["video", "series", "playlist", "lecture", "visual explanation",
                    "follow up video", "short", "deep dive", "essence of series"],
        "backgrounds": ["high school", "engineering", "economics", "biology", "self taught",
                        "class 12", "physics", "first year"],
        "goals": ["research", "machine learning", "a PhD", "olympiads", "quant jobs", "GATE"],
        "companies": ["a hedge fund", "a research lab", "Google DeepMind", "IISc"],
    },
    "exam": {
        "channel": "synthetic_exam",
        "topics": [
            "kinematics", "rotational motion", "electrostatics", "thermodynamics", "optics",
            "modern physics", "organic chemistry", "mole concept", "chemical bonding",
            "integration", "coordinate geometry", "NLM", "work power energy",
            "current electricity", "magnetism", "waves", "NCERT biology", "genetics",
            "human physiology", "p block", "electrochemistry", "trigonometry", "vectors",
            "ray optics", "semiconductors", "gravitation", "fluid mechanics", "SHM",
            "chemical kinetics", "equilibrium", "GOC", "coordination compounds",
            "permutation combination", "probability", "matrices", "limits", "units and dimensions",
            "plant kingdom", "cell biology", "ecology", "atomic structure", "periodic table",
        ],
        "concepts": [
            "the free body diagram", "the sign convention", "the moment of inertia formula",
            "the hybridisation trick", "the shortcut for torque", "the inductive effect",
            "Gauss law", "the lens formula", "the second law", "the derivation at the end",
            "the relative velocity part", "the resonance concept", "the dimensional formula",
            "the last numerical", "the energy diagram", "Kirchhoff law",
        ],
        "errors": [],
        "tools": ["PW app", "the website", "the test portal", "the batch section", "the notes PDF"],
        "formats": ["one shot", "lecture", "DPP", "PYQ solutions", "revision video",
                    "mock test", "notes", "short notes", "formula sheet", "marathon",
                    "crash course", "batch", "mind map", "live class"],
        "backgrounds": ["dropper", "class 11", "class 12", "hindi medium", "commerce",
                        "average", "weak in maths", "repeater", "CBSE", "state board"],
        "goals": ["JEE Mains", "JEE Advanced", "NEET", "boards", "CUET", "NDA", "BITSAT",
                  "MHT CET", "KVPY", "olympiad"],
        "companies": ["IIT Bombay", "IIT Delhi", "AIIMS", "NIT Trichy", "a government college",
                      "BITS Pilani", "IIIT Hyderabad"],
    },
}

ADDRESS = {
    "coding": ["sir", "bhai", "bhaiya", "harry bhai", "didi", "mam", "maam", "bro", "shradha di",
               "aman bhaiya", "team", "guys", ""],
    "math": ["grant", "sir", "", "", "3b1b", "professor", "man", "dude", "everyone"],
    "exam": ["sir", "alakh sir", "mam", "bhaiya", "didi", "pw team", "sir ji", "maam", ""],
}

PLACES = ["Bihar", "UP", "Delhi", "Mumbai", "Kerala", "Assam", "Nepal", "Bangladesh", "Pakistan",
          "Rajasthan", "Odisha", "Punjab", "Kolkata", "Hyderabad", "Pune", "Gujarat", "Indore",
          "Lucknow", "Patna", "Jaipur", "Bhopal", "Nigeria", "Brazil", "Germany", "Egypt",
          "Kashmir", "Tamil Nadu", "Chennai", "Karachi", "Kathmandu", "Ranchi", "Kota", "the USA"]

LANG_SUBS = ["hindi", "english", "arabic", "bangla", "tamil", "telugu", "marathi", "urdu",
             "spanish", "portuguese"]

EMO_POS = ["❤", "❤️", "🔥", "🙏", "👏", "😍", "🥰", "💯", "🎉", "🫡", "👍", "✨", "🙌", "😊", "💙"]
EMO_SAD = ["😭", "😢", "🥲", "😔", "😩", "😞", "🤯", "😵‍💫", "😓"]
EMO_CONF = ["🤔", "😕", "🧐", "😐", "😶", "🙄"]
EMO_FUN = ["😂", "🤣", "💀", "😆", "😅", "😹", "😜", "🗿", "😎"]
EMO_SPAM = ["💰", "💸", "🤑", "🚀", "📈", "✅", "🔥", "👇", "📲", "💎", "🥀"]

PLATFORMS = ["the app", "PW app", "the website", "my phone", "youtube", "the portal", "my laptop",
             "android", "iphone", "smart tv"]
CLOCK = ["2", "3", "1", "4", "5", "11", "12", "6"]
YEARS = ["2025", "2026", "2027", "2024"]

FIRST_NAMES = ["Rahul", "Priya", "Aman", "Sneha", "Rohit", "Anjali", "Vikram", "Pooja", "Arjun",
               "Neha", "Karan", "Riya", "Sanjay", "Meera", "Imran", "Fatima", "Deepak", "Kavya",
               "Ankit", "Simran", "Suresh", "Divya", "Harsh", "Ayesha", "Nikhil", "Tanvi"]
SURNAMES = ["Sharma", "Verma", "Gupta", "Singh", "Khan", "Patel", "Reddy", "Nair", "Das", "Yadav",
            "Mehta", "Joshi", "Mishra", "Iyer", "Chauhan", "Banerjee"]
SPAM_DOMAINS = ["bit.ly", "tinyurl.com", "t.me", "linktr.ee", "rb.gy", "cutt.ly", "wa.me",
                "earnfast.in", "freecourses.xyz", "modapk.site", "cryptogain.io", "grabdeal.shop"]

# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


class Ctx:
    """Slot sampler bound to one random generator and one domain."""

    def __init__(self, rng: random.Random, domain: str):
        self.r = rng
        self.domain = domain
        self.d = DOMAINS[domain]

    def pick(self, seq):
        return self.r.choice(seq)

    def topic(self):
        return self.pick(self.d["topics"])

    def two_topics(self):
        a, b = self.r.sample(self.d["topics"], 2)
        return a, b

    def concept(self):
        return self.pick(self.d["concepts"])

    def fmt(self):
        return self.pick(self.d["formats"])

    def addr(self):
        return self.pick(ADDRESS[self.domain])

    def goal(self):
        return self.pick(self.d["goals"])

    def bg(self):
        return self.pick(self.d["backgrounds"])

    def company(self):
        return self.pick(self.d["companies"])

    def tool(self):
        return self.pick(self.d["tools"])

    def error(self):
        if self.d["errors"]:
            return self.pick(self.d["errors"])
        return self.pick(DOMAINS["coding"]["errors"])

    def time(self):
        if self.r.random() < 0.15:
            return f"{self.r.randint(1, 2)}:{self.r.randint(0, 59):02d}:{self.r.randint(0, 59):02d}"
        return f"{self.r.randint(0, 59)}:{self.r.randint(0, 59):02d}"

    def place(self):
        return self.pick(PLACES)

    def year(self):
        return self.pick(YEARS)

    def emo(self, kind=EMO_POS, lo=1, hi=3):
        e = self.pick(kind)
        n = self.r.randint(lo, hi)
        if self.r.random() < 0.6:
            return e * n
        return "".join(self.pick(kind) for _ in range(n))

    def num(self, lo, hi):
        return str(self.r.randint(lo, hi))

    def name(self):
        return f"{self.pick(FIRST_NAMES)} {self.pick(SURNAMES)}"

    def handle(self):
        base = self.pick(FIRST_NAMES).lower() + self.pick(["_", ".", "", "__"]) + \
            self.pick(["official", "earns", "trader", "fx", "pro", "help", "coder", "x", ""]) + \
            self.num(1, 999)
        return base

    def url(self):
        dom = self.pick(SPAM_DOMAINS)
        slug = "".join(self.r.choice("abcdefghijkmnpqrstuvwxyz23456789") for _ in range(self.r.randint(4, 8)))
        return f"https://{dom}/{slug}"

    def phone(self):
        return "+91 " + str(self.r.randint(6, 9)) + "".join(str(self.r.randint(0, 9)) for _ in range(9))


def addr_prefix(c: Ctx, text: str) -> str:
    a = c.addr()
    return f"{a} {text}" if a else text


# --------------------------------------------------------------------------
# Templates — each returns (text, language)
# --------------------------------------------------------------------------
# Templates are plain functions so they can use arbitrary logic; each is
# registered with a template_id for traceability.

TEMPLATES = {c: [] for c in CLASSES}


def template(cls, tid):
    def deco(fn):
        TEMPLATES[cls].append((tid, fn))
        return fn
    return deco


# ---------------- Question: answerable information / advice --------------

@template("Question", "q_diff")
def _(c):
    a, b = c.two_topics()
    return c.pick([
        f"what is the difference between {a} and {b}?",
        f"{a} vs {b}, which one should i learn first?",
        f"{c.addr()} {a} aur {b} me kya difference hai?",
        f"is {a} better than {b} for {c.goal()}?",
        f"{a} ya {b}, {c.goal()} ke liye kaunsa better hai?",
    ]), "mixed"


@template("Question", "q_scope")
def _(c):
    t = c.topic()
    return c.pick([
        f"is {t} enough to get {c.goal()} in {c.year()}?",
        f"{c.addr()} kya {t} se {c.goal()} mil sakta hai?",
        f"does {t} still have scope in {c.year()}?",
        f"{t} ka scope kya hai future me?",
        f"is it worth learning {t} in {c.year()} or is it dead?",
        f"{t} seekh ke kitna package mil sakta hai?",
    ]), "mixed"


@template("Question", "q_exam")
def _(c):
    t = c.topic()
    g = c.goal()
    return c.pick([
        f"{c.addr()} ye {g} me aayega kya?",
        f"{c.addr()} kya {t} {g} ke syllabus me hai?",
        f"is this enough for {g}?",
        f"is {t} important for {g} {c.year()}?",
        f"{c.addr()} {t} ka weightage kitna hai {g} me?",
        f"how many questions come from {t} in {g}?",
        f"kya ye lecture {g} ke liye kaafi hai?",
    ]), "mixed"


@template("Question", "q_time")
def _(c):
    t = c.topic()
    return c.pick([
        f"how long does it take to learn {t}?",
        f"{t} sikhne me kitna time lagega?",
        f"how many hours a day should i give to {t}?",
        f"{c.addr()} kitne din me {t} complete ho jayega?",
        f"can i finish {t} in {c.num(1, 6)} months?",
    ]), "mixed"


@template("Question", "q_next")
def _(c):
    t = c.topic()
    return c.pick([
        f"what should i learn after {t}?",
        f"{c.addr()} {t} ke baad kya padhna chahiye?",
        f"after finishing {t} where should i go next?",
        f"is there any prerequisite for {t}?",
        f"do i need to know {c.topic()} before starting {t}?",
        f"{t} start karne se pehle kya aana chahiye?",
    ]), "mixed"


@template("Question", "q_background")
def _(c):
    t = c.topic()
    b = c.bg()
    return c.pick([
        f"can a {b} student learn {t}?",
        f"{c.addr()} main {b} se hu, kya main {t} kar sakta hu?",
        f"i am from {b} background, is {t} possible for me?",
        f"{b} wale {c.goal()} ke liye eligible hai kya?",
        f"i am a {b} student, should i go for {t} or {c.topic()}?",
        f"{c.addr()} {b} ke baad kya karu?",
    ]), "mixed"


@template("Question", "q_class")
def _(c):
    n = c.pick(["8", "9", "10", "11", "12", "11th", "12th", "first year", "second year", "third year"])
    return c.pick([
        f"i am in class {n}, where should i start {c.topic()}?",
        f"{c.addr()} main class {n} me hu, abhi se {c.topic()} start karu ya nahi?",
        f"i'm in {n}, is it too early to learn {c.topic()}?",
        f"should a class {n} student focus on boards or {c.goal()}?",
        f"{n} me hu, kaise start karu?",
    ]), "mixed"


@template("Question", "q_concept")
def _(c):
    k = c.concept()
    return c.pick([
        f"why do we use {k} here?",
        f"does {k} also work when the input is negative?",
        f"what happens to {k} if we change the order?",
        f"is {k} the same thing as {c.concept()}?",
        f"{c.addr()} {k} ka real life use kya hai?",
        f"where is {k} actually used in practice?",
        f"why is {k} needed at all?",
        f"how is {k} related to {c.topic()}?",
        f"would {k} still hold in higher dimensions?",
    ]), "en"


@template("Question", "q_setup")
def _(c):
    return c.pick([
        f"which {c.pick(['editor', 'laptop', 'tablet', 'mic', 'software', 'pen tablet', 'theme', 'font'])} are you using?",
        f"what software do you use to make these animations?",
        f"{c.addr()} aap konsa {c.pick(['laptop', 'editor', 'tablet', 'mic'])} use karte ho?",
        f"which laptop is best for {c.topic()} under {c.num(30, 90)}k?",
        f"is {c.num(8, 16)}gb ram enough for {c.topic()}?",
        f"does this work on {c.pick(['mac', 'windows', 'linux', 'ubuntu', 'chromebook', 'mobile'])}?",
        f"can i do this on my phone?",
    ]), "mixed"


@template("Question", "q_course")
def _(c):
    return c.pick([
        f"{c.addr()} batch ki fees kitni hai?",
        f"will we get a certificate after this course?",
        f"is this playlist still relevant in {c.year()}?",
        f"is this course free?",
        f"when will part {c.num(2, 9)} come?",
        f"how many videos are there in this playlist?",
        f"is this course in hindi or english?",
        f"{c.addr()} ye course kab tak chalega?",
    ]), "mixed"


@template("Question", "q_resource")
def _(c):
    t = c.topic()
    return c.pick([
        f"which book should i follow for {t}?",
        f"where can i practice {t} questions?",
        f"any good website to practice {t}?",
        f"{c.addr()} {t} ke liye best book konsi hai?",
        f"which platform is best for {t} practice, {c.pick(['leetcode', 'gfg', 'hackerrank', 'codeforces', 'brilliant', 'HC Verma', 'DC Pandey', 'NCERT'])} or something else?",
        f"should i make notes while watching or just practice?",
    ]), "mixed"


@template("Question", "q_career")
def _(c):
    a, b = c.pick([("CSE", "ECE"), ("BCA", "BTech"), ("IIIT", "NIT"), ("AI ML", "core CSE"),
                   ("MCA", "job"), ("MBA", "MTech"), ("drop", "private college"),
                   ("frontend", "backend"), ("data analyst", "data scientist"),
                   ("government job", "private job"), ("GATE", "placement")])
    return c.pick([
        f"should i choose {a} or {b}?",
        f"{c.addr()} {a} lu ya {b}?",
        f"{a} or {b}, which has better future?",
        f"confused between {a} and {b}, what would you suggest?",
        f"{a} vs {b} kya better rahega mere liye?",
    ]), "mixed"


@template("Question", "q_hindi")
def _(c):
    t = c.topic()
    g = c.goal()
    return c.pick([
        f"क्या {t} {g} के लिए जरूरी है?",
        f"सर {t} कितने दिन में सीख सकते हैं?",
        f"{t} के बाद क्या पढ़ना चाहिए?",
        f"क्या यह कोर्स फ्री है?",
        f"सर कौन सी किताब पढ़नी चाहिए {t} के लिए?",
        f"क्या हिंदी मीडियम वाले {t} सीख सकते हैं?",
        f"{t} और {c.topic()} में क्या अंतर है?",
    ]), "hi"


@template("Question", "q_praise_mix")
def _(c):
    # Praise + a real question -> Question (tie-break).
    p = c.pick(["great video!", "amazing explanation", "loved it", "very helpful",
                "bahut accha samjhaya", "thanks for this", "nice lecture"])
    q = c.pick([f"is {c.topic()} needed before this?", f"does this cover {c.topic()} too?",
                f"which {c.pick(['book', 'course', 'playlist'])} should i follow next?",
                f"kya ye {c.goal()} ke liye kaafi hai?", f"how did you learn {c.topic()} yourself?",
                f"will this help in {c.goal()}?"])
    return f"{p}{c.pick([' ', ', ', '. ', '... '])}{q}".replace("!,", "!").replace("!.", "!"), "mixed"


@template("Question", "q_misc")
def _(c):
    return c.pick([
        f"how do you stay consistent while learning?",
        f"how did you get your first job?",
        f"should i learn {c.topic()} from documentation or videos?",
        f"what is the salary of a {c.topic()} developer in india?",
        f"is maths required for {c.topic()}?",
        f"how to balance college and {c.goal()} preparation?",
        f"{c.addr()} aapne kaise padhai ki thi {c.goal()} ke liye?",
        f"what is the best age to start {c.topic()}?",
        f"is AI going to replace {c.topic()} jobs?",
        f"how is this different from what they teach in college?",
    ]), "en"


# ---------------- Request: make / upload / share content ------------------

@template("Request", "r_make")
def _(c):
    t = c.topic()
    f = c.fmt()
    return c.pick([
        f"please make a {f} on {t}",
        f"{c.addr()} {t} ka {f} bana do please",
        f"{c.addr()} {t} par ek {f} laao",
        f"make a complete {f} on {t} from basic to advanced",
        f"please bring a {t} {f}",
        f"{t} ka bhi {f} le aao {c.addr()}",
        f"waiting for the {t} {f}",
        f"we need a {f} on {t}",
        f"{c.addr()} {t} pe video banao please",
        f"plz {t} ki {f} start karo",
    ]), "mixed"


@template("Request", "r_can_you")
def _(c):
    t = c.topic()
    return c.pick([
        f"can you make a {c.fmt()} on {t}?",
        f"could you please do a video on {t}?",
        f"can you please explain {t} in the next video?",
        f"{c.addr()} kya aap {t} pe video bana sakte ho?",
        f"will you make a {c.fmt()} on {t} please?",
        f"can you cover {c.concept()} in detail sometime?",
    ]), "mixed"


@template("Request", "r_next")
def _(c):
    t = c.topic()
    return c.pick([
        f"next video on {t} please",
        f"next {t} please {c.addr()}",
        f"please cover {t} in the next lecture",
        f"next lecture me {t} karwa do",
        f"please continue this series with {t}",
        f"part {c.num(2, 6)} please, and include {t}",
        f"please upload the next part soon",
        f"{c.addr()} next video jaldi upload karo",
    ]), "mixed"


@template("Request", "r_share")
def _(c):
    return c.pick([
        f"{c.addr()} notes ka link de do",
        f"please share the notes of {c.topic()}",
        f"please share the source code",
        f"github link please",
        f"pdf of this lecture please",
        f"{c.addr()} iska pdf description me daal do",
        f"please upload the slides",
        f"please share the code in the description",
        f"{c.topic()} ke notes share karo please",
        f"please give the dataset link",
        f"please provide the {c.pick(['DPP', 'formula sheet', 'cheat sheet', 'assignment', 'practice questions'])} for {c.topic()}",
    ]), "mixed"


@template("Request", "r_subtitles")
def _(c):
    lang = c.pick(LANG_SUBS)
    return c.pick([
        f"please add {lang} subtitles",
        f"{lang} subtitles please",
        f"{c.addr()} {lang} me bhi video banao",
        f"please make this in {lang} also",
        f"please explain in hindi also",
        f"{c.addr()} hindi me bhi {c.topic()} padhao",
        f"please dub this video in {lang}",
    ]), "mixed"


@template("Request", "r_compare")
def _(c):
    a, b = c.two_topics()
    return c.pick([
        f"please make a comparison video of {a} vs {b}",
        f"{a} vs {b} pe video banao",
        f"a video comparing {a} and {b} would be really helpful please",
    ]), "en"


@template("Request", "r_course")
def _(c):
    t = c.topic()
    return c.pick([
        f"{c.addr()} {t} ka {c.pick(['beginner', 'advanced', 'full', 'free', 'complete'])} course start karo",
        f"please launch a {t} course",
        f"please start a {t} batch for {c.bg()} students",
        f"{c.addr()} {c.goal()} ke liye alag batch lao",
        f"please make a roadmap for {c.goal()}",
        f"please start a {c.pick(['bootcamp', 'cohort', 'live batch', 'weekend batch'])} on {t}",
    ]), "mixed"


@template("Request", "r_praise_mix")
def _(c):
    # Praise + request -> Request (tie-break).
    p = c.pick(["amazing video!", "great explanation", "love your content", "best teacher",
                "bahut badhiya", "thank you so much", "superb", "congrats on 10M"])
    r = c.pick([f"please make one on {c.topic()}", f"{c.topic()} pe bhi video banao",
                f"next do {c.topic()} please", f"please make a {c.fmt()} on {c.topic()} too",
                f"{c.topic()} ka {c.fmt()} bhi le aao"])
    e = c.emo() if c.r.random() < 0.5 else ""
    sep = c.pick([" ", ", ", ". ", " ... "]) if not e else f" {e} "
    return f"{p}{sep}{r}".replace("!,", "!").replace("!.", "!"), "mixed"


@template("Request", "r_wish")
def _(c):
    t = c.topic()
    return c.pick([
        f"would love to see a video about {t}",
        f"a series on {t} would be great",
        f"i wish you would make something on {t}",
        f"it would be awesome if you covered {t} someday",
        f"really hoping for a {t} {c.fmt()} next",
        f"{t} pe ek video aa jaye to maza aa jaye",
    ]), "en"


@template("Request", "r_project")
def _(c):
    t = c.topic()
    return c.pick([
        f"please make a project using {t} and {c.topic()}",
        f"{c.addr()} {t} ka ek real world project banao",
        f"please build a {c.pick(['chat app', 'ecommerce site', 'portfolio', 'todo app', 'clone of netflix', 'resume builder', 'AI agent', 'quiz app'])} with {t}",
        f"make a full stack project video please",
        f"please do a project based {t} series",
    ]), "mixed"


@template("Request", "r_hindi")
def _(c):
    t = c.topic()
    return c.pick([
        f"कृपया {t} पर वीडियो बनाइए",
        f"सर {t} की प्लेलिस्ट बनाइए",
        f"{t} के नोट्स शेयर कीजिए",
        f"सर {t} का वन शॉट लाइए",
        f"अगला वीडियो {t} पर बनाइए प्लीज़",
        f"हिंदी में {t} का कोर्स लाइए",
    ]), "hi"


@template("Request", "r_short")
def _(c):
    t = c.topic()
    return c.pick([
        f"{t} please",
        f"{t} next 🙏",
        f"{t} {c.fmt()} plz",
        f"{c.addr()} {t} karao",
        f"{t} bhi karwa do",
        f"more videos on {t} please",
    ]), "mixed"


# ---------------- Confusion/Problem: stuck, error, broken, lost -----------

@template("Confusion/Problem", "c_error")
def _(c):
    e = c.error()
    t = c.time()
    return c.pick([
        f"getting {e} at {t}",
        f"{c.addr()} mera code run nahi ho raha, {e} aa raha hai",
        f"i am getting {e}, what to do",
        f"{e} aa raha hai {t} pe, help",
        f"it shows {e} when i run it",
        f"same code but i get {e}",
        f"{e} 😭 please help",
        f"{c.addr()} {e} kaise fix kare?",
        f"why am i getting {e} even after copying your code?",
        f"stuck on {e} for {c.num(2, 5)} hours now",
    ]), "mixed"


@template("Confusion/Problem", "c_not_understood")
def _(c):
    t = c.time()
    k = c.concept()
    return c.pick([
        f"i didn't understand the part at {t}",
        f"samajh nahi aaya {k} wala part {c.emo(EMO_CONF, 1, 2)}",
        f"{k} samajh me nahi aaya {c.addr()}",
        f"lost after {t}, can someone explain?",
        f"i'm still confused about {k}",
        f"{k} ka concept clear nahi hua",
        f"i got lost at {k}",
        f"what just happened at {t}? i am totally lost",
        f"couldn't follow {k}, too fast for me",
        f"{t} ke baad kuch samajh nahi aaya",
        f"can someone explain {k} again, i don't get it",
        f"i watched it {c.num(2, 5)} times and still don't get {k}",
    ]), "mixed"


@template("Confusion/Problem", "c_setup")
def _(c):
    tool = c.tool()
    os_ = c.pick(["windows", "windows 11", "mac", "ubuntu", "my laptop", "my pc", "mobile"])
    return c.pick([
        f"{tool} install nahi ho raha {os_} pe",
        f"{tool} is not opening on {os_}",
        f"i can't install {tool}, it keeps failing",
        f"{tool} me terminal show nahi ho raha",
        f"my {tool} looks different from yours and the option is missing",
        f"{tool} crashes every time i open it",
        f"i followed every step but {tool} still doesn't work",
        f"{tool} setup karte time error aa raha hai",
        f"{c.addr()} mere {os_} pe {tool} nahi chal raha",
    ]), "mixed"


@template("Confusion/Problem", "c_output")
def _(c):
    t = c.time()
    return c.pick([
        f"my output is different from yours at {t}",
        f"i did exactly the same but my answer is coming {c.num(2, 99)} instead of {c.num(2, 99)}",
        f"output nahi aa raha, blank screen aa rahi hai",
        f"my page is not updating even after saving",
        f"nothing prints when i run it, no error either",
        f"the {c.pick(['button', 'image', 'css', 'navbar', 'form', 'api', 'loop', 'graph'])} is not working for me",
        f"my answer doesn't match at {t}, where did i go wrong?",
        f"mera answer galat aa raha hai {t} wale question me",
    ]), "mixed"


@template("Confusion/Problem", "c_access")
def _(c):
    return c.pick([
        f"notes ka link kaam nahi kar raha",
        f"the notes link is not working",
        f"payment ho gaya but batch show nahi ho raha",
        f"{c.addr()} i paid for the batch but it is not showing in my account",
        f"link in description is broken",
        f"{c.pick(PLATFORMS)} pe lecture play nahi ho raha",
        f"login nahi ho raha {c.pick(PLATFORMS)} pe",
        f"video is stuck at {c.time()} for me",
        f"audio is out of sync after {c.time()}",
    ]), "mixed"


@template("Confusion/Problem", "c_praise_mix")
def _(c):
    # Praise + stuck -> Confusion/Problem (tie-break).
    p = c.pick(["great video but", "nice explanation but", "loved it, but", "amazing course but",
                "thanks sir but", "very good lecture lekin"])
    s = c.pick([f"my code throws {c.error()}", f"i didn't get {c.concept()}",
                f"{c.concept()} samajh nahi aaya", f"{c.tool()} is not working for me",
                f"i am stuck at {c.time()}", f"the link is not opening"])
    return f"{p} {s}", "mixed"


@template("Confusion/Problem", "c_frustrated")
def _(c):
    return c.pick([
        f"tried for {c.num(2, 6)} hours, still stuck {c.emo(EMO_SAD, 1, 3)}",
        f"kuch samajh nahi aa raha, bahut confusing hai",
        f"i feel so lost in {c.topic()}",
        f"why is {c.topic()} so confusing for me",
        f"i keep making the same mistake in {c.topic()} and can't figure out why",
        f"{c.topic()} mere dimaag ke upar se gaya {c.emo(EMO_SAD, 1, 2)}",
    ]), "mixed"


@template("Confusion/Problem", "c_step")
def _(c):
    step = c.pick(["step 3", "the second step", "the last step", "the installation step",
                   "the import line", "the database connection", "the deployment part",
                   "the npm install", "the env setup", "the substitution", "the integration by parts",
                   "the final simplification", "the free body diagram", "the derivation"])
    return c.pick([
        f"stuck at {step}",
        f"{step} pe atak gaya hu",
        f"i am not able to do {step}",
        f"{step} kaise kiya? mera nahi ho raha",
        f"{step} is not working for me",
        f"can't get past {step}",
    ]), "mixed"


@template("Confusion/Problem", "c_hindi")
def _(c):
    k = c.concept()
    return c.pick([
        f"कुछ समझ नहीं आया {c.emo(EMO_SAD, 1, 2)}",
        f"{k} समझ में नहीं आया सर",
        f"नोट्स डाउनलोड नहीं हो रहे",
        f"मेरा कोड नहीं चल रहा, error आ रहा है",
        f"{c.time()} के बाद समझ नहीं आया",
        f"लिंक काम नहीं कर रहा",
        f"सर पेमेंट हो गया पर बैच नहीं दिख रहा",
    ]), "hi"


# ---------------- Praise ---------------------------------------------------

@template("Praise", "p_best")
def _(c):
    t = c.topic()
    return c.pick([
        f"best explanation of {t} on youtube {c.emo()}",
        f"the best {t} video i have ever watched",
        f"nobody explains {t} like you {c.emo()}",
        f"{t} finally makes sense, thank you",
        f"finally understood {t} after {c.num(2, 5)} years",
        f"this is the best {t} {c.fmt()} on the internet",
        f"{t} itna easy kabhi nahi laga tha {c.emo()}",
        f"better than my college professor's {t} class",
    ]), "mixed"


@template("Praise", "p_thanks")
def _(c):
    return c.pick([
        f"thank you {c.addr()} {c.emo()}",
        f"thanks a lot {c.emo()}",
        f"thank you so much for this free content",
        f"thanks {c.addr()}, this helped me a lot",
        f"shukriya {c.addr()} {c.emo()}",
        f"dhanyavad {c.addr()}",
        f"thank you for making this free",
        f"thank you for your hard work {c.emo()}",
        f"thanks for explaining it so simply",
    ]), "mixed"


@template("Praise", "p_success")
def _(c):
    ach = c.pick([
        f"placement ho gaya {c.company()} me", f"got my first internship at {c.company()}",
        f"got a job at {c.company()}", f"cleared {c.goal()}",
        f"got {c.num(85, 99)} percentile in {c.goal()}", f"got into {c.company()}",
        f"scored {c.num(85, 99)}% in boards", f"got selected in {c.company()}",
        f"built my first project", f"got {c.num(4, 30)} LPA offer",
    ])
    return c.pick([
        f"thank you {c.addr()}, {ach} {c.emo()}",
        f"because of you i {ach}",
        f"{c.addr()} aapki wajah se {ach} {c.emo()}",
        f"started from your videos and today {ach}",
        f"{ach}! all credit goes to this channel {c.emo()}",
    ]), "mixed"


@template("Praise", "p_emoji")
def _(c):
    k = c.r.random()
    if k < 0.5:
        return c.emo(EMO_POS, 2, 6), "emoji"
    return "".join(c.pick(EMO_POS) for _ in range(c.r.randint(2, 6))), "emoji"


@template("Praise", "p_short")
def _(c):
    return c.pick([
        "masterpiece", "superb", "awesome", "brilliant", "nice",
        "very nice", "excellent", "great", "amazing", "mind blowing", "outstanding",
        "zabardast", "mast", "badhiya", "bahut badhiya", "op", "god level", "next level",
        "beautiful", "elegant", "wow", "fantastic", "top notch", "underrated", "gem",
    ]) + c.pick(["", "", " " + c.emo(), "!", "!!", f" {c.addr()}"]), "mixed"


@template("Praise", "p_teaching")
def _(c):
    return c.pick([
        f"your teaching style is amazing {c.emo()}",
        f"you explain so calmly, love it",
        f"aap bahut accha padhate ho {c.addr()}",
        f"{c.addr()} aapka samjhane ka tarika best hai",
        f"the way you explain {c.concept()} is beautiful",
        f"the animations in this video are stunning",
        f"such a clear and visual explanation",
        f"this video is a masterpiece, the visuals are perfect",
        f"you have a really nice voice",
        f"your energy is contagious {c.emo()}",
        f"such a calm and clear teacher",
        f"best teacher on youtube {c.emo()}",
    ]), "mixed"


@template("Praise", "p_channel")
def _(c):
    return c.pick([
        "this channel deserves more subscribers",
        "this deserves millions of views",
        f"congratulations on {c.num(1, 20)}M subscribers {c.emo()}",
        f"congrats {c.addr()} {c.emo()}",
        f"congratulations {c.addr()} for {c.num(1, 20)} million {c.emo()}",
        f"keep it up {c.addr()}, love from {c.place()} {c.emo()}",
        f"love from {c.place()} {c.emo()}",
        f"respect from {c.place()} {c.emo()}",
        "keep growing, you deserve it",
        "keep doing what you do",
        "more power to you",
        f"been following since {c.pick(['2017', '2018', '2019', '2020', '2021'])}, still the best",
        "the internet needs more people like you",
    ]), "mixed"


@template("Praise", "p_feeling")
def _(c):
    return c.pick([
        "watched it twice, still mind blown",
        "my brain just expanded",
        "i got goosebumps at the end",
        "this video gave me so much motivation",
        f"i was scared of {c.topic()} but now i love it",
        "this made my day",
        f"{c.topic()} ab easy lag raha hai {c.emo()}",
        "i wish my teachers taught like this",
        "pure gold content",
        "worth every minute",
    ]), "en"


@template("Praise", "p_hindi")
def _(c):
    return c.pick([
        f"बहुत अच्छा पढ़ाते हैं सर {c.emo()}",
        f"बहुत बढ़िया वीडियो {c.emo()}",
        "धन्यवाद सर",
        f"आप सबसे अच्छे टीचर हैं {c.emo()}",
        f"{c.topic()} अब समझ में आ गया, धन्यवाद",
        "शानदार समझाया आपने",
        f"दिल से धन्यवाद {c.emo()}",
    ]), "hi"


# ---------------- Spam -----------------------------------------------------

@template("Spam", "s_earn")
def _(c):
    amt = c.pick(["₹5000", "₹3000", "₹10000", "$500", "$200", "₹2500", "₹8000", "$1000"])
    return c.pick([
        f"earn {amt} daily from home, DM me on telegram @{c.handle()} {c.emo(EMO_SPAM)}",
        f"work from home job, earn {amt} per day, whatsapp {c.phone()}",
        f"part time job available earn {amt} daily, contact {c.phone()}",
        f"guys i made {amt} in one day with this trick {c.url()}",
        f"ghar baithe {amt} kamao roz, telegram join karo {c.url()}",
        f"easy earning app {amt} daily {c.url()} {c.emo(EMO_SPAM)}",
    ]), "mixed"


@template("Spam", "s_crypto")
def _(c):
    n = c.name()
    return c.pick([
        f"I started investing with {n} and made ${c.num(5, 90)},000 in {c.num(2, 8)} weeks, she is the best",
        f"thanks to Mrs {n} my portfolio grew {c.num(3, 20)}x, contact her on telegram @{c.handle()}",
        f"crypto signals group 100% profit, join {c.url()} {c.emo(EMO_SPAM)}",
        f"bitcoin doubling scheme, send and receive 2x, {c.url()}",
        f"if you are not investing in forex you are missing out, message @{c.handle()}",
        f"{n} is a legit trader, highly recommended {c.emo(EMO_SPAM)} @{c.handle()}",
    ]), "en"


@template("Spam", "s_selfpromo")
def _(c):
    t = c.topic()
    return c.pick([
        f"check out my channel for {t} content {c.emo(EMO_SPAM, 1, 2)}",
        f"subscribe to my channel guys, i also teach {t}",
        f"i made the same project, check it out on my channel",
        f"please visit my channel and support {c.emo(EMO_POS, 1, 2)}",
        f"new {t} video on my channel, go watch {c.emo(EMO_SPAM, 1, 2)}",
        f"anyone who subscribes to me i will sub back",
        f"sub to me i am {c.num(10, 990)} subs away from {c.pick(['1k', '500', '100', '10k'])} {c.emo(EMO_SAD, 1, 2)}",
        f"my channel me {t} ke videos hai, please dekho",
        f"follow me on instagram @{c.handle()} for daily {t} tips",
    ]), "mixed"


@template("Spam", "s_links")
def _(c):
    t = c.topic()
    return c.pick([
        f"{t} full course at 90% off {c.url()}",
        f"all premium courses free {c.url()}",
        f"free udemy {t} course coupon {c.url()}",
        f"download all paid {t} notes here {c.url()}",
        f"best {t} course link: {c.url()}",
        f"use my referral code {c.handle().upper()} for {c.num(30, 80)}% off {c.url()}",
        f"buy cheap followers and likes {c.url()}",
    ]), "mixed"


@template("Spam", "s_piracy")
def _(c):
    return c.pick([
        f"all paid batches free, join telegram {c.url()}",
        f"leaked notes of {c.goal()} batch available, dm @{c.handle()}",
        f"premium app mod apk download {c.url()}",
        f"paid course free me chahiye to telegram join karo @{c.handle()}",
        f"{c.goal()} {c.year()} paper leak, join {c.url()}",
        f"whatsapp {c.phone()} for paid notes free {c.emo(EMO_SPAM)}",
    ]), "mixed"


@template("Spam", "s_agency")
def _(c):
    return c.pick([
        f"Hey, I help creators grow their channel with SEO and thumbnails, email me at {c.handle()}@gmail.com",
        f"I can edit your videos at a low price, contact {c.handle()}@gmail.com",
        f"hi i run a marketing agency, we can get you {c.num(10, 500)}k views, dm @{c.handle()}",
        f"need a thumbnail designer? check my portfolio {c.url()}",
    ]), "en"


@template("Spam", "s_chain")
def _(c):
    return c.pick([
        "if you don't share this with 10 people you will fail your exam",
        f"like this comment in {c.num(5, 30)} seconds and you will crack {c.goal()}",
        "copy paste this to 5 videos or bad luck for 7 years",
        f"read my name if you want free {c.topic()} course",
        "dont read my profile picture",
    ]), "mixed"


@template("Spam", "s_bot")
def _(c):
    # Bot-like gibberish / flooding text. Kept short and varied.
    base = c.pick(["hiiii", "yyyyy", "qwertyuiop", "asdfgh", "hlo hlo hlo", "ok ok ok ok",
                   "test test", "pls see my profile", "👇👇👇 my profile", "xyz xyz xyz"])
    return f"{base} {c.emo(EMO_SPAM, 1, 4)} {c.url() if c.r.random() < 0.5 else '@' + c.handle()}", "mixed"


@template("Spam", "s_hindi")
def _(c):
    return c.pick([
        f"घर बैठे ₹{c.num(2, 9)}000 रोज कमाएं, टेलीग्राम पर जुड़ें {c.url()}",
        f"सभी पेड कोर्स फ्री में पाएं {c.url()}",
        f"मेरे चैनल को सब्सक्राइब करें {c.emo(EMO_SPAM, 1, 2)}",
        f"पैसे कमाने का आसान तरीका, व्हाट्सएप करें {c.phone()}",
    ]), "hi"


# ---------------- Other ----------------------------------------------------

@template("Other", "o_attendance")
def _(c):
    return c.pick([
        f"day {c.num(1, 120)} done ✅",
        f"day {c.num(1, 120)} completed",
        f"lecture {c.num(1, 80)} done",
        "present sir",
        f"present {c.addr()}",
        f"{c.num(1, 100)}/100 done",
        f"completed on {c.pick(['monday', 'sunday', 'friday', 'diwali'])} night",
        "attendance lagao",
        f"video {c.num(2, 50)} of the playlist ✅",
    ]), "mixed"


@template("Other", "o_timestamp")
def _(c):
    t = c.time()
    return c.pick([
        t, f"{t} 😂", f"{t} 💀", f"{t} lol", f"{t} {c.emo(EMO_FUN, 1, 2)}",
        f"{t} the expression 😂", f"{t} {c.addr()} ka reaction",
    ]), "mixed"


@template("Other", "o_watching")
def _(c):
    return c.pick([
        f"who is watching this in {c.year()}?",
        f"anyone watching in {c.year()}?",
        f"anyone else watching at {c.pick(CLOCK)} am?",
        f"who is here before 1M views?",
        f"watching this the night before my exam {c.emo(EMO_FUN, 1, 2)}",
        f"anyone from {c.place()}?",
        f"anyone from {c.company()} here?",
        "me watching this instead of studying 😂",
        "first comment",
        "early gang",
        f"watching this at {c.num(1, 2)}x speed",
        f"here after {c.pick(['the shorts', 'instagram reel', 'my friend sent this', 'my teacher recommended'])}",
    ]), "mixed"


@template("Other", "o_joke")
def _(c):
    x = c.pick(["area 51", "nuclear plant", "time machine", "iron man suit", "my crush's heart",
                "NASA", "the pentagon", "my school website", "my neighbour's wifi",
                "NTA website", "my ex's instagram", "the moon"])
    return c.pick([
        f"{x} next 😂",
        f"next video: how to hack {x} {c.emo(EMO_FUN, 1, 2)}",
        f"now i can build {x} 😎",
        f"me after this video: i am ready to build {x}",
        f"mom said i can become an engineer after this {c.emo(EMO_FUN, 1, 2)}",
        f"{c.addr()} ki smile {c.emo(EMO_FUN, 1, 2)}",
        f"the {c.pick(['cat', 'pi creature', 'plant', 'clock', 'dog', 'whiteboard'])} in the background {c.emo(EMO_FUN, 1, 2)}",
        f"bhai {c.topic()} padh ke {c.pick(['girlfriend', 'naukri', 'shaadi', 'sukoon'])} milega kya 😂",
    ]), "mixed"


@template("Other", "o_correction")
def _(c):
    a, b = c.two_topics()
    return c.pick([
        f"{c.time()} it's {a} not {b}",
        f"small mistake at {c.time()}, it should be {c.num(2, 99)} not {c.num(2, 99)}",
        f"{c.time()} there is a typo on the slide",
        "this explanation was too slow",
        "the audio quality could be better",
        f"i think {a} is overrated",
        f"{a} is dead, nobody uses it now",
        "too much talking, less content",
        "the background music is distracting",
        "i disagree with the point at the end",
    ]), "en"


@template("Other", "o_meta")
def _(c):
    return c.pick([
        f"{c.addr()} pin my comment please",
        f"{c.addr()} reply karo please",
        "please reply sir, i am your biggest fan",
        f"need job {c.addr()}",
        f"{c.addr()} job dila do please",
        "heart my comment please",
        "if you read this, have a nice day",
        "comment for the algorithm",
        "commenting for better reach",
        "bookmark",
        f"note to self: rewatch {c.time()}",
    ]), "mixed"


@template("Other", "o_remark")
def _(c):
    t = c.topic()
    return c.pick([
        f"{t} is used everywhere these days",
        f"my teacher taught {t} today and now this video",
        f"i used {t} in my internship last year",
        f"interesting, my college uses {c.topic()} instead",
        f"so {t} is basically just {c.concept()}",
        f"exam {c.pick(['kal', 'parso', 'next week', 'tomorrow', 'aaj'])} hai aur main yaha hu",
        "hmm",
        "ok",
        "lol",
        "interesting",
        "noted",
        f"{c.place()} se hu main",
        f"i am from {c.place()}",
        f"{c.num(100, 999)} likes and i will start {t} today",
        f"my {c.pick(['dad', 'mom', 'brother', 'sister', 'friend'])} is also learning {t}",
        f"it was raining when i watched this",
    ]), "mixed"


@template("Other", "o_math_idea")
def _(c):
    k = c.concept()
    return c.pick([
        f"my approach: i used {k} and got the same result in a longer way",
        f"another way to see it is through {k}",
        f"fun fact: {k} also shows up in {c.topic()}",
        f"this reminds me of {k} from {c.topic()}",
        f"i noticed {k} works here because everything is symmetric",
        f"for anyone curious, {k} generalises to {c.topic()} too",
    ]), "en"


@template("Other", "o_hindi")
def _(c):
    return c.pick([
        f"दिन {c.num(1, 90)} पूरा ✅",
        f"कौन {c.year()} में देख रहा है?",
        "हाजिरी लगाओ",
        f"मैं {c.place()} से हूँ",
        "कल परीक्षा है और मैं यहाँ हूँ 😂",
        "पहला कमेंट",
    ]), "hi"


@template("Other", "o_emoji_fun")
def _(c):
    return c.emo(EMO_FUN, 2, 5), "emoji"


# --------------------------------------------------------------------------
# Surface noise — makes templates look like real typed comments
# --------------------------------------------------------------------------

SHORTHAND = [
    (r"\bplease\b", ["pls", "plz", "plzz", "plese", "please", "pleaseee"]),
    (r"\byou\b", ["u", "you", "you"]),
    (r"\bnahi\b", ["nhi", "nahi", "nai", "nahin"]),
    (r"\bthanks\b", ["thx", "thanks", "thanku", "thnx"]),
    (r"\bthank you\b", ["thank u", "thankyou", "thank you", "tysm"]),
    (r"\bbecause\b", ["bcz", "coz", "because"]),
    (r"\bvideo\b", ["video", "vdo", "vedio", "vid"]),
    (r"\bkaise\b", ["kaise", "kese", "kaise"]),
    (r"\bhai\b", ["hai", "h", "he"]),
    (r"\bkya\b", ["kya", "kia", "kya"]),
    (r"\bkar raha\b", ["kar raha", "kr rha", "kar rha"]),
    (r"\bi am\b", ["i am", "im", "i'm"]),
]
ELONGATE = {"please", "pls", "plz", "sir", "bro", "bhai", "thanks", "nice", "help", "so",
            "wow", "hi", "yes", "no", "very", "amazing", "superb", "love", "awesome", "op"}
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def add_noise(text: str, rng: random.Random, cls: str) -> str:
    if _DEVANAGARI.search(text) or not re.search(r"[a-zA-Z]", text):
        # Hindi / emoji-only: only occasional emoji tail.
        if rng.random() < 0.2 and cls in ("Praise", "Confusion/Problem"):
            text += " " + rng.choice(EMO_POS if cls == "Praise" else EMO_SAD)
        return text
    for pat, options in SHORTHAND:
        if rng.random() < 0.35:
            text = re.sub(pat, lambda m: rng.choice(options), text)
    # Elongate the last word sometimes ("sirrr", "pleaseee").
    if rng.random() < 0.10:
        words = text.split()
        for i, w in enumerate(words):
            if w.lower() in ELONGATE:
                words[i] = w + w[-1] * rng.randint(2, 4)
                break
        text = " ".join(words)
    # Punctuation: drop question marks sometimes, add !! sometimes.
    if "?" in text and rng.random() < 0.25:
        text = text.replace("?", rng.choice(["", " ??", "??", " ?"]))
    if rng.random() < 0.08 and cls in ("Praise", "Request", "Spam"):
        text += rng.choice(["!", "!!", "!!!"])
    # Emoji tail matched to intent.
    tails = {"Praise": EMO_POS, "Request": ["🙏", "🙏🙏", "😊", "❤"],
             "Confusion/Problem": EMO_SAD + EMO_CONF, "Question": EMO_CONF + ["🙏"],
             "Other": EMO_FUN, "Spam": EMO_SPAM}
    if rng.random() < 0.18:
        text += " " + rng.choice(tails[cls])
    # Casing.
    r = rng.random()
    if r < 0.30:
        text = text[:1].upper() + text[1:]
    elif r < 0.33:
        text = " ".join(w[:1].upper() + w[1:] for w in text.split(" "))
    elif r < 0.355:
        text = text.upper()
    # Address suffix ("... sir") on some non-spam comments.
    if cls != "Spam" and rng.random() < 0.07:
        text += " " + rng.choice(["sir", "bhai", "mam", "bro", "sir ji", "di"])
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------
# Leakage filter against real comments
# --------------------------------------------------------------------------

def near_duplicate_mask(synthetic_texts, real_texts, threshold=LEAKAGE_THRESHOLD):
    """True where a synthetic text has char-TF-IDF cosine >= threshold to any real text.

    Uses the same representation as src.make_splits (char_wb 3-5 grams on
    normalised text) so "near-duplicate" means the same thing in both places.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np

    syn = [normalize(t) for t in synthetic_texts]
    real = [normalize(t) for t in real_texts]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=False)
    vec.fit(real + syn)
    S, R = vec.transform(syn), vec.transform(real)
    max_sim = np.zeros(len(syn))
    step = 2000
    for i in range(0, len(syn), step):
        block = (S[i:i + step] @ R.T).toarray()
        max_sim[i:i + step] = block.max(axis=1) if block.shape[1] else 0
    return max_sim >= threshold, max_sim


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def generate_candidates(cls: str, n: int, rng: random.Random):
    """Yield up to n candidate rows for one class (before global filtering)."""
    templates = TEMPLATES[cls]
    domains = list(DOMAINS)
    weights = [0.55, 0.15, 0.30]   # coding, math, exam — roughly the real channel mix
    rows = []
    for _ in range(n):
        tid, fn = templates[rng.randrange(len(templates))]
        domain = rng.choices(domains, weights)[0]
        c = Ctx(rng, domain)
        text, lang = fn(c)
        text = re.sub(r"\s+", " ", text).strip()
        text = add_noise(text, rng, cls)
        if lang == "mixed":
            lang = "hinglish" if re.search(
                r"\b(hai|ka|ki|ke|kya|nahi|nhi|karo|bana|aur|me|se|ho|kaise|kese|lu|karu|hu|wala|bhi)\b",
                text.lower()) else "en"
        rows.append({"text": text, "label_6class": cls, "language": lang,
                     "template_id": tid, "domain": domain})
    return rows


def generate(per_class: int = PER_CLASS, seed: int = SEED, real_texts=None,
             oversample: int = 6) -> tuple:
    """Return (DataFrame, stats). Deterministic for a given seed."""
    rng = random.Random(seed)
    real_norm = {normalize(t) for t in (real_texts or [])}
    out, stats = [], {"per_class": {}}
    for cls in CLASSES:
        cand = generate_candidates(cls, per_class * oversample, rng)
        df = pd.DataFrame(cand)
        df["norm"] = df["text"].map(normalize)
        df = df[df["norm"].str.len() > 0]
        n0 = len(df)
        df = df.drop_duplicates("norm")
        n_unique = len(df)
        df = df[~df["norm"].isin(real_norm)]
        if real_texts:
            near, _ = near_duplicate_mask(df["text"].tolist(), list(real_texts))
            n_near = int(near.sum())
            df = df[~near]
        else:
            n_near = 0
        if len(df) < per_class:
            raise SystemExit(f"[ERROR] only {len(df)} unique {cls} rows (need {per_class}); "
                             "add templates or raise --oversample")
        # Sample evenly across templates first so no template dominates.
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        df["_rank"] = df.groupby("template_id").cumcount()
        df = df.sort_values(["_rank", "template_id"], kind="stable").head(per_class)
        stats["per_class"][cls] = {"candidates": n0, "unique": n_unique,
                                   "dropped_near_real": n_near, "kept": len(df),
                                   "templates": int(df["template_id"].nunique())}
        out.append(df.drop(columns=["norm", "_rank"]))
    data = pd.concat(out, ignore_index=True)
    data = data.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    data.insert(0, "comment_id", [f"syn_{i:06d}" for i in range(1, len(data) + 1)])
    data.insert(1, "video_id", "synthetic")
    data.insert(2, "channel_title", data.pop("domain").map(lambda d: DOMAINS[d]["channel"]))
    data["label"] = data["label_6class"].replace({"Confusion/Problem": "Question"})
    data["source"] = SOURCE
    data = data[["comment_id", "video_id", "channel_title", "text", "label", "label_6class",
                 "language", "template_id", "source"]]
    stats["rows"] = len(data)
    stats["class_distribution"] = data["label_6class"].value_counts().reindex(CLASSES).to_dict()
    stats["language_distribution"] = data["language"].value_counts().to_dict()
    stats["n_templates"] = int(data["template_id"].nunique())
    return data, stats


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Generate class-balanced synthetic comments.")
    parser.add_argument("--per-class", type=int, default=PER_CLASS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", default=str(OUTPUT))
    parser.add_argument("--no-leakage-filter", action="store_true",
                        help="skip the near-duplicate filter against real labelled comments")
    args = parser.parse_args(argv)

    real = None
    if not args.no_leakage_filter:
        real = pd.read_csv(REAL_LABELLED, dtype=str, keep_default_na=False,
                           encoding=CSV_ENCODING)["text"].tolist()
    data, stats = generate(args.per_class, args.seed, real)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(out, index=False, encoding=CSV_ENCODING)
    stats.update({"seed": args.seed, "per_class_target": args.per_class,
                  "leakage_filter": None if real is None else
                  f"char_wb 3-5 TF-IDF cosine >= {LEAKAGE_THRESHOLD} vs {len(real)} real comments",
                  "source": SOURCE})
    out.with_name(out.stem + "_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"Wrote {len(data)} rows to {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main())
