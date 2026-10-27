"""Tests for src.preprocess (no API key or network needed).

Lemmatisation tests are skipped when the NLTK WordNet data is not installed
(install with:  python -m nltk.downloader wordnet omw-1.4).
"""

import pandas as pd
import pytest

from src.preprocess import lemmatize, normalize, preprocess_frame, remove_stopwords, tokenize


def _has_wordnet():
    try:
        import nltk
        nltk.data.find("corpora/wordnet")
        return True
    except Exception:
        return False


needs_wordnet = pytest.mark.skipif(not _has_wordnet(), reason="WordNet data not installed")


def toks(text):
    return tokenize(normalize(text))


# --- normalisation -------------------------------------------------------

def test_negations_survive_contractions():
    assert toks("I don't understand, it isn't working") == \
        ["i", "do", "not", "understand", "it", "is", "not", "working"]
    assert toks("can't") == ["can", "not"]
    assert toks("I DONT get it") == ["i", "do", "not", "get", "it"]
    assert toks("won't") == ["will", "not"]


def test_placeholders_and_elongation():
    assert toks("12:03 see https://x.com/a @harry soooo good") == \
        ["<time>", "see", "<url>", "<user>", "soo", "good"]


def test_spelling_variants_unified():
    assert toks("plzzz sir notes do") == ["please", "sir", "notes", "do"]
    assert toks("samajh nhi aaya") == ["samajh", "nahi", "aaya"]
    assert toks("love u, ur videos") == ["love", "you", "your", "videos"]
    assert toks("bhut accha chaiye") == ["bahut", "accha", "chahiye"]


def test_emoji_kept_and_variation_selectors_dropped():
    assert toks("Amazing ❤️❤️❤️ 🙏🏼") == ["amazing", "❤", "❤", "🙏"]


# --- tokenisation --------------------------------------------------------

def test_devanagari_words_stay_whole():
    assert toks("बहुत बढ़िया वीडियो धन्यवाद।") == ["बहुत", "बढ़िया", "वीडियो", "धन्यवाद"]


def test_question_and_exclamation_kept():
    assert toks("Is ye exam me aayega????") == ["is", "ye", "exam", "me", "aayega", "?", "?"]
    assert toks("Please!!!") == ["please", "!", "!"]


# --- lemmatisation -------------------------------------------------------

@needs_wordnet
def test_english_inflections_reduced():
    assert lemmatize(["videos", "explained", "watched", "was", "has", "notes", "nodes", "dies"]) == \
        ["video", "explain", "watch", "be", "have", "note", "node", "die"]


@needs_wordnet
def test_wordnet_artefacts_avoided():
    assert lemmatize(["hoping", "coded", "physics", "pass", "boss", "bit", "css", "latest"]) == \
        ["hope", "code", "physics", "pass", "boss", "bit", "css", "latest"]


@needs_wordnet
def test_hinglish_and_negations_untouched():
    words = ["wo", "us", "me", "ye", "bas", "kis", "nahi", "not", "never", "kya", "kaise", "karo"]
    assert lemmatize(words) == words


@needs_wordnet
def test_intent_adjectives_kept():
    assert lemmatize(["confused", "stuck", "amazing", "confusing"]) == \
        ["confused", "stuck", "amazing", "confusing"]


# --- stopwords and frame -------------------------------------------------

def test_stopwords_keep_negations_questions_and_hinglish():
    out = remove_stopwords(["this", "is", "not", "the", "video", "why", "me", "hai", "please"])
    assert out == ["is", "not", "video", "why", "me", "hai", "please"]


def test_original_text_preserved():
    df = pd.DataFrame({"text": ["Sir ye samajh nahi aaya 😭", "बहुत बढ़िया ❤️"]})
    out = preprocess_frame(df, do_lemmatize=False)
    assert list(out["text"]) == list(df["text"])
    assert {"text_norm", "tokens", "lemmas"} <= set(out.columns)


# --- regressions found in the data audit ------------------------------------

def test_ambiguous_hinglish_not_turned_into_negation():
    # "nai" here means "ne", not "nahi"; must not invent a negation.
    assert toks("aap nai is dialogue") == ["aap", "nai", "is", "dialogue"]
    assert toks("sakoon ni milta") == ["sakoon", "nahi", "milta"]


def test_single_letter_r_not_expanded():
    assert "are" not in toks("J.A.R.V.I.S. kase banaya")


def test_digits_never_squeezed():
    assert toks("10000000 and 111") == ["10000000", "and", "111"]


def test_it_acronym_vs_pronoun():
    assert toks("I worked in IT sector") == ["i", "worked", "in", "info_tech", "sector"]
    assert toks("DERIVE IT MYSELF") == ["derive", "it", "myself"]


def test_devanagari_negation_kept():
    assert "नहीं" in toks("समझ नहीं आया")
