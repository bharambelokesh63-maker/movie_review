"""
resources.py - Checks which optional NLP resources are installed.

Every other module asks this file "is X available?" and falls back to a
simpler method if it is not. That way the website never crashes just
because an NLTK corpus or spaCy model is missing.
"""

import nltk


def _has_nltk(path):
    try:
        nltk.data.find(path)
        return True
    except LookupError:
        return False


def _check_spacy():
    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def _check_vader_package():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except Exception:
        return None


def _check_nltk_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except Exception:
        return None


# Run the checks once, when the app starts.
NLP = _check_spacy()  # spaCy pipeline or None
VADER = _check_nltk_vader() or _check_vader_package()  # VADER analyzer or None

HAS = {
    "punkt": _has_nltk("tokenizers/punkt_tab") or _has_nltk("tokenizers/punkt"),
    "stopwords": _has_nltk("corpora/stopwords"),
    "wordnet": _has_nltk("corpora/wordnet.zip") or _has_nltk("corpora/wordnet"),
    "pos_tagger": _has_nltk("taggers/averaged_perceptron_tagger_eng")
    or _has_nltk("taggers/averaged_perceptron_tagger"),
    "spacy_model": NLP is not None,
    "vader": VADER is not None,
}


def status_table():
    """Return a list of (resource, installed, what-happens-if-missing)."""
    return [
        ("NLTK punkt (tokenizer)", HAS["punkt"], "Simple regex tokenizer is used"),
        ("NLTK stopwords", HAS["stopwords"], "scikit-learn English stop-word list is used"),
        ("NLTK WordNet", HAS["wordnet"], "Rule-based lemmatizer is used (labelled as fallback)"),
        ("NLTK POS tagger", HAS["pos_tagger"], "spaCy or a heuristic tagger is used (labelled)"),
        ("spaCy en_core_web_sm", HAS["spacy_model"], "Heuristic capitalised-phrase NER is used (labelled)"),
        ("VADER lexicon", HAS["vader"], "Small built-in word list is used (labelled)"),
    ]
