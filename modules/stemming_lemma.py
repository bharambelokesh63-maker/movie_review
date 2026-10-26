"""
stemming_lemma.py - Assignment 3: Porter stemming and WordNet lemmatization.
"""

from nltk.stem import PorterStemmer, WordNetLemmatizer

from . import resources
from .preprocessing import tokenize, remove_punctuation

_stemmer = PorterStemmer()
_lemmatizer = WordNetLemmatizer() if resources.HAS["wordnet"] else None

# Used only when WordNet is not installed.
_IRREGULAR = {"was": "be", "were": "be", "is": "be", "are": "be", "been": "be", "am": "be",
              "better": "good", "best": "good", "worse": "bad", "worst": "bad",
              "children": "child", "men": "man", "women": "woman", "went": "go",
              "gone": "go", "did": "do", "done": "do", "had": "have", "has": "have",
              "movies": "movie", "ran": "run", "saw": "see", "seen": "see"}


def _rule_lemma(word):
    """Very small rule-based lemmatizer (fallback only)."""
    w = word.lower()
    if w in _IRREGULAR:
        return _IRREGULAR[w]
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("ing") and len(w) > 5:
        base = w[:-3]
        if len(base) > 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if w.endswith("ed") and len(w) > 4:
        base = w[:-2]
        if len(base) > 2 and base[-1] == base[-2]:
            base = base[:-1]
        return base
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def stem(word):
    return _stemmer.stem(word)


def lemmatize(word):
    """Lemmatize as a verb if that changes the word, otherwise as a noun."""
    if _lemmatizer is None:
        return _rule_lemma(word)
    as_verb = _lemmatizer.lemmatize(word, pos="v")
    return as_verb if as_verb != word else _lemmatizer.lemmatize(word, pos="n")


def compare(text, max_words=40):
    """Return rows of (original, stem, lemma) plus notes."""
    words = remove_punctuation(tokenize(text))[:max_words]
    rows = [{"original": w, "stem": stem(w), "lemma": lemmatize(w)} for w in words]
    note = None
    if _lemmatizer is None:
        note = ("WordNet is not installed, so a simple rule-based lemmatizer was used instead. "
                "Run nltk.download('wordnet') for true WordNet lemmatization.")
    return {"rows": rows, "note": note, "wordnet_used": _lemmatizer is not None}


EXPLANATION = (
    "Stemming chops word endings with fixed rules (Porter algorithm), so the result may not be a real "
    "word (e.g. 'studies' -> 'studi'). Lemmatization uses a dictionary (WordNet) and the word's role "
    "to return a real base word (e.g. 'studies' -> 'study', 'better' -> 'good'). Stemming is faster; "
    "lemmatization is more accurate."
)
