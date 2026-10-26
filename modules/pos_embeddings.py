"""
pos_embeddings.py - Assignment 4: POS tagging and word embeddings / similarity.

Word embeddings: only used if spaCy has a model WITH vectors (en_core_web_md/lg).
Otherwise a TF-IDF cosine similarity is shown and clearly labelled as a substitute.
"""

import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import resources
from .preprocessing import tokenize, remove_punctuation

POS_GROUPS = {"NOUN": "Noun", "VERB": "Verb", "ADJ": "Adjective", "ADV": "Adverb"}

_PENN_TO_SIMPLE = {"NN": "NOUN", "VB": "VERB", "JJ": "ADJ", "RB": "ADV"}

# Heuristic fallback word lists
_DETERMINERS = {"the", "a", "an", "this", "that", "these", "those", "my", "your", "his", "her", "its", "our", "their"}
_PRONOUNS = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "us", "them"}
_PREPS = {"in", "on", "at", "of", "to", "for", "with", "by", "from", "about", "as", "into", "over", "after"}
_CONJ = {"and", "but", "or", "so", "because", "although", "while"}
_AUX = {"is", "are", "was", "were", "be", "been", "am", "do", "does", "did", "have", "has", "had",
        "will", "would", "can", "could", "should", "may", "might"}
_COMMON_ADJ = {"good", "bad", "great", "terrible", "awful", "amazing", "boring", "beautiful", "poor",
               "excellent", "brilliant", "dull", "fantastic", "weak", "strong", "best", "worst", "long",
               "slow", "fast", "funny", "sad", "happy", "wonderful", "horrible", "predictable"}


_COMMON_VERBS = {"felt", "made", "went", "saw", "took", "got", "gave", "found", "thought", "knew", "kept",
                 "left", "became", "seemed", "looked", "watched", "loved", "liked", "hated", "think", "feel",
                 "make", "see", "watch", "love", "like", "hate", "want", "need", "recommend", "enjoy", "directed",
                 "wrote", "play", "played", "show", "showed", "ran", "run"}


def _heuristic_tag(word):
    w = word.lower()
    if w in _COMMON_VERBS:
        return "VERB"
    if w in _DETERMINERS:
        return "DET"
    if w in _PRONOUNS:
        return "PRON"
    if w in _PREPS:
        return "ADP"
    if w in _CONJ:
        return "CONJ"
    if w in _AUX:
        return "VERB"
    if w in _COMMON_ADJ:
        return "ADJ"
    if w.endswith("ly") and len(w) > 3:
        return "ADV"
    if w.endswith(("ing", "ed", "ize", "ise")) and len(w) > 4:
        return "VERB"
    if w.endswith(("ful", "ous", "ive", "able", "ible", "al", "ic", "less")) and len(w) > 4:
        return "ADJ"
    if w.isdigit():
        return "NUM"
    return "NOUN"


def pos_tag(text):
    """Return (list of (word, tag), method_description)."""
    if resources.NLP is not None:
        doc = resources.NLP(text)
        return [(t.text, t.pos_) for t in doc if not t.is_space], "spaCy tagger (en_core_web_sm)"
    if resources.HAS["pos_tagger"]:
        try:
            tokens = remove_punctuation(tokenize(text, lowercase=False))
            tagged = nltk.pos_tag(tokens)
            out = [(w, _PENN_TO_SIMPLE.get(t[:2], t)) for w, t in tagged]
            return out, "NLTK averaged perceptron tagger"
        except Exception:
            pass
    tokens = remove_punctuation(tokenize(text, lowercase=False))
    return [(w, _heuristic_tag(w)) for w in tokens], (
        "Heuristic fallback tagger (word lists + suffix rules). Install the NLTK tagger or a spaCy "
        "model for real POS tagging; this version will make mistakes.")


def group_by_pos(tagged):
    groups = {label: [] for label in POS_GROUPS.values()}
    for word, tag in tagged:
        if tag in POS_GROUPS:
            groups[POS_GROUPS[tag]].append(word)
    return groups


def pos_demo(text):
    tagged, method = pos_tag(text)
    return {"tagged": tagged, "method": method, "groups": group_by_pos(tagged)}


def similarity_demo(text_a, text_b):
    """Compare two texts. Uses real spaCy vectors if present, else TF-IDF cosine similarity."""
    if not text_a.strip() or not text_b.strip():
        return {"error": "Please enter both texts."}
    nlp = resources.NLP
    if nlp is not None and nlp.vocab.vectors.shape[0] > 0:
        score = nlp(text_a).similarity(nlp(text_b))
        return {"score": round(float(score), 3), "method": "spaCy word embeddings (real vectors)",
                "is_embedding": True}
    try:
        vec = TfidfVectorizer().fit([text_a, text_b])
        m = vec.transform([text_a, text_b])
        score = cosine_similarity(m[0], m[1])[0][0]
    except ValueError:
        return {"error": "Texts have no usable words."}
    return {
        "score": round(float(score), 3),
        "method": "TF-IDF cosine similarity (SUBSTITUTE - not a true word embedding)",
        "is_embedding": False,
        "note": ("TF-IDF only matches identical words, so 'great' and 'excellent' score 0. "
                 "True embeddings (Word2Vec, GloVe, spaCy md/lg models) place similar-meaning words close "
                 "together. Install 'en_core_web_md' to enable them."),
    }
