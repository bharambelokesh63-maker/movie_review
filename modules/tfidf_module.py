"""
tfidf_module.py - Assignment 6: TF-IDF keyword extraction.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocessing import preprocess

EXPLANATION = (
    "TF-IDF = Term Frequency x Inverse Document Frequency. A word scores high when it appears often in one "
    "review (TF) but in few other reviews (IDF). Common words like 'movie' get low scores; distinctive words "
    "like 'cinematography' get high scores. With only one review there is nothing to compare against, so "
    "scores are based on that review alone."
)


def _clean(text):
    return " ".join(preprocess(text))


def top_keywords(texts, top_n=10):
    """
    texts: list of strings. Returns list of (word, score), highest first.
    Scores are the average TF-IDF of the word across all texts.
    """
    cleaned = [_clean(t) for t in texts if t and t.strip()]
    cleaned = [c for c in cleaned if c]
    if not cleaned:
        return []
    try:
        vec = TfidfVectorizer()
        matrix = vec.fit_transform(cleaned)
    except ValueError:
        return []
    scores = np.asarray(matrix.mean(axis=0)).ravel()
    words = vec.get_feature_names_out()
    order = scores.argsort()[::-1][:top_n]
    return [(words[i], round(float(scores[i]), 4)) for i in order if scores[i] > 0]


def tfidf_table(texts, top_n=10):
    """For the NLP Lab: per-document top words, plus the overall ranking."""
    cleaned = [_clean(t) for t in texts if t and t.strip()]
    cleaned = [c for c in cleaned if c]
    if not cleaned:
        return {"error": "No usable text. Enter at least one non-empty review."}
    try:
        vec = TfidfVectorizer()
        matrix = vec.fit_transform(cleaned)
    except ValueError:
        return {"error": "The text contains no usable words after cleaning."}
    words = vec.get_feature_names_out()
    per_doc = []
    for i in range(matrix.shape[0]):
        row = matrix[i].toarray().ravel()
        order = row.argsort()[::-1][:top_n]
        per_doc.append({
            "doc": i + 1,
            "text": cleaned[i][:80],
            "top": [(words[j], round(float(row[j]), 4)) for j in order if row[j] > 0],
        })
    return {"overall": top_keywords(texts, top_n), "per_doc": per_doc, "n_docs": len(cleaned)}
