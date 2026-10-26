"""
topics.py - Assignment 9: Topic modelling with scikit-learn (LDA or LSI).

Gensim is not required: scikit-learn's LatentDirichletAllocation and TruncatedSVD
(= LSI/LSA) give the same ideas with fewer installation problems.
"""

import numpy as np
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from .preprocessing import preprocess

MIN_DOCS = 4

# Words that appear in nearly every movie review and make topics less useful.
GENERIC = {"movie", "film", "films", "movies", "just", "really", "one", "like", "also", "even", "get", "got"}


def discover_topics(texts, n_topics=3, n_words=6, method="lda"):
    """
    Returns {'topics': [{'id', 'words': [(word, weight)]}], 'method': str} or {'error': str}.
    Never raises.
    """
    cleaned = []
    for t in texts:
        toks = [w for w in preprocess(t) if w not in GENERIC and w.isalpha()]
        if toks:
            cleaned.append(" ".join(toks))
    if len(cleaned) < MIN_DOCS:
        return {"error": f"Topic modelling needs at least {MIN_DOCS} non-empty reviews "
                         f"(found {len(cleaned)}). Add more reviews."}
    try:
        n_topics = max(2, min(int(n_topics), len(cleaned) - 1, 8))
        if method == "lsi":
            vec = TfidfVectorizer(min_df=1)
            matrix = vec.fit_transform(cleaned)
            if matrix.shape[1] <= n_topics:
                return {"error": "Not enough different words to find topics."}
            model = TruncatedSVD(n_components=n_topics, random_state=42)
            model.fit(matrix)
            label = "LSI (TruncatedSVD on TF-IDF)"
        else:
            vec = CountVectorizer(min_df=1)
            matrix = vec.fit_transform(cleaned)
            if matrix.shape[1] <= n_topics:
                return {"error": "Not enough different words to find topics."}
            model = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=30)
            model.fit(matrix)
            label = "LDA (Latent Dirichlet Allocation)"
        words = vec.get_feature_names_out()
        topics = []
        for i, comp in enumerate(model.components_):
            # LSI components can be negative; use absolute size for ranking
            order = np.abs(comp).argsort()[::-1][:n_words]
            total = np.abs(comp).sum() or 1.0
            topics.append({
                "id": i + 1,
                "words": [(words[j], round(float(abs(comp[j]) / total), 3)) for j in order],
            })
        return {"topics": topics, "method": label, "n_docs": len(cleaned),
                "note": "Topics are unlabelled word groups; a human decides what each one means. "
                        "Small datasets give rough topics."}
    except Exception as exc:  # topic models can fail on tiny/odd data
        return {"error": f"Could not train the topic model on this data ({type(exc).__name__}). "
                         "Try adding more or longer reviews."}
