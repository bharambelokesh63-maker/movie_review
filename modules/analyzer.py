"""
analyzer.py - Assignment 11: ties all modules together.

analyze_review()  -> everything about ONE review
analyze_batch()   -> statistics for MANY reviews
"""

from collections import Counter

from . import sentiment, visualization
from .ner import extract_entities
from .ngrams_module import frequent_ngrams
from .preprocessing import preprocess, text_stats
from .tfidf_module import top_keywords
from .topics import discover_topics

MIN_WORDS_WARNING = 5
MAX_REVIEW_CHARS = 20000

GENERIC_WORDS = {"movie", "film", "films", "movies", "one", "would", "also", "really", "get", "got", "even",
                 "much", "make", "makes", "made", "see", "watch", "watched", "thing", "things"}


def analyze_review(movie, text):
    """Returns (result, error). Only real computed values are returned."""
    movie = (movie or "").strip() or "Untitled movie"
    text = (text or "").strip()
    if not text:
        return None, "Please write or paste a review before analyzing."
    if len(text) > MAX_REVIEW_CHARS:
        return None, f"The review is too long ({len(text)} characters). Please keep it under {MAX_REVIEW_CHARS}."

    stats = text_stats(text)
    warnings = []
    if stats["word_count"] == 0:
        return None, "The review has no readable words. Please enter some text."
    if stats["word_count"] < MIN_WORDS_WARNING:
        warnings.append("This review is very short, so the result is less reliable.")
    if stats["language"] not in ("English", "Unknown"):
        warnings.append(f"The text looks like {stats['language']}, but the sentiment tools here are built "
                        "for English. Treat the result as unreliable.")
    elif stats["language"] == "Unknown":
        warnings.append("The language could not be confirmed. Results assume English.")

    score = sentiment.vader_compound(text)
    label = sentiment.label_from_score(score)
    opinions = sentiment.opinion_words(text)
    emotions = sentiment.detect_emotions(text)
    entities = extract_entities(text)
    keywords = top_keywords([text], top_n=8)
    ml = sentiment.ml_predict(text)

    result = {
        "movie": movie, "review": text, "stats": stats, "warnings": warnings,
        "sentiment": label, "score": score, "score_explanation": sentiment.explain_score(score),
        "engine": sentiment.engine_name(),
        "ml": ml,
        "opinions": opinions, "emotions": emotions, "entities": entities,
        "keywords": keywords,
        "bigrams": frequent_ngrams([text], n=2, top_n=5, remove_stops=True),
        "aspects": sentiment.aspect_opinions([text]),
        "summary": sentiment.make_summary(movie, label, score, opinions, emotions, stats["word_count"]),
        "disclaimer": sentiment.DISCLAIMER,
    }
    return result, None


def analyze_batch(reviews, movie="", source="pasted reviews"):
    """
    reviews: list of {'text', 'rating'}.
    Returns (result, error). Ratings are averaged only if they were supplied.
    """
    reviews = [r for r in reviews if r.get("text", "").strip()]
    if not reviews:
        return None, "No reviews to analyze. Paste reviews (separated by blank lines) or upload a CSV."

    rows, counts, scores = [], Counter(), []
    for r in reviews:
        text = r["text"][:MAX_REVIEW_CHARS]
        s = sentiment.vader_compound(text)
        lab = sentiment.label_from_score(s)
        counts[lab] += 1
        scores.append(s)
        rows.append({"text": text, "score": round(s, 3), "label": lab, "rating": r.get("rating")})

    n = len(rows)
    pct = {k: round(100 * counts.get(k, 0) / n, 1) for k in ("Positive", "Negative", "Neutral")}
    ratings = [r["rating"] for r in rows if r["rating"] is not None]
    avg_rating = round(sum(ratings) / len(ratings), 2) if ratings else None

    texts = [r["text"] for r in rows]
    words = Counter()
    for t in texts:
        words.update(w for w in preprocess(t) if w.isalpha() and w not in GENERIC_WORDS)
    top_words = words.most_common(12)

    pos_words, neg_words = Counter(), Counter()
    for t in texts:
        op = sentiment.opinion_words(t, top_n=50)
        pos_words.update(dict(op["positive"]))
        neg_words.update(dict(op["negative"]))

    keywords = [(w, s) for w, s in top_keywords(texts, top_n=12)]
    topics = discover_topics(texts, n_topics=3)
    aspects = sentiment.aspect_opinions(texts)

    # overall verdict
    avg_score = sum(scores) / n
    verdict = sentiment.label_from_score(avg_score, 0.05)
    summary = _batch_summary(movie, n, counts, pct, avg_score, avg_rating, len(ratings), aspects, pos_words, neg_words)

    result = {
        "movie": movie or "Movie", "source": source, "total": n,
        "counts": {k: counts.get(k, 0) for k in ("Positive", "Negative", "Neutral")},
        "percent": pct, "avg_rating": avg_rating, "n_rated": len(ratings),
        "avg_score": round(avg_score, 3), "verdict": verdict,
        "top_words": top_words, "keywords": keywords,
        "pos_words": pos_words.most_common(8), "neg_words": neg_words.most_common(8),
        "bigrams": frequent_ngrams(texts, n=2, top_n=8, remove_stops=True),
        "aspects": aspects, "topics": topics,
        "rows": rows[:50], "rows_hidden": max(0, n - 50),
        "summary": summary, "disclaimer": sentiment.DISCLAIMER,
        "small_sample": n < 10,
        "charts": {
            "sentiment": visualization.sentiment_bar_chart(counts),
            "words": visualization.horizontal_bar_chart(top_words, "Most frequent words"),
            "keywords": visualization.horizontal_bar_chart(keywords, "Top TF-IDF keywords", "#3d7ac8"),
            "ratings": visualization.rating_histogram(ratings),
        },
    }
    return result, None


def _batch_summary(movie, n, counts, pct, avg_score, avg_rating, n_rated, aspects, pos_words, neg_words):
    name = f"'{movie}'" if movie else "these reviews"
    parts = [f"Across the {n} review{'s' if n != 1 else ''} provided for {name}, "
             f"{pct['Positive']}% were positive, {pct['Negative']}% negative and {pct['Neutral']}% neutral."]
    if n_rated:
        parts.append(f"The average of the {n_rated} supplied rating{'s' if n_rated != 1 else ''} is {avg_rating}.")
    else:
        parts.append("No numeric ratings were supplied, so sentiment statistics are shown instead.")
    liked = [a["aspect"] for a in aspects if a["opinion"] == "Positive"][:2]
    disliked = [a["aspect"] for a in aspects if a["opinion"] == "Negative"][:2]
    if liked:
        parts.append("Most positively discussed: " + ", ".join(liked) + ".")
    if disliked:
        parts.append("Most criticised: " + ", ".join(disliked) + ".")
    if n < 10:
        parts.append("This is a small sample, so it should not be taken as the general public opinion.")
    else:
        parts.append("This describes only the reviews you provided, not all viewers.")
    return " ".join(parts)
