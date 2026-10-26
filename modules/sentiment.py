"""
sentiment.py - Assignment 10: Sentiment and opinion mining.

Three layers:
  1. VADER (rule/lexicon based) - always available through a fallback.
  2. TF-IDF + Logistic Regression - trained only when labelled data is given.
  3. Rule-based emotion detection and aspect (topic) opinion mining.

All outputs are ESTIMATES and can be wrong (sarcasm, mixed opinions, slang).
"""

import os
import re
from collections import Counter

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from . import resources
from .preprocessing import tokenize, remove_punctuation, split_sentences, STOP_WORDS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sentiment_model.joblib")

DISCLAIMER = ("Sentiment and emotion results are automatic estimates. They can be wrong, "
              "especially for sarcasm, mixed opinions or slang.")

# ---------------------------------------------------------------- fallback lexicon
_FALLBACK_LEXICON = {
    "good": 1.9, "great": 3.1, "excellent": 3.2, "amazing": 2.8, "awesome": 3.1, "brilliant": 2.8,
    "love": 3.2, "loved": 2.9, "enjoyed": 2.2, "fun": 2.3, "beautiful": 2.9, "wonderful": 2.7,
    "fantastic": 2.6, "best": 3.2, "masterpiece": 3.0, "stunning": 2.6, "superb": 3.0, "perfect": 2.7,
    "entertaining": 2.2, "touching": 2.0, "moving": 1.8, "impressive": 2.3, "recommend": 1.5,
    "bad": -2.5, "terrible": -2.1, "awful": -2.0, "boring": -1.3, "worst": -3.1, "hate": -2.7,
    "hated": -3.0, "poor": -2.1, "dull": -1.7, "waste": -1.8, "disappointing": -2.2,
    "disappointed": -2.2, "weak": -1.9, "predictable": -1.2, "horrible": -2.5, "slow": -0.8,
    "mess": -1.5, "annoying": -1.9, "forgettable": -1.5, "cringe": -2.0, "pointless": -1.8,
}
_NEGATIONS = {"not", "no", "never", "n't", "hardly", "without", "isnt", "wasnt", "dont", "didnt"}


def _lexicon():
    """Word -> valence dictionary (VADER's if available, else the small fallback)."""
    if resources.VADER is not None:
        return resources.VADER.lexicon
    return _FALLBACK_LEXICON


def _fallback_compound(text):
    """Tiny sentiment scorer used only when VADER is missing."""
    toks = remove_punctuation(tokenize(text))
    total = 0.0
    for i, w in enumerate(toks):
        v = _FALLBACK_LEXICON.get(w)
        if v is None:
            continue
        if i > 0 and toks[i - 1] in _NEGATIONS:
            v = -0.74 * v
        total += v
    return total / ((total * total + 15) ** 0.5)  # normalise to -1..1 like VADER


def vader_compound(text):
    if resources.VADER is not None:
        return resources.VADER.polarity_scores(text)["compound"]
    return _fallback_compound(text)


def label_from_score(score, threshold=0.05):
    if score >= threshold:
        return "Positive"
    if score <= -threshold:
        return "Negative"
    return "Neutral"


def engine_name():
    if resources.VADER is not None:
        return "VADER lexicon analyzer"
    return "Small built-in word list (VADER not installed - less accurate)"


def explain_score(score):
    return (f"The compound score ranges from -1 (very negative) to +1 (very positive). "
            f"Between -0.05 and +0.05 is treated as Neutral. This review scored {score:+.2f}. "
            f"It is a measure of how strongly the words lean positive or negative, not a probability.")


# ---------------------------------------------------------------- opinion words
def opinion_words(text, top_n=8):
    """Words in the text that carry positive or negative valence (from the lexicon)."""
    lex = _lexicon()
    toks = remove_punctuation(tokenize(text))
    pos, neg = Counter(), Counter()
    for w in toks:
        v = lex.get(w)
        if v is None or w in STOP_WORDS:
            continue
        if v >= 1.0:
            pos[w] += 1
        elif v <= -1.0:
            neg[w] += 1
    return {"positive": pos.most_common(top_n), "negative": neg.most_common(top_n)}


# ---------------------------------------------------------------- emotions
EMOTION_WORDS = {
    "Happiness": ["happy", "joy", "love", "loved", "enjoy", "enjoyed", "wonderful", "delight", "delightful",
                  "smile", "fun", "charming", "heartwarming", "pleasant", "beautiful", "satisf", "glad"],
    "Excitement": ["exciting", "excited", "thrill", "thrilling", "thriller", "gripping", "edge", "intense",
                   "epic", "amazing", "awesome", "spectacular", "breathtaking", "stunning", "adrenaline", "wow"],
    "Sadness": ["sad", "cry", "cried", "tears", "heartbreaking", "depress", "tragic", "emotional", "grief",
                "lonely", "melanchol", "touching", "moving", "sorrow"],
    "Anger": ["angry", "anger", "hate", "hated", "furious", "annoy", "annoying", "infuriating", "rage",
              "outrag", "irritat", "insult", "stupid", "ridiculous"],
    "Fear": ["scary", "scared", "fear", "terrifying", "creepy", "horror", "frightening", "tense", "dread"],
    "Boredom/Disappointment": ["boring", "bored", "dull", "disappoint", "disappointing", "tedious", "slow",
                               "waste", "meh", "forgettable", "predictable", "underwhelming"],
}


def detect_emotions(text):
    """Count emotion cue words. Simple rule-based method (not a trained emotion model)."""
    toks = remove_punctuation(tokenize(text))
    found = {}
    for emotion, cues in EMOTION_WORDS.items():
        hits = [w for w in toks if any(w == c or (len(c) >= 5 and w.startswith(c)) for c in cues)]
        if hits:
            found[emotion] = sorted(set(hits))
    ranked = sorted(found.items(), key=lambda kv: -len(kv[1]))
    return {
        "emotions": [{"emotion": e, "cue_words": w, "count": len(w)} for e, w in ranked],
        "method": "Rule-based: counts emotion cue words from a small hand-made list.",
    }


# ---------------------------------------------------------------- aspects / topics
ASPECTS = {
    "Acting": ["acting", "actor", "actors", "actress", "cast", "performance", "performances", "role"],
    "Story/Plot": ["story", "plot", "script", "writing", "screenplay", "storyline", "narrative", "twist"],
    "Direction": ["direction", "director", "directed", "filmmaker"],
    "Visuals": ["visuals", "cinematography", "effects", "cgi", "camera", "scenery", "animation", "visual"],
    "Music/Sound": ["music", "soundtrack", "score", "songs", "song", "sound", "background"],
    "Characters": ["character", "characters", "protagonist", "villain", "hero"],
    "Ending": ["ending", "climax", "finale", "conclusion"],
    "Pacing": ["pacing", "pace", "length", "runtime", "slow", "fast", "editing"],
}


def aspect_opinions(texts):
    """
    For each aspect (acting, plot, music...), find sentences that mention it and
    average their VADER score. Only aspects that were actually mentioned are returned.
    """
    stats = {a: {"mentions": 0, "score_sum": 0.0} for a in ASPECTS}
    for text in texts:
        for sent in split_sentences(text):
            words = set(remove_punctuation(tokenize(sent)))
            for aspect, cues in ASPECTS.items():
                if words & set(cues):
                    stats[aspect]["mentions"] += 1
                    stats[aspect]["score_sum"] += vader_compound(sent)
    out = []
    for aspect, s in stats.items():
        if s["mentions"]:
            avg = s["score_sum"] / s["mentions"]
            out.append({"aspect": aspect, "mentions": s["mentions"],
                        "avg_score": round(avg, 2), "opinion": label_from_score(avg, 0.1)})
    out.sort(key=lambda x: -x["mentions"])
    return out


# ---------------------------------------------------------------- ML model
def find_columns(df, text_col=None, label_col=None):
    """Find the review text and label columns. Returns (text_col, label_col, error)."""
    lower = {c.lower().strip(): c for c in df.columns}
    if text_col and text_col not in df.columns:
        return None, None, f"Column '{text_col}' not found. Available columns: {', '.join(map(str, df.columns))}"
    if label_col and label_col not in df.columns:
        return None, None, f"Column '{label_col}' not found. Available columns: {', '.join(map(str, df.columns))}"
    if not text_col:
        for name in ("review", "text", "comment", "content", "headline", "sentence"):
            if name in lower:
                text_col = lower[name]
                break
    if not label_col:
        for name in ("sentiment", "label", "class", "target", "polarity"):
            if name in lower:
                label_col = lower[name]
                break
    if not text_col:
        return None, None, ("Could not find a review text column. Use one of: review, text, comment "
                            "or pick the column manually. Found: " + ", ".join(map(str, df.columns)))
    return text_col, label_col, None


def normalise_label(value):
    """Turn 1/0/pos/neg/positive/negative/neutral into Positive/Negative/Neutral (or None)."""
    v = str(value).strip().lower()
    if v in ("positive", "pos", "1", "1.0", "good", "p"):
        return "Positive"
    if v in ("negative", "neg", "0", "0.0", "-1", "bad", "n"):
        return "Negative"
    if v in ("neutral", "neu", "2", "mixed"):
        return "Neutral"
    return None


def train_model(df, text_col=None, label_col=None, source_name="uploaded data"):
    """
    Train TF-IDF + Logistic Regression on labelled reviews and evaluate it on a held-out 25% test set.
    Returns (report_dict, error_message). Metrics are computed on data the model has NOT seen.
    """
    text_col, label_col, err = find_columns(df, text_col, label_col)
    if err:
        return None, err
    if not label_col:
        return None, ("No label column found. Add a column named 'sentiment' or 'label' "
                      "(values like positive/negative), or choose it manually.")
    data = pd.DataFrame({
        "text": df[text_col].astype(str),
        "label": df[label_col].map(normalise_label),
    }).dropna()
    data = data[data["text"].str.strip().str.len() > 0]
    if len(data) < 10:
        return None, f"Need at least 10 labelled reviews to train; found {len(data)} usable rows."
    counts = data["label"].value_counts()
    if len(counts) < 2:
        return None, "The labels contain only one class. At least two classes (e.g. Positive and Negative) are needed."
    if counts.min() < 2:
        return None, "Each class needs at least 2 reviews so it can be split into train and test sets."

    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["label"], test_size=0.25, random_state=42, stratify=data["label"])
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(vec.fit_transform(X_train), y_train)
    pred = clf.predict(vec.transform(X_test))

    classes = sorted(data["label"].unique())
    prec, rec, f1, support = precision_recall_fscore_support(y_test, pred, labels=classes, zero_division=0)
    w_prec, w_rec, w_f1, _ = precision_recall_fscore_support(y_test, pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, pred, labels=classes)

    # Compare with VADER on the very same test reviews
    vader_pred = [label_from_score(vader_compound(t)) for t in X_test]
    vader_acc = accuracy_score(y_test, vader_pred)

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"vectorizer": vec, "model": clf, "classes": list(clf.classes_), "source": source_name}, MODEL_PATH)

    report = {
        "source": source_name,
        "n_total": int(len(data)), "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "class_counts": {k: int(v) for k, v in counts.items()},
        "accuracy": round(float(accuracy_score(y_test, pred)), 3),
        "weighted": {"precision": round(float(w_prec), 3), "recall": round(float(w_rec), 3), "f1": round(float(w_f1), 3)},
        "per_class": [{"label": c, "precision": round(float(p), 3), "recall": round(float(r), 3),
                       "f1": round(float(f), 3), "support": int(s)}
                      for c, p, r, f, s in zip(classes, prec, rec, f1, support)],
        "classes": classes, "confusion": cm.tolist(),
        "vader_accuracy": round(float(vader_acc), 3),
        "small_data_warning": len(data) < 200,
    }
    return report, None


_model_cache = {"obj": None, "mtime": None}


def load_model():
    """Load the saved model if one exists, else None."""
    if not os.path.exists(MODEL_PATH):
        return None
    mtime = os.path.getmtime(MODEL_PATH)
    if _model_cache["mtime"] != mtime:
        try:
            _model_cache["obj"] = joblib.load(MODEL_PATH)
            _model_cache["mtime"] = mtime
        except Exception:
            return None
    return _model_cache["obj"]


def ml_predict(text):
    """Prediction from the trained model, or None if no model has been trained yet."""
    bundle = load_model()
    if bundle is None:
        return None
    probs = bundle["model"].predict_proba(bundle["vectorizer"].transform([text]))[0]
    best = int(probs.argmax())
    return {"label": bundle["classes"][best], "probability": round(float(probs[best]), 3),
            "source": bundle["source"]}


def make_summary(movie, label, score, opinions, emotions, n_words):
    """One short plain-English summary built only from the computed facts."""
    pos = [w for w, _ in opinions["positive"][:3]]
    neg = [w for w, _ in opinions["negative"][:3]]
    parts = [f"The review of '{movie}' reads as {label.lower()} (score {score:+.2f})."]
    if pos:
        parts.append("Positive wording: " + ", ".join(pos) + ".")
    if neg:
        parts.append("Negative wording: " + ", ".join(neg) + ".")
    if pos and neg:
        parts.append("The opinion is mixed, with both praise and criticism.")
    if emotions["emotions"]:
        parts.append("Main emotion cue: " + emotions["emotions"][0]["emotion"].lower() + ".")
    if not pos and not neg:
        parts.append("No strong opinion words were found, so the result is uncertain.")
    return " ".join(parts)
