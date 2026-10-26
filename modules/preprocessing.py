"""
preprocessing.py - Assignment 2: Text preprocessing and news dataset analysis.

Functions: detect_language, split_sentences, tokenize, remove_stopwords,
preprocess, text_stats, analyze_news_dataset.
"""

import re
import string

import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from . import resources

# Negation words are kept out of the stop list for sentiment work
# ("not good" must not become "good").
KEEP_WORDS = {"no", "not", "nor", "never", "n't"}

if resources.HAS["stopwords"]:
    from nltk.corpus import stopwords as _sw
    STOP_WORDS = set(_sw.words("english")) - KEEP_WORDS
else:
    STOP_WORDS = set(ENGLISH_STOP_WORDS) - KEEP_WORDS

# Tiny stop-word lists used only for language guessing.
_LANG_WORDS = {
    "English": {"the", "and", "is", "was", "to", "of", "it", "in", "that", "this", "with", "for", "but"},
    "Spanish": {"el", "la", "los", "las", "es", "una", "que", "de", "y", "en", "pero", "muy", "con"},
    "French": {"le", "la", "les", "est", "une", "des", "et", "que", "de", "mais", "très", "avec", "pour"},
    "German": {"der", "die", "das", "ist", "und", "ein", "eine", "nicht", "mit", "aber", "sehr", "zu"},
    "Italian": {"il", "lo", "gli", "è", "una", "che", "di", "e", "ma", "molto", "con", "per"},
    "Portuguese": {"o", "os", "as", "é", "uma", "que", "de", "e", "mas", "muito", "com", "para"},
}


def detect_language(text):
    """Guess the language. Returns (language_name, note)."""
    if not text or not text.strip():
        return "Unknown", "No text given."
    # Script check first (Hindi/Marathi, Arabic, Chinese, ...)
    scripts = {
        "Devanagari (Hindi/Marathi)": r"[ऀ-ॿ]",
        "Arabic": r"[؀-ۿ]",
        "Chinese/Japanese": r"[一-鿿぀-ヿ]",
        "Cyrillic (Russian etc.)": r"[Ѐ-ӿ]",
        "Tamil": r"[஀-௿]",
    }
    letters = [c for c in text if c.isalpha()]
    if letters:
        for name, pattern in scripts.items():
            hits = len(re.findall(pattern, text))
            if hits / len(letters) > 0.3:
                return name, "Detected from the writing script."
    try:  # optional: langdetect gives better results if installed
        from langdetect import detect
        code = detect(text)
        names = {"en": "English", "es": "Spanish", "fr": "French", "de": "German",
                 "it": "Italian", "pt": "Portuguese", "hi": "Hindi", "mr": "Marathi"}
        return names.get(code, code), "Detected with langdetect."
    except Exception:
        pass
    words = re.findall(r"[a-zA-Zà-ÿ']+", text.lower())
    if len(words) < 3:
        return "Unknown", "Text too short to guess the language."
    scores = {lang: sum(w in vocab for w in words) for lang, vocab in _LANG_WORDS.items()}
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return "Unknown", "No common words from supported languages found."
    return best, "Estimated by counting common words (stop-word overlap)."


def split_sentences(text):
    """Split text into sentences (NLTK punkt if installed, else regex)."""
    if not text or not text.strip():
        return []
    if resources.HAS["punkt"]:
        try:
            return sent_tokenize(text)
        except Exception:
            pass
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def tokenize(text, lowercase=True):
    """Split text into word tokens."""
    if not text:
        return []
    text = text.lower() if lowercase else text
    if resources.HAS["punkt"]:
        try:
            return word_tokenize(text)
        except Exception:
            pass
    return re.findall(r"[a-zA-Z0-9]+(?:'[a-z]+)?|[^\sa-zA-Z0-9]", text)


def remove_punctuation(tokens):
    return [t for t in tokens if t not in string.punctuation and any(c.isalnum() for c in t)]


def remove_stopwords(tokens):
    return [t for t in tokens if t not in STOP_WORDS]


def preprocess(text, remove_stops=True):
    """Full cleaning pipeline: lowercase -> tokenize -> drop punctuation -> drop stop words."""
    tokens = remove_punctuation(tokenize(text))
    if remove_stops:
        tokens = remove_stopwords(tokens)
    return [t for t in tokens if len(t) > 1 or t.isdigit()]


def text_stats(text):
    """Word count, sentence count and language of a text."""
    tokens = remove_punctuation(tokenize(text))
    sentences = split_sentences(text)
    lang, note = detect_language(text)
    return {
        "word_count": len(tokens),
        "sentence_count": len(sentences),
        "char_count": len(text or ""),
        "language": lang,
        "language_note": note,
    }


def preprocessing_demo(text):
    """Show every preprocessing step (used by the NLP Lab)."""
    tokens = tokenize(text)
    no_punct = remove_punctuation(tokens)
    no_stop = remove_stopwords(no_punct)
    stats = text_stats(text)
    return {
        "stats": stats,
        "sentences": split_sentences(text),
        "tokens": tokens,
        "no_punct": no_punct,
        "clean": no_stop,
        "removed_stopwords": [t for t in no_punct if t in STOP_WORDS],
    }


def analyze_news_dataset(df):
    """
    Basic statistics for a HuffPost-style news CSV
    (columns such as headline, category, short_description).
    Returns (stats_dict, error_message). Never raises.
    """
    if df is None or df.empty:
        return None, "The CSV file is empty."
    cols = {c.lower().strip(): c for c in df.columns}
    text_col = None
    for name in ("headline", "short_description", "title", "text", "review"):
        if name in cols:
            text_col = cols[name]
            break
    if text_col is None:
        return None, ("Could not find a text column. Expected one of: headline, "
                      "short_description, title, text. Found: " + ", ".join(map(str, df.columns)))
    texts = df[text_col].fillna("").astype(str)
    missing = int((texts.str.strip() == "").sum())
    texts = texts[texts.str.strip() != ""]
    if texts.empty:
        return None, f"Column '{text_col}' has no usable text."

    sample = texts.head(2000)  # keep it fast for big datasets
    word_counts = sample.apply(lambda t: len(remove_punctuation(tokenize(t))))
    all_tokens = [tok for t in sample for tok in preprocess(t)]
    from collections import Counter
    top_words = Counter(all_tokens).most_common(15)
    langs = Counter(detect_language(t)[0] for t in sample.head(200))
    stats = {
        "text_column": text_col,
        "rows": int(len(df)),
        "rows_used": int(len(sample)),
        "missing_text": missing,
        "avg_words": round(float(word_counts.mean()), 2),
        "max_words": int(word_counts.max()),
        "min_words": int(word_counts.min()),
        "vocabulary_size": len(set(all_tokens)),
        "top_words": top_words,
        "languages": langs.most_common(),
        "categories": None,
        "examples": [
            {"original": t, "clean": " ".join(preprocess(t))} for t in sample.head(5)
        ],
    }
    if "category" in cols:
        stats["categories"] = df[cols["category"]].value_counts().head(10).items()
        stats["categories"] = [(str(k), int(v)) for k, v in stats["categories"]]
    return stats, None
