"""
data_loader.py - Reading CSV files and pasted text safely.

Every function returns (result, error_message). When error_message is not None
the page shows it as a friendly message instead of crashing.
"""

import io
import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

MAX_ROWS = 5000        # keeps the demo fast
MAX_UPLOAD_MB = 10


def read_csv_bytes(raw, filename="file.csv"):
    """Read uploaded CSV bytes into a DataFrame."""
    if not raw:
        return None, "The uploaded file is empty."
    if len(raw) > MAX_UPLOAD_MB * 1024 * 1024:
        return None, f"File is larger than {MAX_UPLOAD_MB} MB. Please upload a smaller sample."
    if not str(filename).lower().endswith((".csv", ".txt", ".tsv")):
        return None, "Please upload a .csv file."
    df = None
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=encoding)  # normal comma-separated file
            except pd.errors.ParserError:
                # maybe tab/semicolon separated: let pandas guess the separator
                df = pd.read_csv(io.BytesIO(raw), encoding=encoding, sep=None, engine="python")
            break
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            return None, "The CSV file has no data."
        except Exception:
            return None, ("Could not read this file as a CSV. Check that it has a header row "
                          "and comma-separated values.")
    if df is None:
        return None, "Could not decode the file. Save it as UTF-8 CSV and try again."
    if df.empty:
        return None, "The CSV has a header but no rows."
    return df.head(MAX_ROWS), None


def read_csv_path(path):
    if not os.path.exists(path):
        return None, f"File not found: {os.path.basename(path)}"
    with open(path, "rb") as fh:
        return read_csv_bytes(fh.read(), path)


def parse_pasted_reviews(text):
    """
    Split pasted text into separate reviews.
    Reviews are separated by a blank line; if there are no blank lines, each line is one review.
    """
    if not text or not text.strip():
        return []
    text = text.replace("\r\n", "\n").strip()
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    if len(blocks) == 1 and "\n" in blocks[0]:
        blocks = [l.strip() for l in blocks[0].split("\n") if l.strip()]
    return [{"text": b, "rating": None} for b in blocks]


def detect_rating_column(df):
    lower = {c.lower().strip(): c for c in df.columns}
    for name in ("rating", "ratings", "score", "stars", "star", "rate"):
        if name in lower:
            return lower[name]
    return None


def reviews_from_df(df, text_col=None, rating_col=None):
    """
    Turn a DataFrame into a list of {'text', 'rating'} dicts.
    Returns (reviews, info, error). Ratings are used only if they are real numbers in the file.
    """
    lower = {c.lower().strip(): c for c in df.columns}
    if text_col:
        if text_col not in df.columns:
            return None, None, f"Column '{text_col}' not found. Available: {', '.join(map(str, df.columns))}"
    else:
        for name in ("review", "text", "comment", "content", "body", "reviews"):
            if name in lower:
                text_col = lower[name]
                break
    if not text_col:
        return None, None, ("Could not find a review column. Name it 'review' or 'text', or type the "
                            "column name in 'Review column'. Found columns: " + ", ".join(map(str, df.columns)))
    if rating_col and rating_col not in df.columns:
        return None, None, f"Rating column '{rating_col}' not found."
    rating_col = rating_col or detect_rating_column(df)

    texts = df[text_col].fillna("").astype(str)
    ratings = pd.to_numeric(df[rating_col], errors="coerce") if rating_col else None
    reviews, skipped = [], 0
    for i in range(len(df)):
        t = texts.iloc[i].strip()
        if not t or t.lower() == "nan":
            skipped += 1
            continue
        r = None
        if ratings is not None and pd.notna(ratings.iloc[i]):
            r = float(ratings.iloc[i])
        reviews.append({"text": t, "rating": r})
    if not reviews:
        return None, None, f"Column '{text_col}' contains no usable review text."
    info = {"text_col": text_col, "rating_col": rating_col, "skipped_empty": skipped}
    return reviews, info, None
