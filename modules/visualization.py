"""
visualization.py - Matplotlib charts returned as base64 PNG strings
(so templates can show them with <img src="data:image/png;base64,...">).
"""

import base64
import io

import matplotlib

matplotlib.use("Agg")  # no GUI window needed on a server
import matplotlib.pyplot as plt

COLORS = {"Positive": "#2e9e6b", "Negative": "#d6453d", "Neutral": "#8a94a6"}
BG = "#fffdf8"


def _to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def sentiment_bar_chart(counts):
    """counts: {'Positive': n, 'Negative': n, 'Neutral': n}"""
    labels = ["Positive", "Neutral", "Negative"]
    values = [counts.get(l, 0) for l in labels]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    bars = ax.bar(labels, values, color=[COLORS[l] for l in labels], width=0.55)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Number of reviews")
    ax.set_title("Sentiment distribution")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(values + [1]) * 1.2)
    return _to_base64(fig)


def horizontal_bar_chart(pairs, title, color="#c8553d"):
    """pairs: list of (label, value). Highest value at the top."""
    if not pairs:
        return None
    pairs = list(pairs)[:12][::-1]
    labels = [str(p[0]) for p in pairs]
    values = [p[1] for p in pairs]
    fig, ax = plt.subplots(figsize=(5, max(2.4, 0.32 * len(pairs) + 0.8)))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.barh(labels, values, color=color)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    return _to_base64(fig)


def rating_histogram(ratings):
    if not ratings:
        return None
    fig, ax = plt.subplots(figsize=(5, 3.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.hist(ratings, bins=min(10, max(3, len(set(ratings)))), color="#e0a526", edgecolor="white")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of reviews")
    ax.set_title("Rating distribution")
    ax.spines[["top", "right"]].set_visible(False)
    return _to_base64(fig)
