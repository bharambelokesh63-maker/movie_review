"""
ngrams_module.py - Assignments 7 and 8: N-gram analysis and N-gram language model.
"""

import math
from collections import Counter

from nltk.util import ngrams

from .preprocessing import preprocess, tokenize, remove_punctuation, split_sentences


# ---------------- Assignment 7: N-gram analysis ----------------

def make_ngrams(tokens, n):
    if n < 1 or len(tokens) < n:
        return []
    return [" ".join(g) for g in ngrams(tokens, n)]


def frequent_ngrams(texts, n=2, top_n=10, remove_stops=False):
    """Most frequent n-grams over a list of texts (n between 1 and 5)."""
    n = max(1, min(int(n), 5))
    counter = Counter()
    for t in texts:
        tokens = preprocess(t, remove_stops=remove_stops)
        counter.update(make_ngrams(tokens, n))
    return counter.most_common(top_n)


def ngram_demo(text, n=2, top_n=10):
    if not text or not text.strip():
        return {"error": "Please enter some text."}
    tokens = preprocess(text, remove_stops=False)
    if len(tokens) < n:
        return {"error": f"Need at least {n} words to build {n}-grams; the text has {len(tokens)}."}
    all_grams = make_ngrams(tokens, n)
    return {"n": n, "total": len(all_grams), "all": all_grams[:30],
            "top": Counter(all_grams).most_common(top_n)}


# ---------------- Assignment 8: N-gram language model ----------------

class BigramModel:
    """Bigram language model with add-one (Laplace) smoothing."""

    START, END, UNK = "<s>", "</s>", "<unk>"

    def __init__(self, texts, min_count=1):
        raw = []
        for t in texts:
            for s in split_sentences(t):
                toks = remove_punctuation(tokenize(s))
                if toks:
                    raw.append(toks)
        word_freq = Counter(w for s in raw for w in s)
        self.vocab = {w for w, c in word_freq.items() if c >= min_count}
        self.vocab |= {self.START, self.END, self.UNK}
        self.unigrams, self.bigrams = Counter(), Counter()
        for s in raw:
            s = [self.START] + [w if w in self.vocab else self.UNK for w in s] + [self.END]
            self.unigrams.update(s)
            self.bigrams.update(zip(s[:-1], s[1:]))
        self.V = len(self.vocab)
        self.n_sentences = len(raw)

    def _map(self, w):
        return w if w in self.vocab else self.UNK

    def prob(self, prev, word):
        """P(word | prev) = (count(prev,word) + 1) / (count(prev) + V)."""
        prev, word = self._map(prev), self._map(word)
        c_pair = self.bigrams[(prev, word)]
        c_prev = self.unigrams[prev]
        return (c_pair + 1) / (c_prev + self.V), c_pair, c_prev

    def sequence_probability(self, sentence):
        toks = remove_punctuation(tokenize(sentence))
        if not toks:
            return {"error": "Please enter at least one word."}
        seq = [self.START] + toks + [self.END]
        steps, log_p = [], 0.0
        for prev, word in zip(seq[:-1], seq[1:]):
            p, c_pair, c_prev = self.prob(prev, word)
            log_p += math.log(p)
            steps.append({
                "pair": f"P({word} | {prev})",
                "formula": f"({c_pair} + 1) / ({c_prev} + {self.V})",
                "prob": p,
                "unknown": word not in self.vocab or prev not in self.vocab,
            })
        perplexity = math.exp(-log_p / len(steps))
        return {"steps": steps, "log_prob": log_p, "probability": math.exp(log_p),
                "perplexity": perplexity, "tokens": toks}

    def next_words(self, prev, k=5):
        prev = self._map(prev.lower())
        cands = [(w, c) for (p, w), c in self.bigrams.items() if p == prev and w not in (self.END,)]
        cands.sort(key=lambda x: -x[1])
        return cands[:k]


def language_model_demo(training_texts, test_sentence):
    texts = [t for t in training_texts if t and t.strip()]
    if len(texts) == 0:
        return {"error": "No training text available."}
    model = BigramModel(texts)
    if model.n_sentences < 2:
        note = "Very little training text: probabilities will be unreliable."
    else:
        note = None
    result = model.sequence_probability(test_sentence)
    if "error" in result:
        return result
    result["vocab_size"] = model.V
    result["train_sentences"] = model.n_sentences
    result["note"] = note
    last = result["tokens"][-1]
    result["next_suggestions"] = model.next_words(last)
    return result
