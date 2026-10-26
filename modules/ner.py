"""
ner.py - Assignment 5: Named Entity Recognition.

Uses spaCy when en_core_web_sm is installed. Otherwise a simple heuristic finds
runs of Capitalised words and labels them "PROPER_NOUN?" - it cannot tell a PERSON
from an ORG, and says so honestly.
"""

import re

from . import resources

EXPLANATION = (
    "NER finds real-world names in text and labels them (PERSON, ORG, GPE = country/city, DATE...). "
    "The result depends on the language model and on context: 'Apple' can be a fruit or a company, "
    "and unusual names may be missed. Results are estimates."
)

_SKIP_START = {"The", "A", "An", "I", "It", "This", "That", "But", "And", "He", "She", "We", "They",
               "My", "His", "Her", "In", "On", "At", "If", "So", "Is", "Was", "What", "When", "Why"}


def _heuristic_ner(text):
    entities = []
    # runs of Capitalised words, e.g. "Christopher Nolan", "New York"
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", text):
        phrase = m.group(1)
        words = phrase.split()
        # ignore a lone capitalised word at the start of a sentence
        start = m.start()
        sentence_start = start == 0 or text[:start].rstrip().endswith((".", "!", "?"))
        if len(words) == 1 and (words[0] in _SKIP_START or sentence_start):
            continue
        if words[0] in _SKIP_START:
            words = words[1:]
            phrase = " ".join(words)
            if not phrase:
                continue
        entities.append({"text": phrase, "label": "PROPER_NOUN?"})
    for m in re.finditer(r"\b(?:19|20)\d{2}\b", text):
        entities.append({"text": m.group(0), "label": "DATE?"})
    return entities


def extract_entities(text):
    """Return {'entities': [...], 'method': str}. Never invents entities."""
    if not text or not text.strip():
        return {"entities": [], "method": "none", "note": "No text given."}
    if resources.NLP is not None:
        doc = resources.NLP(text)
        ents = [{"text": e.text, "label": e.label_} for e in doc.ents]
        method = "spaCy en_core_web_sm statistical NER"
        note = None
    else:
        ents = _heuristic_ner(text)
        method = "Heuristic fallback (capitalised words) - no spaCy model installed"
        note = ("Install spaCy and run 'python -m spacy download en_core_web_sm' for real PERSON / ORG / "
                "GPE / DATE labels. Labels ending in '?' are guesses.")
    seen, unique = set(), []
    for e in ents:
        key = (e["text"].lower(), e["label"])
        if key not in seen:
            seen.add(key)
            unique.append(e)
    return {"entities": unique, "method": method, "note": note}
