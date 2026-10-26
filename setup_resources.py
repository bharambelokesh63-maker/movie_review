"""
setup_resources.py - OPTIONAL one-time download of NLP resources.

Run:  python setup_resources.py
Every download is optional: if one fails (e.g. no internet), the website still works
and shows which resource is missing on the NLP Lab page.
"""

import subprocess
import sys

import nltk

NLTK_PACKAGES = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4",
                 "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng", "vader_lexicon"]

print("== NLTK resources ==")
for pkg in NLTK_PACKAGES:
    try:
        ok = nltk.download(pkg, quiet=True)
    except Exception as exc:  # network or permission problems
        ok = False
        print("   error:", exc)
    print(f"  {pkg:35s}", "OK" if ok else "FAILED (optional)")

print("\n== spaCy model (optional, for real NER and POS tagging) ==")
try:
    import spacy  # noqa: F401
    result = subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                            capture_output=True, text=True)
    print("  en_core_web_sm", "OK" if result.returncode == 0 else "FAILED (optional)")
    if result.returncode != 0:
        print(result.stderr[-300:])
except ImportError:
    print("  spaCy is not installed. Install it with: pip install spacy   (optional)")

print("\nDone. Start the website with: python app.py")
