# 🎬 Movie Review Analyzer & Opinion Mining System

A beginner-friendly Flask website that covers the NLP lab assignments 2–11. Enter a movie name and a review (or many reviews, or a CSV) and the Python backend reports sentiment, opinion words, keywords, emotions, named entities, n-grams, topics and charts.

> **Honest scope:** results describe only the text you provide, not "all public opinion". Sentiment and emotion outputs are estimates and can be wrong.

---

## 1. Folder structure

```
movie_review_analyzer/
├── app.py                  Flask routes
├── requirements.txt
├── setup_resources.py      optional NLTK / spaCy downloads
├── smoke_test.py           checks every page and error case
├── modules/                one file per NLP topic
│   ├── resources.py        detects installed NLP resources (fallbacks)
│   ├── preprocessing.py    Assignment 2
│   ├── stemming_lemma.py   Assignment 3
│   ├── pos_embeddings.py   Assignment 4
│   ├── ner.py              Assignment 5
│   ├── tfidf_module.py     Assignment 6
│   ├── ngrams_module.py    Assignments 7 and 8
│   ├── topics.py           Assignment 9
│   ├── sentiment.py        Assignment 10
│   ├── analyzer.py         Assignment 11 (integration)
│   ├── visualization.py    Matplotlib charts
│   └── data_loader.py      safe CSV / pasted-text reading
├── templates/              HTML (Jinja2)
├── static/css/style.css    styling
├── static/js/main.js       spinner, tabs, character counter
├── data/                   demo_labeled.csv, sample_reviews.csv, news_sample.csv
├── models/                 trained model is saved here (created on demand)
├── docs/                   ASSIGNMENT_MAPPING.md, VIVA_QA.md
└── Colab_Run.ipynb         ready-to-run Google Colab notebook
```

## 2. Run on your own computer

Requires Python 3.9 or newer.

```bash
cd movie_review_analyzer
python -m venv venv
# Windows:   venv\Scripts\activate
# Mac/Linux: source venv/bin/activate

pip install -r requirements.txt
python setup_resources.py      # optional but recommended (NLTK data); see section 6
python smoke_test.py           # optional: should print ALL PASSED
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

Optional, for real NER / POS tags:

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

## 3. Run in Google Colab

Easiest: upload this project as a zip and open **`Colab_Run.ipynb`**, or follow these cells manually.

**Cell 1 – upload and unzip**
```python
from google.colab import files
files.upload()                       # choose movie_review_analyzer.zip
!unzip -q -o movie_review_analyzer.zip
%cd movie_review_analyzer
```

**Cell 2 – install**
```python
!pip install -q -r requirements.txt
!python setup_resources.py
```

**Cell 3 – start Flask and open the site**
```python
from threading import Thread
from app import app

Thread(target=lambda: app.run(host="0.0.0.0", port=5000, use_reloader=False), daemon=True).start()

from google.colab import output
output.serve_kernel_port_as_window(5000)     # click the link that appears
```

If the link does not open (browser pop-up blocker, or a Colab change), use a tunnel instead:

```python
# Option A: Cloudflare quick tunnel (no account needed)
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
!chmod +x cloudflared
import subprocess, re, time
proc = subprocess.Popen(["./cloudflared", "tunnel", "--url", "http://localhost:5000"],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    m = re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", line)
    if m:
        print("Open:", m.group(0)); break
```
```python
# Option B: ngrok (free account + authtoken from https://dashboard.ngrok.com)
!pip install -q pyngrok
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_TOKEN")
print(ngrok.connect(5000))
```

Keep the Colab cell/runtime running while you use the website.

## 4. Using the website

| Page | What to do |
|---|---|
| **Home** | Type a movie name, paste a review, press *Analyze Review* (or *Use example review*). |
| **Multiple Reviews** | Paste reviews separated by a blank line, **or** upload a CSV. |
| **Dashboard** | Shows stats and charts for the last batch. *Load demo reviews* uses `data/sample_reviews.csv` (fictional demo data). |
| **NLP Lab** | One tab per assignment concept with input, output and explanation. |
| **Evaluation** | Train TF-IDF + Logistic Regression on a labelled CSV (or the demo file) and see accuracy, precision, recall, F1 and a confusion matrix. |

### Dataset upload instructions
- **Local:** use the file chooser on the page.
- **Colab:** open the website through the tunnel link and use the same file chooser (the file goes from your computer to the Flask server). Alternatively, upload a file in the Colab file panel and place it in `data/`.

## 5. Dataset formats

**Reviews for the Multiple Reviews page** (`review` or `text` column; `rating` is optional and is averaged only if present):
```csv
review,rating
"Great acting and a beautiful soundtrack.",9
"Boring and too long.",3
```

**Labelled data for the Evaluation page** (text column + `sentiment` or `label`; values positive/negative/neutral, pos/neg or 1/0):
```csv
review,sentiment
"A wonderful film with brilliant acting.",positive
"Awful script and dull characters.",negative
```

**News data for Assignment 2** (HuffPost style):
```csv
category,headline,short_description
POLITICS,"City council approves new budget","Local officials voted on Tuesday..."
```

Different column names? Type them in the "Column names differ?" boxes on the page. Missing columns, empty files and unreadable files produce a clear message instead of a crash.

**Included sample files:** `data/demo_labeled.csv` (56 short original reviews for testing), `data/sample_reviews.csv` (20 fictional reviews with ratings), `data/news_sample.csv` (15 fictional headlines). These are demonstration data written for this project, not real HuffPost or IMDb data.

**For a real evaluation,** download a larger labelled dataset such as the IMDb 50K movie reviews CSV (columns `review`, `sentiment`) from Kaggle and upload it on the Evaluation page (the app reads up to 5000 rows).

## 6. Optional NLP resources and fallbacks

| Resource | Install | If missing |
|---|---|---|
| NLTK data (punkt, stopwords, wordnet, tagger) | `python setup_resources.py` | regex tokenizer, scikit-learn stop-words, rule-based lemmatizer, heuristic tagger (all labelled on screen) |
| VADER | `pip install vaderSentiment` (in requirements) | small built-in word list, less accurate |
| spaCy `en_core_web_sm` | `pip install spacy && python -m spacy download en_core_web_sm` | capitalised-word heuristic for entities, labels end with "?" |
| spaCy `en_core_web_md` | `python -m spacy download en_core_web_md` | similarity uses TF-IDF (labelled as a substitute, not embeddings) |

The NLP Lab page ends with a table showing which of these are installed.

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'flask'` (or nltk, sklearn…) | Activate your virtual environment, then `pip install -r requirements.txt`. |
| `LookupError: Resource punkt not found` | Run `python setup_resources.py`. The app still works with fallbacks. |
| `nltk.download` fails with SSL or "Security Violation" / proxy errors | Corporate or sandbox proxy. Download on another network, or use the app with fallbacks. |
| `pip install langdetect` fails to build | Skip it; it is optional. |
| spaCy install or model download fails | Skip it; entities use the heuristic fallback. On Colab spaCy is usually pre-installed: just run the model download. |
| `numpy` / `scikit-learn` version conflicts | `pip install --upgrade numpy scikit-learn pandas matplotlib`, then restart the runtime (Colab). |
| Port 5000 already in use | `PORT=5001 python app.py` (Windows PowerShell: `$env:PORT=5001; python app.py`). On macOS, turn off AirPlay Receiver or use another port. |
| Colab link does not open | Allow pop-ups, or use the Cloudflare/ngrok option in section 3. |
| "Topic modelling needs at least 4 reviews" | Provide more reviews; topic models need enough text. |
| Evaluation says each class needs 2 reviews | Add more labelled rows for every class. |

## 8. Assignment mapping and viva

See `docs/ASSIGNMENT_MAPPING.md` (assignment → file → function → webpage) and `docs/VIVA_QA.md` (39 viva questions with answers).

| Assignment | File | Webpage |
|---|---|---|
| 2 Preprocessing, news CSV | `preprocessing.py` | NLP Lab: A2 Preprocess, A2 News CSV |
| 3 Stemming, lemmatization | `stemming_lemma.py` | NLP Lab: A3 |
| 4 POS, embeddings/similarity | `pos_embeddings.py` | NLP Lab: A4 POS, A4 Similarity |
| 5 NER | `ner.py` | NLP Lab: A5; result page |
| 6 TF-IDF | `tfidf_module.py` | NLP Lab: A6; result page |
| 7 N-grams | `ngrams_module.py` | NLP Lab: A7 |
| 8 N-gram language model | `ngrams_module.py` (`BigramModel`) | NLP Lab: A8 |
| 9 Topic modelling | `topics.py` | NLP Lab: A9; batch results |
| 10 Sentiment, emotions | `sentiment.py` | Home; NLP Lab: A10 |
| 11 Integrated project | `analyzer.py`, `app.py` | All pages, Evaluation |

## 9. Notes for your demonstration

- Sentiment default: **VADER**. After you train on the Evaluation page, single-review results also show the trained model's prediction.
- Topic modelling uses scikit-learn (LDA / LSI) instead of Gensim to avoid install problems.
- Metrics appear only after a real train/test run. On the tiny demo file they are rough; say so in the viva.
