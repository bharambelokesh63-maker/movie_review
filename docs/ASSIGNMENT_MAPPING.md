# Assignment → File → Function → Webpage

| Assignment | Concept | File (in `modules/`) | Main functions | Where to see it on the website |
|---|---|---|---|---|
| 2 | Text preprocessing, language detection, counts, tokenizing, stop-word removal | `preprocessing.py` | `detect_language`, `split_sentences`, `tokenize`, `remove_punctuation`, `remove_stopwords`, `preprocess`, `text_stats`, `preprocessing_demo` | NLP Lab → **A2 Preprocess**; word/sentence counts on every analysis page |
| 2 | HuffPost-style news CSV statistics, missing-column handling | `preprocessing.py` | `analyze_news_dataset` | NLP Lab → **A2 News CSV** (uses `data/news_sample.csv` if nothing is uploaded) |
| 3 | Porter stemming and WordNet lemmatization | `stemming_lemma.py` | `stem`, `lemmatize`, `compare` | NLP Lab → **A3 Stem / Lemma** |
| 4 | POS tagging (noun, verb, adjective, adverb) | `pos_embeddings.py` | `pos_tag`, `group_by_pos`, `pos_demo` | NLP Lab → **A4 POS** |
| 4 | Word embeddings / semantic similarity | `pos_embeddings.py` | `similarity_demo` (spaCy vectors if available, otherwise TF-IDF cosine, clearly labelled as a substitute) | NLP Lab → **A4 Similarity** |
| 5 | Named Entity Recognition | `ner.py` | `extract_entities` | NLP Lab → **A5 NER**; "Named entities" card on the review result page |
| 6 | TF-IDF keywords | `tfidf_module.py` | `top_keywords`, `tfidf_table` | NLP Lab → **A6 TF-IDF**; "Keywords" card and dashboard keyword chart |
| 7 | N-gram analysis (unigram / bigram / trigram…) | `ngrams_module.py` | `make_ngrams`, `frequent_ngrams`, `ngram_demo` | NLP Lab → **A7 N-grams** (choose n); "word pairs" on result pages |
| 8 | N-gram language model with smoothing | `ngrams_module.py` | class `BigramModel` (`prob`, `sequence_probability`), `language_model_demo` | NLP Lab → **A8 Language Model** |
| 9 | Topic modelling (LDA or LSI) | `topics.py` | `discover_topics` | NLP Lab → **A9 Topics**; "Discovered topics" on batch results and dashboard |
| 10 | Sentiment (VADER, TF-IDF + Logistic Regression), emotions, opinion words, aspects | `sentiment.py` | `vader_compound`, `label_from_score`, `train_model`, `ml_predict`, `detect_emotions`, `opinion_words`, `aspect_opinions` | NLP Lab → **A10 Sentiment**; Home → Analyze Review |
| 11 | Integrated project, evaluation | `analyzer.py`, `sentiment.py`, `visualization.py`, `app.py` | `analyze_review`, `analyze_batch`, `train_model` (accuracy, precision, recall, F1) | Home, Multiple Reviews, Dashboard, **Evaluation** |

Support files: `resources.py` (detects which optional NLP resources are installed), `data_loader.py` (safe CSV and pasted-text reading), `visualization.py` (Matplotlib charts).

## Flask routes

| Route | Purpose | Template |
|---|---|---|
| `/` | Home page and review form | `index.html` |
| `/analyze` (POST) | Analyze one review | `analyze.html` |
| `/batch` | Paste many reviews or upload CSV | `batch.html`, `batch_result.html` |
| `/dashboard` | Statistics of the last batch (or `?sample=1` for the demo file) | `dashboard.html` |
| `/nlp-lab` | Try each assignment concept | `nlp_lab.html` |
| `/evaluate` | Train and evaluate the sentiment model | `evaluate.html` |
