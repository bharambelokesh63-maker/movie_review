"""
app.py - Flask web application for the Movie Review Analyzer & Opinion Mining System.

Run locally:   python app.py        then open http://127.0.0.1:5000
All NLP analysis happens in the Python modules/ folder; JavaScript only handles the page.
"""

import os

from flask import Flask, redirect, render_template, request, url_for

from modules import (analyzer, data_loader, ngrams_module, ner, pos_embeddings, preprocessing,
                     resources, sentiment, stemming_lemma, tfidf_module, topics)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = data_loader.MAX_UPLOAD_MB * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEMO_LABELED = os.path.join(BASE_DIR, "data", "demo_labeled.csv")
SAMPLE_REVIEWS = os.path.join(BASE_DIR, "data", "sample_reviews.csv")
NEWS_SAMPLE = os.path.join(BASE_DIR, "data", "news_sample.csv")

# The dashboard shows the most recent batch analysis (kept in memory while the app runs).
LAST_BATCH = {"result": None}


@app.context_processor
def inject_globals():
    return {"nlp_status": resources.status_table(), "engine": sentiment.engine_name()}


# ------------------------------------------------------------------ Home
@app.route("/")
def home():
    return render_template("index.html")


# ------------------------------------------------------------------ Single review
@app.route("/analyze", methods=["GET", "POST"])
def analyze():
    if request.method == "GET":
        return redirect(url_for("home"))
    movie = request.form.get("movie", "")
    review = request.form.get("review", "")
    result, error = analyzer.analyze_review(movie, review)
    if error:
        return render_template("index.html", error=error, movie=movie, review=review), 400
    return render_template("analyze.html", r=result)


# ------------------------------------------------------------------ Batch
@app.route("/batch", methods=["GET", "POST"])
def batch():
    if request.method == "GET":
        return render_template("batch.html")

    movie = request.form.get("movie", "").strip()
    pasted = request.form.get("reviews", "")
    text_col = request.form.get("text_col", "").strip() or None
    rating_col = request.form.get("rating_col", "").strip() or None
    upload = request.files.get("csv_file")
    form = {"movie": movie, "reviews": pasted, "text_col": text_col or "", "rating_col": rating_col or ""}

    reviews, source = [], "pasted reviews"
    if upload and upload.filename:
        df, err = data_loader.read_csv_bytes(upload.read(), upload.filename)
        if err:
            return render_template("batch.html", error=err, form=form), 400
        reviews, info, err = data_loader.reviews_from_df(df, text_col, rating_col)
        if err:
            return render_template("batch.html", error=err, form=form, columns=list(df.columns)), 400
        source = f"CSV file '{upload.filename}'"
    elif pasted.strip():
        reviews = data_loader.parse_pasted_reviews(pasted)
    else:
        return render_template("batch.html", error="Paste some reviews or choose a CSV file first.", form=form), 400

    result, err = analyzer.analyze_batch(reviews, movie, source)
    if err:
        return render_template("batch.html", error=err, form=form), 400
    LAST_BATCH["result"] = result
    return render_template("batch_result.html", r=result)


# ------------------------------------------------------------------ Dashboard
@app.route("/dashboard")
def dashboard():
    sample_used = False
    if request.args.get("sample") == "1":
        df, err = data_loader.read_csv_path(SAMPLE_REVIEWS)
        if err:
            return render_template("dashboard.html", r=None, error=err)
        reviews, _, err = data_loader.reviews_from_df(df)
        result, err = analyzer.analyze_batch(reviews, "Demo sample (fictional reviews)", "bundled demo file sample_reviews.csv")
        LAST_BATCH["result"] = result
        sample_used = True
    return render_template("dashboard.html", r=LAST_BATCH["result"], sample_used=sample_used)


# ------------------------------------------------------------------ NLP Lab
def _blocks(text):
    return [r["text"] for r in data_loader.parse_pasted_reviews(text)]


def _demo_corpus():
    df, err = data_loader.read_csv_path(DEMO_LABELED)
    if err:
        return []
    return df["review"].astype(str).tolist()


@app.route("/nlp-lab", methods=["GET", "POST"])
def nlp_lab():
    defaults = {
        "text": "The acting was brilliant, but the ending felt rushed. Christopher Nolan directed the film in 2023, "
                "and critics in London praised the studying cinematographers.",
        "text2": "I loved this wonderful film.",
        "n": "2",
    }
    if request.method == "GET":
        return render_template("nlp_lab.html", lab=None, form=defaults)

    concept = request.form.get("concept", "")
    text = request.form.get("text", "")
    form = {**defaults, **{k: v for k, v in request.form.items()}}
    lab = {"concept": concept, "error": None}

    try:
        if concept == "news":
            upload = request.files.get("csv_file")
            if upload and upload.filename:
                df, err = data_loader.read_csv_bytes(upload.read(), upload.filename)
                lab["source"] = f"uploaded file '{upload.filename}'"
            else:
                df, err = data_loader.read_csv_path(NEWS_SAMPLE)
                lab["source"] = "bundled demo file news_sample.csv (fictional headlines, not real HuffPost data)"
            if err:
                lab["error"] = err
            else:
                lab["news"], err = preprocessing.analyze_news_dataset(df)
                lab["error"] = err
        elif not text.strip() and concept not in ("topics", "lm"):
            lab["error"] = "Please enter some text first."
        elif concept == "preprocess":
            lab["pre"] = preprocessing.preprocessing_demo(text)
        elif concept == "stem":
            lab["stem"] = stemming_lemma.compare(text)
            lab["stem_explanation"] = stemming_lemma.EXPLANATION
        elif concept == "pos":
            lab["pos"] = pos_embeddings.pos_demo(text)
        elif concept == "similarity":
            lab["sim"] = pos_embeddings.similarity_demo(text, request.form.get("text2", ""))
            lab["error"] = lab["sim"].get("error")
        elif concept == "ner":
            lab["ner"] = ner.extract_entities(text)
            lab["ner_explanation"] = ner.EXPLANATION
        elif concept == "tfidf":
            lab["tfidf"] = tfidf_module.tfidf_table(_blocks(text))
            lab["tfidf_explanation"] = tfidf_module.EXPLANATION
            lab["error"] = lab["tfidf"].get("error")
        elif concept == "ngram":
            n = max(1, min(int(form.get("n", 2) or 2), 5))
            lab["ngram"] = ngrams_module.ngram_demo(text, n)
            lab["error"] = lab["ngram"].get("error")
        elif concept == "lm":
            train = _blocks(form.get("train_text", ""))
            lab["lm_source"] = "your training text"
            if not train:
                train = _demo_corpus()
                lab["lm_source"] = "the bundled demo reviews (56 short reviews)"
            sentence = form.get("lm_sentence", "").strip() or "the acting was great"
            lab["lm"] = ngrams_module.language_model_demo(train, sentence)
            lab["error"] = lab["lm"].get("error")
        elif concept == "topics":
            docs = _blocks(text)
            lab["topics_source"] = "your pasted reviews"
            if not docs:
                docs = _demo_corpus()
                lab["topics_source"] = "the bundled demo reviews"
            method = form.get("method", "lda")
            lab["topics"] = topics.discover_topics(docs, n_topics=int(form.get("n_topics", 3) or 3), method=method)
            lab["error"] = lab["topics"].get("error")
        elif concept == "sentiment":
            score = sentiment.vader_compound(text)
            lab["sent"] = {
                "score": score, "label": sentiment.label_from_score(score),
                "explanation": sentiment.explain_score(score),
                "opinions": sentiment.opinion_words(text),
                "emotions": sentiment.detect_emotions(text),
                "ml": sentiment.ml_predict(text),
                "disclaimer": sentiment.DISCLAIMER,
            }
        else:
            lab["error"] = "Unknown concept selected."
    except ValueError:
        lab["error"] = "Please check your inputs (numbers must be whole numbers)."
    except Exception:  # last safety net: show a friendly message, never a traceback
        app.logger.exception("NLP lab failure")
        lab["error"] = "Something went wrong while processing this text. Try a different input."
    return render_template("nlp_lab.html", lab=lab, form=form)


# ------------------------------------------------------------------ Evaluation
@app.route("/evaluate", methods=["GET", "POST"])
def evaluate():
    report, error, columns = None, None, None
    if request.method == "POST":
        text_col = request.form.get("text_col", "").strip() or None
        label_col = request.form.get("label_col", "").strip() or None
        upload = request.files.get("csv_file")
        if upload and upload.filename:
            df, error = data_loader.read_csv_bytes(upload.read(), upload.filename)
            source = f"uploaded file '{upload.filename}'"
        else:
            df, error = data_loader.read_csv_path(DEMO_LABELED)
            source = "bundled demo_labeled.csv (56 short demonstration reviews)"
        if not error:
            columns = list(df.columns)
            report, error = sentiment.train_model(df, text_col, label_col, source)
    model = sentiment.load_model()
    return render_template("evaluate.html", report=report, error=error, columns=columns,
                           model_source=model["source"] if model else None)


# ------------------------------------------------------------------ Error pages
@app.errorhandler(404)
def not_found(_):
    return render_template("error.html", title="Page not found", message="That page does not exist."), 404


@app.errorhandler(413)
def too_large(_):
    return render_template("error.html", title="File too large",
                           message=f"Uploads must be smaller than {data_loader.MAX_UPLOAD_MB} MB."), 413


@app.errorhandler(500)
def server_error(_):
    return render_template("error.html", title="Something went wrong",
                           message="The server hit an unexpected problem. Please try again with different input."), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
