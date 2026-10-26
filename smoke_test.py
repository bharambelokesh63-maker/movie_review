"""
smoke_test.py - Quick check that every page and error case works.
Run:  python smoke_test.py
"""

import io
import os

from app import app

client = app.test_client()
failures = []


def check(name, response, expect_status=200, must_contain=(), must_not_contain=("Traceback",)):
    body = response.get_data(as_text=True)
    ok = response.status_code == expect_status
    ok &= all(s in body for s in must_contain)
    ok &= not any(s in body for s in must_not_contain)
    print(("PASS" if ok else "FAIL"), name, response.status_code)
    if not ok:
        failures.append(name)
        print("   expected", expect_status, "and", must_contain)
        print("   body starts:", body[:300].replace("\n", " "))


# ---- pages
for path, word in [("/", "Analyze a review"), ("/batch", "Analyze multiple"), ("/dashboard", "No reviews analyzed"),
                   ("/nlp-lab", "NLP Learning Lab"), ("/evaluate", "evaluation")]:
    check("GET " + path, client.get(path), must_contain=[word])
check("GET /analyze redirects", client.get("/analyze"), 302)
check("404 page", client.get("/nope"), 404, ["Page not found"])

# ---- single review
good = "The acting was brilliant and the soundtrack is beautiful, but the ending was a disappointment."
check("analyze good review", client.post("/analyze", data={"movie": "Test", "review": good}),
      must_contain=["Positive", "Opinion words", "Named entities"])
check("analyze negative", client.post("/analyze", data={"movie": "T", "review": "Terrible, boring and awful. I hated it."}),
      must_contain=["Negative"])
check("analyze empty", client.post("/analyze", data={"movie": "T", "review": "   "}), 400, ["Please write"])
check("analyze very short", client.post("/analyze", data={"movie": "T", "review": "good"}), must_contain=["very short"])
check("analyze symbols only", client.post("/analyze", data={"movie": "T", "review": "!!! ???"}), 400)
check("analyze hindi", client.post("/analyze", data={"movie": "T", "review": "यह फिल्म बहुत अच्छी थी और कहानी सुंदर थी"}),
      must_contain=["Devanagari"])
check("analyze html injection", client.post("/analyze", data={"movie": "<script>x</script>", "review": good}),
      must_contain=["&lt;script&gt;"], must_not_contain=["<script>x</script>"])

# ---- batch
check("batch pasted", client.post("/batch", data={"movie": "M", "reviews": "Great film, loved it.\n\nAwful and boring.\n\nIt was okay."}),
      must_contain=["reviews analyzed", "Sentiment distribution"])
check("batch empty", client.post("/batch", data={"reviews": ""}), 400, ["Paste some reviews"])
sample = open(os.path.join("data", "sample_reviews.csv"), "rb").read()
check("batch CSV with ratings", client.post("/batch", data={"csv_file": (io.BytesIO(sample), "sample_reviews.csv")},
                                             content_type="multipart/form-data"), must_contain=["average rating"])
check("batch CSV no rating col", client.post("/batch", data={"csv_file": (io.BytesIO(b"review\nGood film\nBad film\n"), "a.csv")},
                                              content_type="multipart/form-data"), must_contain=["No numeric ratings"])
check("batch CSV wrong column", client.post("/batch", data={"csv_file": (io.BytesIO(b"foo,bar\n1,2\n"), "a.csv")},
                                             content_type="multipart/form-data"), 400, ["Could not find a review column"])
check("batch CSV empty file", client.post("/batch", data={"csv_file": (io.BytesIO(b""), "a.csv")},
                                           content_type="multipart/form-data"), 400, ["empty"])
check("batch wrong extension", client.post("/batch", data={"csv_file": (io.BytesIO(b"x"), "a.pdf")},
                                            content_type="multipart/form-data"), 400, ["upload a .csv"])
check("dashboard after batch", client.get("/dashboard"), must_contain=["Movie Review Dashboard"])
check("dashboard demo", client.get("/dashboard?sample=1"), must_contain=["demo"])

# ---- NLP lab
lab_text = "The acting was brilliant, but the ending felt rushed. Christopher Nolan directed the film in 2023."
for concept, word in [("preprocess", "Clean tokens"), ("stem", "Porter stem"), ("pos", "Method"), ("ner", "Method"),
                      ("ngram", "grams generated"), ("sentiment", "score")]:
    check("lab " + concept, client.post("/nlp-lab", data={"concept": concept, "text": lab_text, "n": "2"}), must_contain=[word])
check("lab similarity", client.post("/nlp-lab", data={"concept": "similarity", "text": "great film", "text2": "great movie"}),
      must_contain=["Similarity"])
check("lab similarity missing B", client.post("/nlp-lab", data={"concept": "similarity", "text": "great film", "text2": ""}),
      must_contain=["enter both"])
check("lab tfidf", client.post("/nlp-lab", data={"concept": "tfidf", "text": "good acting\n\nbad plot\n\nnice music"}),
      must_contain=["Overall top keywords"])
check("lab lm demo corpus", client.post("/nlp-lab", data={"concept": "lm", "lm_sentence": "the acting was great xyzunknown"}),
      must_contain=["Sentence probability", "unknown word"])
check("lab lm own text", client.post("/nlp-lab", data={"concept": "lm", "train_text": "I like movies. I like music.", "lm_sentence": "i like movies"}),
      must_contain=["Sentence probability"])
check("lab topics demo", client.post("/nlp-lab", data={"concept": "topics", "text": "", "method": "lda", "n_topics": "3"}),
      must_contain=["Topic 1"])
check("lab topics lsi", client.post("/nlp-lab", data={"concept": "topics", "text": "", "method": "lsi", "n_topics": "3"}),
      must_contain=["LSI"])
check("lab topics too small", client.post("/nlp-lab", data={"concept": "topics", "text": "one review only"}),
      must_contain=["at least 4"])
check("lab empty text", client.post("/nlp-lab", data={"concept": "ner", "text": ""}), must_contain=["enter some text"])
check("lab ngram too short", client.post("/nlp-lab", data={"concept": "ngram", "text": "hi", "n": "3"}), must_contain=["Need at least"])
check("lab bad n", client.post("/nlp-lab", data={"concept": "ngram", "text": "a b c d", "n": "abc"}), must_contain=["whole numbers"])
check("lab news demo", client.post("/nlp-lab", data={"concept": "news"}), must_contain=["avg words"])
check("lab news bad column", client.post("/nlp-lab", data={"concept": "news", "csv_file": (io.BytesIO(b"a,b\n1,2\n"), "x.csv")},
                                          content_type="multipart/form-data"), must_contain=["Could not find a text column"])
check("lab unknown concept", client.post("/nlp-lab", data={"concept": "zzz", "text": "x"}), must_contain=["Unknown concept"])

# ---- evaluation
check("evaluate demo data", client.post("/evaluate", data={}), must_contain=["accuracy", "Confusion matrix"])
check("analyze shows ML after training", client.post("/analyze", data={"movie": "T", "review": good}), must_contain=["Trained model"])
tiny = b"review,sentiment\ngood,positive\nbad,negative\n"
check("evaluate too few rows", client.post("/evaluate", data={"csv_file": (io.BytesIO(tiny), "t.csv")},
                                            content_type="multipart/form-data"), must_contain=["at least 10"])
nolabel = b"review\n" + b"a nice film\n" * 12
check("evaluate no label column", client.post("/evaluate", data={"csv_file": (io.BytesIO(nolabel), "t.csv")},
                                               content_type="multipart/form-data"), must_contain=["No label column"])
one_class = b"review,label\n" + b"a nice film,positive\n" * 12
check("evaluate one class", client.post("/evaluate", data={"csv_file": (io.BytesIO(one_class), "t.csv")},
                                         content_type="multipart/form-data"), must_contain=["only one class"])
check("evaluate wrong col name", client.post("/evaluate", data={"text_col": "nope", "csv_file": (io.BytesIO(tiny), "t.csv")},
                                              content_type="multipart/form-data"), must_contain=["not found"])

print()
print("ALL PASSED" if not failures else f"{len(failures)} FAILED: {failures}")
raise SystemExit(1 if failures else 0)
