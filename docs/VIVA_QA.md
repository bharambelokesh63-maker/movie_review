# Viva Questions and Answers

## A. Project and workflow

**Q1. What does your project do?**
It is a Flask website where a user enters a movie name and a review. The Python backend cleans the text, finds sentiment (positive / negative / neutral), opinion words, keywords, emotions and named entities, and shows a short summary. It can also analyze many reviews and show statistics and charts.

**Q2. Explain the workflow from input to output.**
Browser form → Flask route (`/analyze`) → `analyzer.analyze_review()` → preprocessing (tokenize, remove punctuation and stop words) → sentiment (VADER) → opinion words, emotions, NER, TF-IDF keywords → result dictionary → Jinja2 template → HTML page.

**Q3. Why is the analysis done in Python and not JavaScript?**
NLP libraries (NLTK, scikit-learn) are in Python. JavaScript only handles page behaviour such as the loading spinner and tabs.

**Q4. Does the application show what the public thinks of a movie?**
No. It describes only the reviews the user provides. With a handful of reviews the result is not representative of all viewers; the application says this on screen.

**Q5. What happens if a library or model is missing?**
`resources.py` checks what is installed. Each module has a simpler fallback (regex tokenizer, scikit-learn stop-word list, rule-based lemmatizer, heuristic tagger/NER) and labels the output as a fallback.

## B. Preprocessing (Assignment 2)

**Q6. What is tokenization?** Splitting text into units (words or sentences). Example: "I loved it!" → `I`, `loved`, `it`, `!`.

**Q7. What are stop words and why remove them?** Very common words (the, is, and) that carry little meaning. Removing them lets frequency counts and TF-IDF focus on informative words. We keep negations such as "not" because "not good" means the opposite of "good".

**Q8. How does language detection work here?** First the writing script is checked (e.g. Devanagari). Then `langdetect` is used if installed; otherwise the number of common words from a few languages is counted. It is an estimate and is unreliable for very short text.

## C. Stemming and lemmatization (Assignment 3)

**Q9. Difference between stemming and lemmatization?** Stemming cuts endings with rules and may give non-words ("studies" → "studi"). Lemmatization uses a dictionary (WordNet) and returns a real base word ("studies" → "study", "better" → "good"). Stemming is faster; lemmatization is more accurate.

**Q10. Which stemmer did you use?** Porter stemmer (`nltk.stem.PorterStemmer`), a rule-based suffix stripping algorithm.

**Q11. Why does lemmatization need POS information?** The same word can have different lemmas by role: "saw" is "see" as a verb and "saw" (the tool) as a noun. Our function tries verb first and then noun.

## D. POS tagging and embeddings (Assignment 4)

**Q12. What is POS tagging?** Labelling each word with its grammatical role (noun, verb, adjective, adverb…). It helps find opinion words, because adjectives like "boring" or "brilliant" often carry sentiment.

**Q13. How did you demonstrate word embeddings?** If a spaCy model with vectors is installed, similarity uses real embeddings. Otherwise TF-IDF cosine similarity is shown and **labelled as a substitute**, because TF-IDF only matches identical words ("great" and "excellent" score 0), while embeddings capture meaning.

**Q14. What is cosine similarity?** The cosine of the angle between two vectors: 1 means same direction (very similar), 0 means nothing in common.

## E. Named Entity Recognition (Assignment 5)

**Q15. What is NER?** Finding names of people (PERSON), organisations (ORG), places (GPE), dates (DATE) etc. in text.

**Q16. Why can NER be wrong?** It is a statistical model, so it depends on the training data and the context ("Apple" the fruit vs the company). Without spaCy the project uses a simple capitalised-word heuristic and marks results with "?".

## F. TF-IDF (Assignment 6)

**Q17. Explain TF-IDF.** TF = how often a word appears in a document. IDF = log(total documents / documents containing the word) (scikit-learn uses a smoothed version). TF × IDF is high for words frequent in one review but rare in others, so they are informative keywords.

**Q18. Why not just count words?** Common words ("movie") would dominate. TF-IDF down-weights words that appear in every review.

**Q19. What if only one review is given?** There are no other documents to compare with, so IDF is the same for all words and scores reflect only frequency within that review.

## G. N-grams and language model (Assignments 7 and 8)

**Q20. What is an n-gram?** A sequence of n consecutive words: unigram (1), bigram (2), trigram (3). Bigrams like "not good" capture context that single words lose.

**Q21. How does the bigram language model compute probability?**
P(word | previous) = count(previous, word) / count(previous). A sentence probability is the product of these values, starting from `<s>` and ending with `</s>`.

**Q22. Why smoothing? Which did you use?** An unseen word pair would get probability 0 and make the whole sentence 0. Add-one (Laplace) smoothing uses (count(prev, word) + 1) / (count(prev) + V), where V is the vocabulary size.

**Q23. How are unknown words handled?** They are mapped to `<unk>`, which is part of the vocabulary, so they receive a small non-zero probability.

**Q24. What is perplexity?** exp(−average log probability per word). Lower perplexity means the model finds the text less surprising.

## H. Topic modelling (Assignment 9)

**Q25. What is LDA?** Latent Dirichlet Allocation assumes each document is a mixture of topics and each topic is a distribution over words. It discovers word groups that occur together. LSI uses SVD on a TF-IDF matrix instead.

**Q26. Why does your app sometimes say it cannot find topics?** LDA needs enough text. With fewer than 4 non-empty reviews the app shows a message instead of a misleading result.

**Q27. Are the topics labelled automatically?** No. The model only gives word lists; a human interprets them.

## I. Sentiment and evaluation (Assignments 10 and 11)

**Q28. How does VADER work?** It is a lexicon- and rule-based analyzer: each word has a valence score, and rules handle negation ("not good"), intensifiers ("very good"), punctuation and capitalisation. The compound score ranges from −1 to +1; we label ≥ 0.05 Positive, ≤ −0.05 Negative, otherwise Neutral.

**Q29. Is the compound score a probability?** No. It measures how strongly the words lean positive or negative.

**Q30. How does the machine-learning model work?** Reviews are converted to TF-IDF vectors (unigrams and bigrams) and a Logistic Regression classifier is trained on labelled reviews. It outputs class probabilities.

**Q31. How did you evaluate it?** The labelled data is split 75% training / 25% testing (stratified). Accuracy, precision, recall and F1 are computed on the unseen test part, and a confusion matrix is shown. VADER is evaluated on the same test reviews for comparison.

**Q32. Define accuracy, precision, recall and F1.** Accuracy = correct / all. Precision = true positives / predicted positives. Recall = true positives / actual positives. F1 = 2·P·R / (P + R).

**Q33. Is your model accurate?** Only the evaluation page can say. The bundled demo set has just 56 short reviews, so its metrics are rough and can vary with the split. In our own run VADER scored higher than the model trained on this tiny set. A real dataset (thousands of reviews, e.g. IMDb) is needed for a trustworthy model.

**Q34. How do you detect emotions?** A simple rule-based method: lists of cue words for happiness, excitement, sadness, anger, fear and disappointment are matched in the text. It is not a trained emotion model and can miss sarcasm or context.

**Q35. Limitations of sentiment analysis?** Sarcasm, mixed opinions, slang, domain words (e.g. "sick" or "dark" in films), negation far from the word, and non-English text.

**Q36. What does the aspect table show?** For topics such as acting, plot, music, ending, it finds sentences mentioning them and averages their VADER scores. This is a basic form of aspect-based opinion mining.

## J. Libraries

**Q37. Why Flask?** A light Python web framework: routes map URLs to Python functions, and Jinja2 templates build HTML.

**Q38. Role of each library?** NLTK: tokenizing, stemming, lemmatizing, n-grams, POS. scikit-learn: TF-IDF, Logistic Regression, LDA/SVD, metrics. pandas: reading CSVs. Matplotlib: charts. spaCy (optional): NER and POS. vaderSentiment: sentiment lexicon.

**Q39. Why scikit-learn for topic modelling instead of Gensim?** It gives LDA and LSI (via `LatentDirichletAllocation` and `TruncatedSVD`) with fewer installation problems; the ideas are the same.
