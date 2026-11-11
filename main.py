# Fake news detection with Multinomial Naive Bayes.
# 2026 update of main_2023.py: adds the three steps the project report
# describes (cross-validation, class probabilities, an input loop) and a
# fixed random seed so results can be reproduced.

import re
import string
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline

SEED = 42

df_fake = pd.read_csv("Fake.csv", low_memory=False)
df_true = pd.read_csv("True.csv", low_memory=False)
df_fake["class"] = 0
df_true["class"] = 1
df = pd.concat([df_fake, df_true], axis=0).reset_index(drop=True)


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\n", "", text)
    text = re.sub(r"\w*\d\w*", "", text)
    return text


df["text"] = df["text"].apply(preprocess_text)

x_train, x_test, y_train, y_test = train_test_split(
    df["text"], df["class"], test_size=0.3, random_state=SEED
)

# The vectorizer sits inside the pipeline so each fold fits TF-IDF on its own
# training part only; fitting it on all training data first would leak.
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

start_time = time.time()
cv_scores = cross_val_score(model, x_train, y_train, cv=5)
training_time = time.time() - start_time
print(f"Training time (5-fold cross-validation): {training_time:.2f} seconds")
print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(f"Test accuracy: {np.mean(y_pred == y_test):.4f}")
print(classification_report(y_test, y_pred, target_names=["Fake", "Genuine"]))

while True:
    news = input("Enter a news article (or 'exit' to stop): ")
    if news.strip().lower() == "exit":
        break
    p_fake, p_genuine = model.predict_proba([preprocess_text(news)])[0]
    label = "Fake News" if p_fake > p_genuine else "Not A Fake News"
    print(f"{label} (fake: {p_fake:.1%}, genuine: {p_genuine:.1%})")
