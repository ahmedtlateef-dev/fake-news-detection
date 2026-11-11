import pandas as pd
import numpy as np
import time
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# load data
start_time = time.time()

df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")
df_fake["class"] = 0
df_true["class"] = 1
df = pd.concat([df_fake, df_true], axis=0).reset_index(drop=True)

# preprocess data
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub("\\W", " ", text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        return text
    else:
        return ""

df["text"] = df["text"].apply(preprocess_text)

x_train, x_test, y_train, y_test = train_test_split(df["text"], df["class"], test_size=0.3)

# vectorize text data
vectorizer = TfidfVectorizer()
xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)

# train Naive Bayes model and measure the training time
nb = MultinomialNB()
nb.fit(xv_train, y_train)
end_time = time.time()
training_time = end_time - start_time

# evaluate model
y_pred = nb.predict(xv_test)
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# make manual predictions
def predict_news(text):
    # preprocess and vectorize text data
    processed_text = preprocess_text(text)
    xv = vectorizer.transform([processed_text])
    # make prediction
    pred = nb.predict(xv)[0]
    return "Fake News" if pred == 0 else "Not A Fake News"

news = input("Enter a news article: ")
print(predict_news(news))
print(f"Training Time: {training_time:.2f} seconds")
