# Detecting Fake News With Machine Learning

B.Sc. graduation project, Computer Science / Artificial Intelligence,
University of Technology, Baghdad (2022–2023). Final mark 99/100.
Presented at the 6th AUE Student Research Competition, American University
in the Emirates, September 2023.

Author and project lead: Ahmed Thamer Lateef, with Sarah Hayder and Zahraa Adel.
Supervisor: Prof. Dr. Hanaa Mohsin Ahmed.

The program classifies a news article as fake or genuine from its text, using
TF-IDF features and a Multinomial Naive Bayes classifier.

## Files

| File | What it is |
|---|---|
| `main_2023.py` | The original code, as submitted in 2023 |
| `main.py` | 2026 update: adds 5-fold cross-validation, class probabilities and an input loop, as described in the report, plus a fixed random seed |

## Results (`main.py`, seed 42, 70/30 split)

| Measure | Value |
|---|---|
| Cross-validation accuracy (5 folds, training set) | 93.7% ± 0.3% |
| Test accuracy (13,482 articles) | 93.5% |
| Fake news: precision / recall | 93% / 95% |
| Genuine news: precision / recall | 94% / 92% |

The 2023 report gives 92.5% test accuracy. Re-running `main_2023.py` on five
random splits in 2026 gave 93.4–93.9%.

## Run it

1. Download `Fake.csv` and `True.csv` from the Kaggle
   [Fake and Real News dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
   into this folder.
2. Install the requirements and run:

```bash
pip install -r requirements.txt
python main.py
```

Type an article to classify it, or `exit` to stop.

## Limitations

- Naive Bayes assumes words are independent of each other.
- Many genuine articles in this dataset come from Reuters and share its style
  (for example, they open with "WASHINGTON (Reuters)"). The model can learn
  that style instead of the truth of the content, so accuracy on news from
  other sources would likely be lower.
