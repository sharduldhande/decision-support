"""Check whether the excerpts still give away CABG vs PCI.

Fits a bag-of-words logistic regression with 5-fold cross-validation on
validation/data/excerpts.jsonl. Accuracy near 50% means the excerpts carry little
label signal beyond the anatomy; accuracy well above that means something leaks.
Prints scores and the most predictive words, never excerpt text.
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score

DATA = Path(__file__).parent / "data"


def main(path=DATA / "excerpts.jsonl"):
    rows = [json.loads(line) for line in open(path)]
    texts = [r.get("excerpt_raw", r["excerpt"]) for r in rows]   # without the age/sex header
    y = np.array([r["label"] == "CABG" for r in rows], dtype=int)

    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, token_pattern=r"(?u)\b[A-Za-z][A-Za-z]+\b")
    X = vec.fit_transform(texts)
    model = LogisticRegression(C=1.0, max_iter=2000)

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    proba = cross_val_predict(model, X, y, cv=folds, method="predict_proba")[:, 1]
    print(f"cases: {len(rows)}  (CABG {y.sum()}, PCI {len(y) - y.sum()})")
    print(f"cross-validated accuracy: {accuracy_score(y, proba > 0.5):.3f}")
    print(f"cross-validated AUC:      {roc_auc_score(y, proba):.3f}")

    by_source = {}
    for r, p, t in zip(rows, proba, y):
        by_source.setdefault(r["source"], []).append((p > 0.5) == t)
    for source, hits in sorted(by_source.items()):
        print(f"  accuracy on {source:11} {np.mean(hits):.3f}  (n={len(hits)})")

    model.fit(X, y)
    words = np.array(vec.get_feature_names_out())
    order = np.argsort(model.coef_[0])
    print("words pointing to PCI: ", ", ".join(words[order[:20]]))
    print("words pointing to CABG:", ", ".join(words[order[-20:][::-1]]))


if __name__ == "__main__":
    main(*sys.argv[1:])
