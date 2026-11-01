"""Train and evaluate support-ticket category classifiers."""
from pathlib import Path
import json
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report, ConfusionMatrixDisplay,
                             f1_score, precision_score, recall_score)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data import load_dataset
from src.preprocessing import clean_text

MODEL_DIR, REPORT_DIR = ROOT / "models", ROOT / "reports"


def build_pipeline(classifier):
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, ngram_range=(1, 2),
                                  min_df=2, max_features=30000, sublinear_tf=True,
                                  strip_accents="unicode")),
        ("classifier", classifier),
    ])


def train_and_evaluate(train: pd.DataFrame, test: pd.DataFrame):
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=3.0),
        "Linear SVM": LinearSVC(class_weight="balanced", C=1.0),
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
    }
    results, fitted = {}, {}
    y_train, y_test = train["category"], test["category"]
    for name, estimator in candidates.items():
        pipeline = build_pipeline(estimator)
        pipeline.fit(train["text"], y_train)
        predicted = pipeline.predict(test["text"])
        results[name] = {
            "accuracy": float(accuracy_score(y_test, predicted)),
            "precision_macro": float(precision_score(y_test, predicted, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, predicted, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, predicted, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(y_test, predicted, average="weighted", zero_division=0)),
        }
        fitted[name] = (pipeline, predicted)
    best_name = max(results, key=lambda name: results[name]["f1_macro"])
    best_model, best_predictions = fitted[best_name]
    return results, best_name, best_model, y_test, best_predictions


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)
    train, test = load_dataset(download=True)
    results, best_name, best_model, y_test, best_predictions = train_and_evaluate(train, test)
    labels = sorted(set(y_test) | set(best_predictions))
    report = classification_report(y_test, best_predictions, output_dict=True, zero_division=0)
    metrics = {
        "dataset": "Zenodo Classification of IT Support Tickets (DOI: 10.5281/zenodo.7648117)",
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "category_count": int(train["category"].nunique()),
        "categories": sorted(train["category"].unique().tolist()),
        "best_model": best_name, "selection_metric": "macro F1",
        "models": results,
        "priority_method": "Explainable keyword rules; no ground-truth priority labels in this dataset.",
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (REPORT_DIR / "classification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    joblib.dump(best_model, MODEL_DIR / "category_classifier.joblib")
    joblib.dump({"model_name": best_name, "categories": labels}, MODEL_DIR / "metadata.joblib")

    fig, ax = plt.subplots(figsize=(9, 7))
    ConfusionMatrixDisplay.from_predictions(y_test, best_predictions, labels=labels,
                                            xticks_rotation=35, cmap="Blues", ax=ax,
                                            colorbar=False)
    ax.set_title(f"Ticket category confusion matrix — {best_name}")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "category_confusion_matrix.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    score_frame = pd.DataFrame(results).T.sort_values("f1_macro", ascending=True)
    score_frame["f1_macro"].plot(kind="barh", ax=ax)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro F1-score")
    ax.set_title("Model comparison on the held-out test set")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "model_comparison.png", dpi=180)
    plt.close(fig)

    counts = pd.concat([train["category"], test["category"]]).value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(9, 5))
    counts.plot(kind="barh", ax=ax)
    ax.set_xlabel("Number of tickets")
    ax.set_title("Ticket category distribution")
    fig.tight_layout()
    fig.savefig(REPORT_DIR / "category_distribution.png", dpi=180)
    plt.close(fig)

    print(json.dumps(metrics, indent=2))
    print("\nClassification report:\n", classification_report(y_test, best_predictions, zero_division=0))
    print(f"\nSaved best model: {MODEL_DIR / 'category_classifier.joblib'}")


if __name__ == "__main__":
    main()
