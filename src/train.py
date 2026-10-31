"""Train an NLP ticket-category baseline on a generated demo corpus."""
from pathlib import Path
import json
import re

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR, REPORT_DIR = ROOT / "models", ROOT / "reports"
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

TEMPLATES = {
    "Account Access": [
        "I cannot log in to my account", "My password reset link is not working",
        "Please help me unlock my account", "Two factor authentication code is missing",
        "I am unable to sign in after changing my password",
    ],
    "Billing": [
        "I was charged twice for my subscription", "Please explain this invoice amount",
        "My payment was declined but money was deducted", "I need a refund for my last purchase",
        "The billing address on my receipt is incorrect",
    ],
    "Delivery": [
        "My order has not arrived yet", "The package tracking has not updated",
        "I received the wrong item in my delivery", "Please provide an update on shipment",
        "The parcel arrived damaged",
    ],
    "Technical Issue": [
        "The application crashes when I open settings", "The website shows an error on checkout",
        "The mobile app is running very slowly", "I cannot upload a file because the page freezes",
        "The dashboard is not loading correctly",
    ],
}


def build_demo_data() -> pd.DataFrame:
    rows = []
    for label, phrases in TEMPLATES.items():
        for phrase in phrases:
            for prefix in ("", "Hello support, ", "Urgent: ", "Please assist, "):
                rows.append({"text": prefix + phrase, "category": label})
    return pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)


def priority_hint(text: str) -> str:
    """Simple keyword heuristic, not a learned or validated urgency model."""
    urgent = {"urgent", "blocked", "cannot access", "security", "fraud", "down"}
    normalized = re.sub(r"\s+", " ", text.lower())
    return "High" if any(term in normalized for term in urgent) else "Normal"


def main() -> None:
    df = build_demo_data()
    df.to_csv(ROOT / "data_demo.csv", index=False)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["category"], test_size=0.25, random_state=42, stratify=df["category"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    (REPORT_DIR / "classification_report.json").write_text(json.dumps(report, indent=2))
    joblib.dump(model, MODEL_DIR / "ticket_classifier.joblib")
    ConfusionMatrixDisplay.from_predictions(y_test, predictions, xticks_rotation=30)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=160)
    print(classification_report(y_test, predictions, zero_division=0))
    print("Model saved to models/ticket_classifier.joblib")


if __name__ == "__main__":
    main()
