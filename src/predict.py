"""Predict a ticket category using the locally trained model."""
from pathlib import Path
import argparse
import joblib

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Support ticket text")
    args = parser.parse_args()
    model_path = ROOT / "models" / "ticket_classifier.joblib"
    if not model_path.exists():
        raise SystemExit("Model not found. Run: python src/train.py")
    model = joblib.load(model_path)
    category = model.predict([args.text])[0]
    print({"category": category})


if __name__ == "__main__":
    main()
