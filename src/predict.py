"""Command-line support-ticket category and priority prediction."""
from pathlib import Path
import argparse
import sys
import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def predict_ticket(text: str) -> dict:
    model_path = ROOT / "models" / "category_classifier.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run python src/train.py first.")
    model = joblib.load(model_path)
    from src.priority import assign_priority
    category = str(model.predict([text])[0])
    return {"category": category, **assign_priority(text)}


def main():
    parser = argparse.ArgumentParser(description="Classify and prioritize an IT support ticket.")
    parser.add_argument("text", help="Ticket subject/description")
    args = parser.parse_args()
    print(predict_ticket(args.text))


if __name__ == "__main__":
    main()
