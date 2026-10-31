# Support Ticket Classification | Future Interns ML Task 2

An NLP baseline that classifies customer-support ticket text into issue categories and evaluates its predictions.

## Project status
Starter implementation committed. The built-in demo corpus is **synthetic**, created from transparent text templates so the workflow runs without downloading private or external data. Its scores are not representative of real-world performance.

## Quick start
```bash
python -m venv .venv
pip install -r requirements.txt
python src/train.py
python src/predict.py "I cannot log in after resetting my password"
```

## Pipeline
- Normalize text (lowercase and whitespace cleanup).
- Split examples with stratification.
- Convert text to TF-IDF features.
- Train a Logistic Regression classifier.
- Evaluate with precision, recall, F1, and a confusion matrix.
- Demonstrate a separate transparent priority heuristic; it is not a trained urgency model.

## Categories
The demo uses Billing, Technical Issue, Account Access, and Delivery. Replace the synthetic examples with a suitably licensed, anonymized ticket dataset for meaningful evaluation.

## Structure
```
src/train.py       # generate demo corpus, train and evaluate
src/predict.py     # load model and classify a text
models/            # generated locally; not committed
reports/           # generated evaluation artefacts
requirements.txt
```

## Responsible use
Ticket priority suggestions are heuristic and should be reviewed by staff. Do not upload customer-identifiable data to a public repository. Monitor class imbalance and category-specific errors before deployment.
