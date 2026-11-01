# TicketIQ — Support Ticket Classification & Prioritization

**Future Interns · Machine Learning Task 2 (2026)**

TicketIQ is an end-to-end NLP decision-support project for IT helpdesks. It predicts a ticket's issue category from its text and adds a transparent High / Medium / Low triage recommendation.

## Objective

Support teams spend time manually sorting incoming tickets. This project demonstrates how text preprocessing, TF-IDF and supervised classification can help route tickets, while a separate explainable policy flags possible urgency.

## Dataset

This project uses the **Classification of IT Support Tickets** dataset published on Zenodo: https://doi.org/10.5281/zenodo.7648117.

- 2,229 manually classified and anonymized IT support tickets.
- Seven human-assigned categories: Active Directory, Computer-Services, EOL, Fileservice, O365, Software and Support general.
- The publisher describes the corpus as multilingual (mainly English, German, Portuguese and Spanish).
- The official train/test split is used rather than randomly mixing the supplied records.

**Priority-label limitation:** this dataset contains category labels, not High / Medium / Low urgency labels. It would be misleading to claim supervised priority-model accuracy. Category is therefore predicted by ML, while priority is assigned using documented keyword rules. The priority output is a suggestion requiring human review.

The four official CSV files are downloaded automatically into data/raw/ on the first training run and are not committed to this repository.

## ML workflow

1. Download and merge the official train/test text and label files by ticket ID.
2. Clean text: lowercase, Unicode normalization, URL/email and redacted-placeholder removal, punctuation and whitespace normalization.
3. Extract word unigram and bigram features with TF-IDF.
4. Train and compare Logistic Regression, Linear SVM and Multinomial Naive Bayes.
5. Evaluate on the supplied held-out test set using accuracy, macro precision, macro recall, macro F1 and weighted F1.
6. Select the best model by macro F1 to account for class imbalance; save the pipeline and evaluation artefacts.
7. Apply a separate transparent triage policy to assign High / Medium / Low with an explanation.

## Structure

~~~text
FUTURE_ML_02/
├── app.py
├── src/
│   ├── data.py
│   ├── preprocessing.py
│   ├── priority.py
│   ├── train.py
│   └── predict.py
├── data/raw/       # downloaded dataset; git-ignored
├── models/         # generated model files; git-ignored
├── reports/        # generated metrics and charts; git-ignored
├── tests/
├── requirements.txt
└── README.md
~~~

## Setup

Python 3.10 or newer is recommended.

~~~bash
python -m venv .venv
pip install -r requirements.txt
~~~

Windows PowerShell activation: .venv\Scripts\Activate.ps1

macOS/Linux activation: source .venv/bin/activate

## Train and evaluate

Internet access is required on the first run to download the Zenodo CSV files.

~~~bash
python src/train.py
~~~

Generated locally:
- models/category_classifier.joblib
- models/metadata.joblib
- reports/metrics.json
- reports/classification_report.json
- reports/category_confusion_matrix.png
- reports/model_comparison.png
- reports/category_distribution.png

## Predict a ticket

~~~bash
python src/predict.py "Production server is down and multiple users cannot access the system"
~~~

The result includes the predicted category, suggested priority and rule explanation.

## Launch the dashboard

~~~bash
streamlit run app.py
~~~

Dashboard pages:
- **Overview:** dataset and model summary plus category distribution.
- **Classify a ticket:** live category prediction, priority recommendation and class probabilities when supported.
- **Model evaluation:** comparison table, confusion matrix and class-wise report.
- **Triage policy:** explicit priority rules and their limitations.

## Responsible use and limitations

- The model predicts one of the seven categories represented in this dataset; it may not generalize to another organization's taxonomy.
- The dataset is multilingual, while this baseline uses general word-level TF-IDF and does not perform language-specific lemmatization or translation. Review language-specific errors before operational use.
- Priority is heuristic because no verified urgency labels are supplied. Do not report priority accuracy or treat recommendations as automatic incident severity.
- Predictions must be reviewed by a human. Do not use the system as the sole basis for security, safety, or service-impact decisions.
- Do not add real customer-identifiable ticket data to a public repository.

## Reference

Benitez Pereira, L. S. (2023). *Classification of IT Support Tickets* [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7648117

Built as a learning project for Future Interns ML Task 2 (2026).
