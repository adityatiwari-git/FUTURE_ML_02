# Dataset

This project downloads the **Classification of IT Support Tickets** dataset from Zenodo (DOI: https://doi.org/10.5281/zenodo.7648117). It contains 2,229 manually classified and anonymized IT support tickets in seven categories. The publisher describes the tickets as multilingual (mainly English, German, Portuguese and Spanish).

The four official split files (X_train.csv, X_test.csv, y_train.csv, y_test.csv) are downloaded into data/raw/ and are intentionally not committed.

**Important:** the dataset contains category labels but no priority labels. High / Medium / Low is therefore assigned by an editable, explainable rule-based triage layer. These are operational suggestions, not ground-truth predictions or a separately trained priority model.
