"""Download and assemble the anonymized Zenodo IT support-ticket dataset."""
from pathlib import Path
from urllib.request import urlretrieve
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
BASE_URL = "https://zenodo.org/records/7648117/files/{}?download=1"
FILES = ("X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv")


def download_dataset(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        destination = RAW_DIR / filename
        if force or not destination.exists() or destination.stat().st_size == 0:
            print(f"Downloading {filename}...")
            urlretrieve(BASE_URL.format(filename), destination)


def load_dataset(download: bool = True):
    """Return train/test frames with columns id, text, category."""
    if download:
        download_dataset()
    missing = [name for name in FILES if not (RAW_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing dataset files: {', '.join(missing)}")
    x_train = pd.read_csv(RAW_DIR / "X_train.csv")
    x_test = pd.read_csv(RAW_DIR / "X_test.csv")
    y_train = pd.read_csv(RAW_DIR / "y_train.csv")
    y_test = pd.read_csv(RAW_DIR / "y_test.csv")
    train = x_train.merge(y_train, on="id", validate="one_to_one")
    test = x_test.merge(y_test, on="id", validate="one_to_one")
    train = train.rename(columns={"category_truth": "category"})
    test = test.rename(columns={"category_truth": "category"})
    for frame in (train, test):
        frame["text"] = frame["text"].fillna("").astype(str)
        frame["category"] = frame["category"].fillna("").astype(str)
        frame.drop_duplicates(subset=["id"], inplace=True)
        invalid = (frame["text"].str.strip() == "") | (frame["category"].str.strip() == "")
        frame.drop(frame[invalid].index, inplace=True)
    return train.reset_index(drop=True), test.reset_index(drop=True)
