"""Text normalization for short, noisy IT support-ticket descriptions."""
import re
import unicodedata

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
PLACEHOLDER_RE = re.compile(
    r"\[(?:NAME|EMAIL|PHONE|ADDRESS|COMPANY|LOCATION|SERVER|USERNAME|TICKET ID|IP_ADDRESS|SOFTWARE|PASSWORD|AWS_ACCESS_KEY)\]",
    re.IGNORECASE,
)
NON_WORD_RE = re.compile(r"[^\w\s]", re.UNICODE)
SPACE_RE = re.compile(r"\s+")


def clean_text(value: object) -> str:
    """Normalize text while preserving useful technical tokens and accents."""
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text).lower()
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = PLACEHOLDER_RE.sub(" ", text)
    text = NON_WORD_RE.sub(" ", text)
    return SPACE_RE.sub(" ", text).strip()
