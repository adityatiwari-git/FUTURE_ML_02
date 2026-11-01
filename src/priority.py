"""Explainable triage rules; priority is a heuristic, not a trained target."""
import re

HIGH_PATTERNS = (
    r"\b(outage|service down|service is down|system down|system is down|production down|production server is down|network down|network is down|server down|server is down)\b",
    r"\b(all users|entire team|everyone|multiple users)\b.*\b(cannot|unable|down|blocked)\b",
    r"\b(security breach|data breach|ransomware|malware|compromised|unauthorized access)\b",
    r"\b(critical failure|business critical|complete failure|emergency)\b",
)
LOW_PATTERNS = (
    r"\b(how do i|how can i|information about|question about|documentation)\b",
    r"\b(feature request|suggestion|general inquiry|general question)\b",
)
MEDIUM_PATTERNS = (
    r"\b(not working|does not work|doesn't work|unable to|cannot|can't|error|failed|failure|issue|problem)\b",
    r"\b(access|permission|install|installation|configuration|setup|request)\b",
)


def assign_priority(text: str) -> dict[str, str]:
    """Return priority and a human-readable reason using transparent rules."""
    normalized = re.sub(r"\s+", " ", str(text).lower())
    if any(re.search(pattern, normalized) for pattern in HIGH_PATTERNS):
        return {"priority": "High", "reason": "Potential service outage, broad impact, or security risk keyword detected."}
    if any(re.search(pattern, normalized) for pattern in LOW_PATTERNS):
        return {"priority": "Low", "reason": "Informational or non-urgent request wording detected."}
    if any(re.search(pattern, normalized) for pattern in MEDIUM_PATTERNS):
        return {"priority": "Medium", "reason": "Operational issue or service request detected; staff review is recommended."}
    return {"priority": "Medium", "reason": "Defaulted to Medium because the text does not establish urgency."}
