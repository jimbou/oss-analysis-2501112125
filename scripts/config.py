"""
Central configuration for the pipeline.

Edit this file to change repositories, date windows, model choices, or
filtering thresholds. All other scripts import from here.
"""

from pathlib import Path

# ── Repositories ──────────────────────────────────────────────────────────────
REPOS = [
    "scikit-learn/scikit-learn",
    "vercel/next.js",
    "kubernetes/kubernetes",
]

# ── Date windows ──────────────────────────────────────────────────────────────
# COLLECTION window is wider than the FOCAL window.  The extra months after
# FOCAL_END serve as the follow-up period for the retention indicators.
#
# Safety check:
#   FOCAL_END + 180 days ≤ COLLECTION_END
#   2025-03-01 + 180d  = 2025-08-28 ≤ 2025-09-30  ✓
#
COLLECTION_START = "2024-01-01"
COLLECTION_END   = "2025-09-30"

FOCAL_START = "2024-01-01"
FOCAL_END   = "2025-03-01"   # All focal PRs have a full 180-day follow-up window.

# ── Retention horizons (days) ─────────────────────────────────────────────────
RETENTION_HORIZONS = [30, 90, 180]

# ── Quality filters ───────────────────────────────────────────────────────────
# Drop focal PRs with fewer received comments than this threshold.
MIN_RECEIVED_COMMENTS = 1

# Drop comment bodies shorter than this many characters before NLP scoring.
MIN_TEXT_LENGTH = 10

# Substrings that flag a GitHub login as a bot (case-insensitive).
BOT_SUBSTRINGS = {
    "bot", "dependabot", "renovate", "github-actions",
    "codecov", "coveralls", "stale", "lock", "snyk",
    "semantic-release", "mergify", "backstroke", "probot",
    "auto-assign", "greenkeeper", "allcontributors", "copilot",
}

# ── NLP models ────────────────────────────────────────────────────────────────
# These are HuggingFace model IDs.  They are downloaded on first use.
TOXICITY_MODEL  = "unitary/toxic-bert"
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL   = "j-hartmann/emotion-english-distilroberta-base"

# Reduce batch size if you run out of memory.
NLP_BATCH_SIZE = 32
NLP_MAX_LENGTH = 512   # Transformer token limit; texts are truncated to this.

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).parent.parent
DATA_DIR      = ROOT_DIR / "data"
RAW_DIR       = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR   = ROOT_DIR / "results"

for _d in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR, RESULTS_DIR / "figures"):
    _d.mkdir(parents=True, exist_ok=True)
