"""
Script 04 — Score comment text with NLP models.

Reads:
  data/raw/pr_comments.csv

Produces:
  data/processed/comments_scored.csv   — individual comment scores
  data/processed/pr_nlp_features.csv   — PR-level aggregated NLP features

NLP signals computed per comment:
  toxicity   (unitary/toxic-bert)                  ∈ [0, 1]
  sentiment  (cardiffnlp/twitter-roberta-…)        ∈ [-1, 1]  (P(pos) − P(neg))
  anger      (j-hartmann/emotion-…-distilroberta)  ∈ [0, 1]
  politeness (lexical marker count / word count)   ∈ [0, 1]

PR-level features (aggregated over RECEIVED comments only — i.e., comments
by someone other than the PR author, from non-bot accounts, with sufficient
text length):
  mean_toxicity, max_toxicity
  mean_sentiment
  mean_anger
  mean_politeness

Efficiency note
───────────────
We deduplicate on (stripped) comment body text before running the models,
then broadcast the scores back to all occurrences.  This typically reduces
model calls by 3–10×.

Code-block stripping
─────────────────────
Markdown code blocks (``` … ```) and inline backtick spans are removed
before scoring.  NLP models were trained on natural language; code tokens
produce noisy, meaningless scores.

Usage:
    python scripts/04_score_text.py

GPU is used automatically if available; otherwise falls back to CPU.
Set NLP_BATCH_SIZE in config.py to a lower value if you run out of memory.
"""

import re
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import pipeline

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    TOXICITY_MODEL, SENTIMENT_MODEL, EMOTION_MODEL,
    NLP_BATCH_SIZE, NLP_MAX_LENGTH, MIN_TEXT_LENGTH,
    BOT_SUBSTRINGS, RAW_DIR, PROCESSED_DIR,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
SCORE_MAP_FILE = PROCESSED_DIR / "text_score_map.csv"
COMMENT_FILE    = RAW_DIR / "pr_comments.csv"
PR_STRUCT_FILE  = PROCESSED_DIR / "pr_dataset_structural.csv"
OUTPUT_COMMENTS = PROCESSED_DIR / "comments_scored.csv"
OUTPUT_NLP      = PROCESSED_DIR / "pr_nlp_features.csv"

# Politeness marker words (lower-case).
_POLITENESS_MARKERS = {
    "please", "thanks", "thank", "sorry", "appreciate", "apologies",
    "could you", "would you", "might", "perhaps", "maybe",
    "kindly", "i think", "i believe", "i suggest", "feel free",
}


# ── Text preprocessing ────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    """Strip code, mentions, and URLs; normalize whitespace."""
    # Remove fenced code blocks (``` … ```).
    text = re.sub(r"```[\s\S]*?```", " ", text)
    # Remove inline code (`…`).
    text = re.sub(r"`[^`\n]+`", " ", text)
    # Remove URLs.
    text = re.sub(r"https?://\S+", " ", text)
    # Remove @mentions and #issue references.
    text = re.sub(r"[@#]\w+", " ", text)
    # Collapse whitespace.
    text = " ".join(text.split())
    return text.strip()


def politeness_score(text: str) -> float:
    """Simple lexical politeness: marker density over total words."""
    words = text.lower().split()
    if not words:
        return 0.0
    text_lower = text.lower()
    count = sum(
        1 for marker in _POLITENESS_MARKERS
        if marker in text_lower
    )
    # Normalize by log(1 + word count) so very long texts are not penalized.
    return min(count / (1 + np.log1p(len(words))), 1.0)


# ── Model scoring helpers ─────────────────────────────────────────────────────

def _batch_score(pipe, texts: list[str], batch_size: int) -> list[dict]:
    """Run a pipeline on texts in batches; return list of raw outputs."""
    results = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"  {pipe.model.name_or_path}", leave=False):
        batch = texts[i : i + batch_size]
        out = pipe(
            batch,
            truncation=True,
            max_length=NLP_MAX_LENGTH,
            top_k=None,          # Return scores for ALL labels.
            padding=True,
        )
        results.extend(out)
    return results


def scores_from_label_list(
    label_outputs: list[list[dict]],
    positive_labels: set[str],
    negative_labels: set[str] | None = None,
) -> list[float]:
    """
    Given pipeline output with top_k=None, extract a scalar score.

    If negative_labels is None: return the max score for any positive label.
    If negative_labels is set:  return P(positive) − P(negative).
    """
    out = []
    for item in label_outputs:
        scores = {d["label"].lower(): d["score"] for d in item}
        pos = sum(scores.get(lbl, 0.0) for lbl in positive_labels)
        if negative_labels is not None:
            neg = sum(scores.get(lbl, 0.0) for lbl in negative_labels)
            out.append(float(np.clip(pos - neg, -1.0, 1.0)))
        else:
            out.append(float(np.clip(pos, 0.0, 1.0)))
    return out


def extract_toxicity(label_outputs: list[list[dict]]) -> list[float]:
    """
    unitary/toxic-bert outputs multi-label scores.
    We take the score for the "toxic" label as the primary signal.
    If no "toxic" label is found, fall back to max non-neutral score.
    """
    out = []
    for item in label_outputs:
        scores = {d["label"].lower(): d["score"] for d in item}
        if "toxic" in scores:
            out.append(float(scores["toxic"]))
        else:
            # Fallback: treat any label whose name is not "non_toxic" as toxic.
            tox = max(
                (v for k, v in scores.items() if "non" not in k),
                default=0.0,
            )
            out.append(float(tox))
    return out


def extract_sentiment(label_outputs: list[list[dict]]) -> list[float]:
    """
    cardiffnlp/twitter-roberta-base-sentiment-latest outputs
    positive / neutral / negative labels.
    We return P(positive) − P(negative) ∈ [−1, 1].
    """
    return scores_from_label_list(
        label_outputs,
        positive_labels={"positive"},
        negative_labels={"negative"},
    )


def extract_anger(label_outputs: list[list[dict]]) -> list[float]:
    """
    j-hartmann/emotion-english-distilroberta-base outputs 7 emotions.
    We extract the anger score.
    """
    return scores_from_label_list(
        label_outputs,
        positive_labels={"anger"},
        negative_labels=None,
    )
# SCORE_MAP_FILE = PROCESSED_DIR / "text_score_map.csv"

# ── Bot filter ────────────────────────────────────────────────────────────────

def is_bot(login: str | None) -> bool:
    if login is None:
        return True
    login_lower = str(login).lower()
    return any(sub in login_lower for sub in BOT_SUBSTRINGS)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not COMMENT_FILE.exists():
        log.error("Missing %s. Run 02_collect_comments.py first.", COMMENT_FILE)
        sys.exit(1)

    device = 0 if torch.cuda.is_available() else -1
    log.info("Using device: %s", "CUDA" if device == 0 else "CPU")

    # ── Load and clean comments ───────────────────────────────────────────────
    log.info("Loading comments …")
    cmts = pd.read_csv(COMMENT_FILE)
    cmts["created_at"] = pd.to_datetime(cmts["created_at"], utc=True, errors="coerce")

    cmts = cmts[~cmts["author"].apply(is_bot)]
    cmts = cmts[cmts["body"].notna()]
    cmts["body"] = cmts["body"].astype(str)
    cmts["text_clean"] = cmts["body"].apply(preprocess)
    cmts = cmts[cmts["text_clean"].str.len() >= MIN_TEXT_LENGTH].copy()

    log.info("%d scoreable comment rows.", len(cmts))

    unique_texts = cmts["text_clean"].unique().tolist()
    log.info(
        "Unique texts to score: %d (%.1f%% deduplication).",
        len(unique_texts),
        100 * (1 - len(unique_texts) / len(cmts)),
    )

    # ── Load existing score map if present ────────────────────────────────────
    if SCORE_MAP_FILE.exists():
        score_map = pd.read_csv(SCORE_MAP_FILE)
        done_texts = set(score_map["text_clean"].astype(str))
        log.info("Loaded existing score map with %d texts.", len(done_texts))
    else:
        score_map = pd.DataFrame(columns=[
            "text_clean", "toxicity", "sentiment", "anger", "politeness"
        ])
        done_texts = set()

    missing_texts = [t for t in unique_texts if t not in done_texts]
    log.info("Texts still needing scoring: %d", len(missing_texts))

    if missing_texts:
        log.info("Loading toxicity model (%s) …", TOXICITY_MODEL)
        tox_pipe = pipeline("text-classification", model=TOXICITY_MODEL,
                            device=device, framework="pt")

        log.info("Loading sentiment model (%s) …", SENTIMENT_MODEL)
        sent_pipe = pipeline("text-classification", model=SENTIMENT_MODEL,
                             device=device, framework="pt")

        log.info("Loading emotion model (%s) …", EMOTION_MODEL)
        emo_pipe = pipeline("text-classification", model=EMOTION_MODEL,
                            device=device, framework="pt")

        log.info("Scoring toxicity …")
        tox_raw = _batch_score(tox_pipe, missing_texts, NLP_BATCH_SIZE)

        log.info("Scoring sentiment …")
        sent_raw = _batch_score(sent_pipe, missing_texts, NLP_BATCH_SIZE)

        log.info("Scoring emotions …")
        emo_raw = _batch_score(emo_pipe, missing_texts, NLP_BATCH_SIZE)

        new_scores = pd.DataFrame({
            "text_clean": missing_texts,
            "toxicity": extract_toxicity(tox_raw),
            "sentiment": extract_sentiment(sent_raw),
            "anger": extract_anger(emo_raw),
            "politeness": [politeness_score(t) for t in missing_texts],
        })

        score_map = pd.concat([score_map, new_scores], ignore_index=True)
        score_map.to_csv(SCORE_MAP_FILE, index=False)
        log.info("Saved score map with %d total texts to %s.", len(score_map), SCORE_MAP_FILE)
    else:
        log.info("All unique texts already scored; skipping model inference.")

    # ── Build comments_scored.csv ─────────────────────────────────────────────
    cmts_scored = cmts.merge(score_map, on="text_clean", how="left")
    cmts_scored = cmts_scored.drop(columns=["text_clean"])
    cmts_scored.to_csv(OUTPUT_COMMENTS, index=False)
    log.info("Saved %d scored comments to %s.", len(cmts_scored), OUTPUT_COMMENTS)

    # ── PR-level aggregation ──────────────────────────────────────────────────
    if not PR_STRUCT_FILE.exists():
        log.warning(
            "%s not found — skipping PR-level aggregation. "
            "Run 03_build_dataset.py first, then re-run this script.",
            PR_STRUCT_FILE,
        )
        return

    pr_struct = pd.read_csv(PR_STRUCT_FILE, usecols=["repo", "pr_number", "author"])

    scored_with_pr = cmts_scored.merge(
        pr_struct.rename(columns={"author": "pr_author"}),
        on=["repo", "pr_number"],
        how="inner",
    )

    received = scored_with_pr[
        scored_with_pr["author"] != scored_with_pr["pr_author"]
    ].copy()

    nlp_features = (
        received
        .groupby(["repo", "pr_number"])
        .agg(
            mean_toxicity=("toxicity", "mean"),
            max_toxicity=("toxicity", "max"),
            mean_sentiment=("sentiment", "mean"),
            mean_anger=("anger", "mean"),
            mean_politeness=("politeness", "mean"),
        )
        .reset_index()
    )

    nlp_features.to_csv(OUTPUT_NLP, index=False)
    log.info("Saved PR-level NLP features for %d PRs to %s.", len(nlp_features), OUTPUT_NLP)
    
if __name__ == "__main__":
    main()
