"""
Script 03 — Build the structural dataset.

Reads:
  data/raw/pull_requests.csv   (wide collection window)
  data/raw/pr_comments.csv     (focal PRs only)

Produces:
  data/processed/pr_dataset_structural.csv

Each row is one FOCAL pull request and contains:

  Identifiers / outcomes
  ─────────────────────
  repo, pr_number, author, created_at, merged

  Structural communication features
  ──────────────────────────────────
  num_comments_received        comments from others (non-author, non-bot, non-empty)
  num_review_comments_received subset of above that are review comments
  num_issue_comments_received  subset that are issue-thread comments
  num_participants             distinct non-author, non-bot commenters
  first_response_delay_hours   hours from PR creation to first comment received

  Contributor history features
  ────────────────────────────
  prior_pr_count   PRs by this author in this repo BEFORE created_at
                   (uses the full collection window; lower-bounded at 0)
  is_newcomer      1 if prior_pr_count == 0

  Retention indicators
  ────────────────────
  returned_30 / returned_90 / returned_180
    1 if the author opened another PR in the same repo within N days.
    NaN if the follow-up window extends beyond COLLECTION_END
    (right-censoring safety check).

Notes on the return definition
──────────────────────────────
Return is defined conservatively as "opened another PR in the same repo."
This is clearly observable and unambiguous.  It excludes contributors who
only comment or review — a known limitation discussed in the paper.
The wide collection window (COLLECTION_END = 2025-09-30) ensures that all
focal PRs (FOCAL_END = 2025-03-01) have a full 180-day observation window.

Usage:
    python scripts/03_build_dataset.py
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    FOCAL_START, FOCAL_END, COLLECTION_END,
    RETENTION_HORIZONS, MIN_RECEIVED_COMMENTS, BOT_SUBSTRINGS,
    RAW_DIR, PROCESSED_DIR,
)

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PR_FILE       = RAW_DIR / "pull_requests.csv"
COMMENT_FILE  = RAW_DIR / "pr_comments.csv"
OUTPUT        = PROCESSED_DIR / "pr_dataset_structural.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_bot(login: str | None) -> bool:
    if login is None:
        return True
    login_lower = str(login).lower()
    return any(sub in login_lower for sub in BOT_SUBSTRINGS)


def strip_bots(df: pd.DataFrame, col: str = "author") -> pd.DataFrame:
    return df[~df[col].apply(is_bot)]


# ── Main logic ────────────────────────────────────────────────────────────────

def build_structural_features(
    focal_prs: pd.DataFrame,
    comments: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-PR structural communication features."""

    # Keep only relevant comment columns and clean authors.
    cmts = comments[["repo", "pr_number", "comment_type", "author", "created_at", "body"]].copy()
    cmts = strip_bots(cmts)
    cmts = cmts[cmts["body"].str.strip().str.len() > 0]

    rows = []

    for _, pr in focal_prs.iterrows():
        pr_cmts = cmts[
            (cmts["repo"] == pr["repo"]) &
            (cmts["pr_number"] == pr["pr_number"])
        ]

        # "Received" = comments by someone other than the PR author.
        received = pr_cmts[pr_cmts["author"] != pr["author"]]

        num_received        = len(received)
        num_review_received = (received["comment_type"] == "review_comment").sum()
        num_issue_received  = (received["comment_type"] == "issue_comment").sum()
        num_participants    = received["author"].nunique()

        # First response delay.
        if num_received > 0:
            first_resp = received["created_at"].min()
            delay_hours = (first_resp - pr["created_at"]).total_seconds() / 3600
            delay_hours = max(delay_hours, 0.0)   # Guard against timestamp jitter.
        else:
            delay_hours = np.nan

        rows.append({
            "repo":                        pr["repo"],
            "pr_number":                   pr["pr_number"],
            "author":                      pr["author"],
            "created_at":                  pr["created_at"],
            "merged":                      int(pr["merged"]),
            "num_comments_received":       num_received,
            "num_review_comments_received": int(num_review_received),
            "num_issue_comments_received":  int(num_issue_received),
            "num_participants":            num_participants,
            "first_response_delay_hours":  delay_hours,
        })

    return pd.DataFrame(rows)


def add_contributor_history(
    focal_df: pd.DataFrame,
    all_prs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add prior_pr_count and is_newcomer.

    prior_pr_count: number of PRs by the same author in the same repo
    with created_at strictly before this PR's created_at, within the
    full collection window.
    """
    # Work with a stripped-down table for efficiency.
    pr_index = all_prs[["repo", "author", "created_at"]].copy()
    pr_index = strip_bots(pr_index)
    pr_index["created_at"] = pd.to_datetime(pr_index["created_at"], utc=True)

    # Sort by (repo, author, created_at) and use cumcount as prior_pr_count.
    # cumcount() gives the 0-based rank within the group, so PR #0 for an
    # author has prior_pr_count=0, PR #1 has prior_pr_count=1, etc.
    pr_index_sorted = pr_index.sort_values(["repo", "author", "created_at"])
    pr_index_sorted["prior_pr_count"] = pr_index_sorted.groupby(
        ["repo", "author"]
    ).cumcount()

    # Merge back into focal_df on exact (repo, author, created_at).
    focal_df["created_at"] = pd.to_datetime(focal_df["created_at"], utc=True)
    merged = focal_df.merge(
        pr_index_sorted[["repo", "author", "created_at", "prior_pr_count"]],
        on=["repo", "author", "created_at"],
        how="left",
    )
    merged["prior_pr_count"] = merged["prior_pr_count"].fillna(0).astype(int)
    merged["is_newcomer"] = (merged["prior_pr_count"] == 0).astype(int)
    return merged


def add_return_indicators(
    focal_df: pd.DataFrame,
    all_prs: pd.DataFrame,
    horizons: list[int],
    collection_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Add returned_N columns for each N in horizons.

    returned_N = 1 if the author opened another PR in the same repo
    within N days after the focal PR's created_at.
    returned_N = NaN if the follow-up window extends past collection_end
    (right-censoring: we do not have enough data to observe a return).

    This uses the wide collection window — all_prs includes PRs beyond
    the focal window so that returns CAN be observed.
    """
    # All activity rows: (repo, author, created_at) for every PR in collection.
    activity = all_prs[["repo", "author", "created_at"]].copy()
    activity = strip_bots(activity)
    activity["created_at"] = pd.to_datetime(activity["created_at"], utc=True)

    focal_df = focal_df.copy()
    focal_df["created_at"] = pd.to_datetime(focal_df["created_at"], utc=True)

    for n in horizons:
        col = f"returned_{n}"
        returned_flags = []

        for _, row in focal_df.iterrows():
            t0 = row["created_at"]
            t1 = t0 + pd.Timedelta(days=n)

            # Right-censoring: if we cannot observe the full window, mark NaN.
            if t1 > collection_end:
                returned_flags.append(np.nan)
                continue

            future_activity = activity[
                (activity["repo"]   == row["repo"]) &
                (activity["author"] == row["author"]) &
                (activity["created_at"] > t0) &
                (activity["created_at"] <= t1)
            ]
            returned_flags.append(1 if len(future_activity) > 0 else 0)

        focal_df[col] = returned_flags

    return focal_df


def main() -> None:
    if OUTPUT.exists():
        log.info("%s already exists — skipping. Delete to re-run.", OUTPUT)
        return

    if not PR_FILE.exists():
        log.error("Missing %s.  Run 01_collect_prs.py first.", PR_FILE)
        sys.exit(1)
    if not COMMENT_FILE.exists():
        log.error("Missing %s.  Run 02_collect_comments.py first.", COMMENT_FILE)
        sys.exit(1)

    log.info("Loading raw data …")
    all_prs  = pd.read_csv(PR_FILE)
    comments = pd.read_csv(COMMENT_FILE)

    all_prs["created_at"]  = pd.to_datetime(all_prs["created_at"],  utc=True)
    all_prs["closed_at"]   = pd.to_datetime(all_prs["closed_at"],   utc=True, errors="coerce")
    all_prs["merged_at"]   = pd.to_datetime(all_prs["merged_at"],   utc=True, errors="coerce")
    comments["created_at"] = pd.to_datetime(comments["created_at"], utc=True, errors="coerce")

    # Drop bot-authored PRs from the full set used for return/history computation.
    all_prs_clean = strip_bots(all_prs, col="author")

    # Select focal PRs.
    focal_mask = (
        (all_prs_clean["created_at"] >= pd.Timestamp(FOCAL_START, tz="UTC")) &
        (all_prs_clean["created_at"] <= pd.Timestamp(FOCAL_END,   tz="UTC"))
    )
    focal_prs = all_prs_clean[focal_mask].copy()
    log.info("%d focal PRs in [%s, %s].", len(focal_prs), FOCAL_START, FOCAL_END)

    # 1. Structural communication features.
    log.info("Computing structural features …")
    df = build_structural_features(focal_prs, comments)

    # 2. Contributor history.
    log.info("Computing contributor history …")
    df = add_contributor_history(df, all_prs_clean)

    # 3. Retention indicators.
    log.info("Computing retention indicators …")
    collection_end = pd.Timestamp(COLLECTION_END, tz="UTC")
    df = add_return_indicators(df, all_prs_clean, RETENTION_HORIZONS, collection_end)

    # 4. Apply quality filter: at least MIN_RECEIVED_COMMENTS.
    before = len(df)
    df = df[df["num_comments_received"] >= MIN_RECEIVED_COMMENTS].copy()
    log.info(
        "Dropped %d PRs with fewer than %d received comments; %d remain.",
        before - len(df), MIN_RECEIVED_COMMENTS, len(df),
    )

    # 5. Summary.
    log.info("\nDataset summary:")
    log.info("  Total rows:    %d", len(df))
    log.info("  Merged:        %.1f%%", 100 * df["merged"].mean())
    log.info("  Newcomers:     %.1f%%", 100 * df["is_newcomer"].mean())
    for n in RETENTION_HORIZONS:
        col = f"returned_{n}"
        valid = df[col].notna()
        log.info(
            "  returned_%d:   %.1f%% (N=%d, %d NaN)",
            n, 100 * df.loc[valid, col].mean(), valid.sum(), (~valid).sum(),
        )

    df.to_csv(OUTPUT, index=False)
    log.info("Saved structural dataset to %s.", OUTPUT)


if __name__ == "__main__":
    main()
