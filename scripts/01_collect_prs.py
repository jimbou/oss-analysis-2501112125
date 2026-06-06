"""
Script 01 — Collect pull request metadata.

For each configured repository, pages through ALL pull requests
(open + closed) within the COLLECTION window and saves the result to
  data/raw/pull_requests.csv

This wide window is intentional: PRs outside the focal analysis window
are still collected so that we can compute:
  - prior_pr_count  (PRs by the same author before the focal PR)
  - returned_*      (PRs by the same author after the focal PR)

Usage:
    python scripts/01_collect_prs.py

Requires:
    GITHUB_TOKEN env variable (or in .env file)

Output columns:
    repo, pr_number, author, created_at, closed_at, merged_at, merged,
    state, title, body_length, num_comments, num_review_comments
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

# Allow imports from the scripts/ directory.
sys.path.insert(0, str(Path(__file__).parent))
import github_client as gh
from config import REPOS, COLLECTION_START, COLLECTION_END, RAW_DIR

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT = RAW_DIR / "pull_requests.csv"


def _parse_dt(s: str | None) -> str | None:
    """Return ISO-format datetime string, or None."""
    return s  # GitHub already returns ISO-8601 strings; keep as-is.


def collect_prs_for_repo(owner: str, repo: str) -> list[dict]:
    """
    Page through /repos/{owner}/{repo}/pulls?state=all&sort=created&direction=asc
    and collect all PRs with created_at in [COLLECTION_START, COLLECTION_END].

    We iterate in ascending order so we can stop as soon as created_at
    exceeds COLLECTION_END.
    """
    end_dt = datetime.fromisoformat(COLLECTION_END).replace(tzinfo=timezone.utc)
    start_dt = datetime.fromisoformat(COLLECTION_START).replace(tzinfo=timezone.utc)

    rows: list[dict] = []
    path = f"/repos/{owner}/{repo}/pulls"
    params = {"state": "all", "sort": "created", "direction": "desc"}

    page_num = 0
    for page in gh.paginate(path, params):
        page_num += 1
        if not page:
            break

        for pr in page:
            created = datetime.fromisoformat(
                pr["created_at"].replace("Z", "+00:00")
            )

            if created > end_dt:
                continue  # too new, skip

            if created < start_dt:
                log.info(
                    "  Reached COLLECTION_START (%s) at page %d — stopping.",
                    COLLECTION_START, page_num,
                )
                return rows

            rows.append({
                "repo":                f"{owner}/{repo}",
                "pr_number":           pr["number"],
                "author":              pr["user"]["login"] if pr.get("user") else None,
                "created_at":          pr["created_at"],
                "closed_at":           pr.get("closed_at"),
                "merged_at":           pr.get("merged_at"),
                "merged":              pr.get("merged_at") is not None,
                "state":               pr["state"],
                "title":               pr.get("title", ""),
                "body_length":         len(pr.get("body") or ""),
                "num_comments":        pr.get("comments", 0),
                "num_review_comments": pr.get("review_comments", 0),
            })

        log.info(
            "  %s/%s — page %d, %d PRs collected so far.",
            owner, repo, page_num, len(rows),
        )

    return rows


def main() -> None:
    if OUTPUT.exists():
        log.info("%s already exists — skipping collection. Delete to re-run.", OUTPUT)
        return

    all_rows: list[dict] = []

    for repo_slug in REPOS:
        owner, repo = repo_slug.split("/", 1)
        log.info("Collecting PRs for %s ...", repo_slug)
        rows = collect_prs_for_repo(owner, repo)
        log.info("  → %d PRs collected for %s.", len(rows), repo_slug)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    # Normalise types.
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True)
    df["closed_at"]  = pd.to_datetime(df["closed_at"],  utc=True, errors="coerce")
    df["merged_at"]  = pd.to_datetime(df["merged_at"],  utc=True, errors="coerce")
    df["merged"]     = df["merged"].astype(bool)

    df.to_csv(OUTPUT, index=False)
    log.info("Saved %d rows to %s.", len(df), OUTPUT)


if __name__ == "__main__":
    main()
