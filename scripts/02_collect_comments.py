"""
Script 02 — Collect pull request comments (resumable).

For each FOCAL pull request in data/raw/pull_requests.csv, fetches:
  - Issue-thread comments  (/repos/{owner}/{repo}/issues/{number}/comments)
  - Review comments        (/repos/{owner}/{repo}/pulls/{number}/comments)

Both types are unioned into a single table saved to:
  data/raw/pr_comments.csv

A separate progress file is maintained:
  data/raw/pr_comments_progress.csv

This makes the script resumable:
- already processed PRs are skipped on restart
- PRs with zero comments are still marked as processed
- partial progress is preserved if interrupted

Usage:
    python scripts/02_collect_comments.py

Output columns:
    repo, pr_number, comment_type (issue_comment | review_comment),
    comment_id, author, created_at, body
"""

import sys
import logging
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import github_client as gh
from config import FOCAL_START, FOCAL_END, RAW_DIR

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PR_FILE = RAW_DIR / "pull_requests.csv"
OUTPUT = RAW_DIR / "pr_comments.csv"
PROGRESS = RAW_DIR / "pr_comments_progress.csv"

COMMENT_COLUMNS = [
    "repo",
    "pr_number",
    "comment_type",
    "comment_id",
    "author",
    "created_at",
    "body",
]

PROGRESS_COLUMNS = [
    "repo",
    "pr_number",
    "status",   # success | failed
]


def fetch_issue_comments(owner: str, repo: str, pr_number: int) -> list[dict]:
    rows = []
    for page in gh.paginate(f"/repos/{owner}/{repo}/issues/{pr_number}/comments"):
        for c in page:
            rows.append({
                "comment_type": "issue_comment",
                "comment_id": c["id"],
                "author": c["user"]["login"] if c.get("user") else None,
                "created_at": c["created_at"],
                "body": c.get("body", "") or "",
            })
    return rows


def fetch_review_comments(owner: str, repo: str, pr_number: int) -> list[dict]:
    rows = []
    for page in gh.paginate(f"/repos/{owner}/{repo}/pulls/{pr_number}/comments"):
        for c in page:
            rows.append({
                "comment_type": "review_comment",
                "comment_id": c["id"],
                "author": c["user"]["login"] if c.get("user") else None,
                "created_at": c["created_at"],
                "body": c.get("body", "") or "",
            })
    return rows


def append_comments(rows: list[dict]) -> None:
    """Append comment rows to OUTPUT, creating the file with a header if needed."""
    if not rows:
        return
    df = pd.DataFrame(rows, columns=COMMENT_COLUMNS)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    write_header = not OUTPUT.exists()
    df.to_csv(OUTPUT, mode="a", header=write_header, index=False)


def append_progress(repo: str, pr_number: int, status: str) -> None:
    """Append one processed PR record to the progress file."""
    df = pd.DataFrame(
        [{"repo": repo, "pr_number": pr_number, "status": status}],
        columns=PROGRESS_COLUMNS,
    )
    write_header = not PROGRESS.exists()
    df.to_csv(PROGRESS, mode="a", header=write_header, index=False)


def load_processed() -> set[tuple[str, int]]:
    """Load already processed PR keys from the progress file."""
    if not PROGRESS.exists():
        return set()
    df = pd.read_csv(PROGRESS, usecols=["repo", "pr_number"])
    return set((r, int(n)) for r, n in zip(df["repo"], df["pr_number"]))


def ensure_output_files_exist() -> None:
    """Create empty output/progress files with headers if missing."""
    if not OUTPUT.exists():
        pd.DataFrame(columns=COMMENT_COLUMNS).to_csv(OUTPUT, index=False)
    if not PROGRESS.exists():
        pd.DataFrame(columns=PROGRESS_COLUMNS).to_csv(PROGRESS, index=False)


def main() -> None:
    if not PR_FILE.exists():
        log.error("%s not found. Run 01_collect_prs.py first.", PR_FILE)
        sys.exit(1)

    prs = pd.read_csv(PR_FILE, parse_dates=["created_at"])
    prs["created_at"] = pd.to_datetime(prs["created_at"], utc=True)

    focal_mask = (
        (prs["created_at"] >= pd.Timestamp(FOCAL_START, tz="UTC")) &
        (prs["created_at"] <= pd.Timestamp(FOCAL_END, tz="UTC"))
    )
    focal_prs = prs[focal_mask].copy()
    log.info("Found %d focal PRs in the analysis window.", len(focal_prs))

    ensure_output_files_exist()
    processed = load_processed()
    log.info("Loaded %d already processed PRs from checkpoint.", len(processed))

    remaining_mask = [
        (row["repo"], int(row["pr_number"])) not in processed
        for _, row in focal_prs.iterrows()
    ]
    remaining_prs = focal_prs[remaining_mask].copy()
    log.info("Need to fetch comments for %d remaining focal PRs.", len(remaining_prs))

    if remaining_prs.empty:
        log.info("Nothing to do — all focal PRs are already processed.")
        return

    success_count = 0
    fail_count = 0
    comment_count = 0

    for _, row in tqdm(remaining_prs.iterrows(), total=len(remaining_prs), desc="PRs"):
        repo_slug = row["repo"]
        owner, repo = repo_slug.split("/", 1)
        pr_number = int(row["pr_number"])
        prefix = {"repo": repo_slug, "pr_number": pr_number}

        try:
            issue_cmts = fetch_issue_comments(owner, repo, pr_number)
            review_cmts = fetch_review_comments(owner, repo, pr_number)
            rows = [{**prefix, **c} for c in (issue_cmts + review_cmts)]

            append_comments(rows)
            append_progress(repo_slug, pr_number, "success")

            success_count += 1
            comment_count += len(rows)

        except Exception as exc:
            log.warning("Failed to fetch comments for %s#%d: %s", repo_slug, pr_number, exc)
            append_progress(repo_slug, pr_number, "failed")
            fail_count += 1

    log.info(
        "Done. Success: %d PRs, Failed: %d PRs, New comments saved: %d, Output: %s",
        success_count, fail_count, comment_count, OUTPUT,
    )
    log.info("Progress checkpoint saved to %s", PROGRESS)


if __name__ == "__main__":
    main()