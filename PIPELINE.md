# Pipeline Guide

The analysis runs in five sequential scripts. Each script reads from and writes to specific files. Run them in order.

---

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install pandas numpy torch transformers tqdm statsmodels matplotlib seaborn scipy python-dotenv requests

# Configure GitHub token (required for data collection)
cp .env.example .env
# Edit .env and set: GITHUB_TOKEN=your_personal_access_token
```

---

## Step 1 — Collect Pull Requests
**Script:** `scripts/01_collect_prs.py`
**Writes:** `data/raw/pull_requests.csv`

Fetches PR metadata from the GitHub API for the three target repositories (Kubernetes, Next.js, scikit-learn) across the full collection window (Jul 2023 – Sep 2025). Each row is one PR with fields: repo, pr_number, title, author, state, merged, created_at, merged_at, closed_at.

```bash
python scripts/01_collect_prs.py
```

Rate-limit aware. Expects `GITHUB_TOKEN` in `.env`.

---

## Step 2 — Collect Comments
**Script:** `scripts/02_collect_comments.py`
**Reads:** `data/raw/pull_requests.csv`
**Writes:** `data/raw/pr_comments.csv`

Fetches all review comments and issue comments for each focal PR (those in the Jan 2024 – Mar 2025 window). The script is **resumable** — if interrupted, re-running continues from where it stopped using a progress file.

```bash
python scripts/02_collect_comments.py
```

This is the slowest step. Expect several hours on a full run due to API rate limits. The output (`pr_comments.csv`) is ~253MB.

---

## Step 3 — Build Structural Dataset
**Script:** `scripts/03_build_dataset.py`
**Reads:** `data/raw/pull_requests.csv`
**Writes:** `data/processed/pr_dataset_structural.csv`

Constructs the one-row-per-PR analysis dataset with:
- Binary outcomes: `merged`, `returned_30`, `returned_90`, `returned_180`
- Structural features: `num_comments_received`, `num_participants`, `first_response_delay_hours`
- Contributor history: `prior_pr_count`, `is_newcomer`
- Repository fixed effects

Newcomer classification uses the six-month lookback (Jul–Dec 2023) to avoid data leakage: a contributor is only labelled as a newcomer if they have zero PRs in both the lookback *and* focal windows prior to the focal PR.

```bash
python scripts/03_build_dataset.py
```

---

## Step 4 — Score Comments with NLP
**Script:** `scripts/04_score_text.py`
**Reads:** `data/raw/pr_comments.csv`, `data/processed/pr_dataset_structural.csv`
**Writes:** `data/processed/comments_scored.csv`, `data/processed/pr_nlp_features.csv`, `data/processed/text_score_map.csv`

Scores every comment on four signals:

| Signal | Method | Model |
|---|---|---|
| Toxicity | Transformer | `unitary/toxic-bert` |
| Sentiment | Transformer | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Anger | Transformer | `j-hartmann/emotion-english-distilroberta-base` |
| Politeness | Lexical heuristic | Marker-word density (please, thanks, sorry, could you, …) |

**Preprocessing before scoring:** code blocks (``` ``` ```), inline code, URLs, @mentions, and #references are stripped so models only see natural language.

**Deduplication:** identical comment texts are scored once and scores are broadcast back — reduces model calls by 3–10×.

**Aggregation:** only *received* comments (by someone other than the PR author, from non-bot accounts) are aggregated to the PR level, producing: `mean_toxicity`, `max_toxicity`, `mean_sentiment`, `mean_anger`, `mean_politeness`.

GPU is used automatically if available.

```bash
python scripts/04_score_text.py
```

---

## Step 5 — Analysis and Visualisation
**Script:** `scripts/05_analysis.py`
**Reads:** `data/processed/pr_dataset_structural.csv`, `data/processed/pr_nlp_features.csv`
**Writes:** `results/descriptive_stats.txt`, `results/regression_results.txt`, `results/figures/*.png`

Runs:
- Descriptive statistics by repository
- VIF check on communication variables
- Logistic regressions for all four RQs (merge outcome, return at 30/90/180 days, newcomer interaction, politeness moderation)
- Binary toxicity robustness checks (0.05 and 0.10 thresholds)
- Author fixed-effects models (failed — singular matrix)
- All six figures

```bash
python scripts/05_analysis.py
```

---

## Config

Key parameters are in `scripts/config.py`:

| Parameter | Default | Meaning |
|---|---|---|
| `REPOS` | kubernetes, vercel/next.js, scikit-learn | Target repositories |
| `FOCAL_START` | 2024-01-01 | Start of focal window |
| `FOCAL_END` | 2025-03-31 | End of focal window |
| `LOOKBACK_START` | 2023-07-01 | Start of contributor history window |
| `RETENTION_HORIZONS` | [30, 90, 180] | Return windows in days |
| `NLP_BATCH_SIZE` | 32 | Reduce if GPU runs out of memory |
| `MIN_TEXT_LENGTH` | 20 | Minimum comment length in characters |
