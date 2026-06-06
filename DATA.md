# Data Description

---

## Collection Design

The collection window was deliberately wider than the analysis window to solve two problems:

```
Jul 2023          Jan 2024                    Mar 2025        Sep 2025
    |                 |                           |               |
    |←─ lookback ────→|←────── focal PRs ─────────→|←─ follow-up ─→|
    |   (6 months)    |     (15 months)            |  (6 months)   |
```

- **Lookback (Jul–Dec 2023):** Used only to establish contributor history. A contributor with PRs here is not a newcomer when their first focal PR appears in January 2024. Without this window, anyone active before the dataset starts looks like a newcomer — which would be data leakage.
- **Focal window (Jan 2024 – Mar 2025):** The 12,098 PRs that are actually analysed.
- **Follow-up (Mar–Sep 2025):** Needed to label 180-day return outcomes for the latest focal PRs.

---

## Source and Method

- **Source:** GitHub REST API (v3) using a personal access token
- **Endpoints used:** `GET /repos/{owner}/{repo}/pulls` for PR metadata; `GET /pulls/{number}/comments` and `GET /issues/{number}/comments` for discussion text
- **Repositories:**
  - `kubernetes/kubernetes` — large systems project, highest toxicity rate
  - `vercel/next.js` — frontend framework, lowest toxicity rate
  - `scikit-learn/scikit-learn` — ML library, highest newcomer rate (27%)

---

## Files

### `data/raw/pull_requests.csv` — 4.1 MB
One row per PR across the full collection window.

| Column | Type | Description |
|---|---|---|
| `repo` | string | `owner/name` |
| `pr_number` | int | PR number within the repo |
| `author` | string | GitHub login of the PR author |
| `state` | string | `open`, `closed`, `merged` |
| `merged` | bool | Whether the PR was merged |
| `created_at` | datetime | When the PR was opened |
| `merged_at` | datetime | When merged (null if not) |
| `closed_at` | datetime | When closed |

### `data/raw/pr_comments.csv` — 253 MB *(moved)*
One row per comment on focal PRs.

| Column | Type | Description |
|---|---|---|
| `repo` | string | Repository |
| `pr_number` | int | PR number |
| `comment_id` | int | GitHub comment ID |
| `author` | string | Commenter login |
| `body` | string | Raw comment text |
| `created_at` | datetime | Comment timestamp |

### `data/processed/pr_dataset_structural.csv` — 1.2 MB
**Main analysis file.** One row per focal PR. Used as the left-hand side of all regressions.

| Column | Type | Description |
|---|---|---|
| `repo` | string | Repository |
| `pr_number` | int | PR number |
| `author` | string | PR author login |
| `merged` | 0/1 | Outcome: merged |
| `returned_30/90/180` | 0/1 | Outcome: opened another PR within N days |
| `num_comments_received` | int | Comments by others (bots excluded) |
| `num_participants` | int | Distinct non-author commenters |
| `first_response_delay_hours` | float | Hours from PR open to first review comment |
| `prior_pr_count` | int | PRs by this author before this focal PR |
| `is_newcomer` | 0/1 | 1 if prior_pr_count == 0 in both lookback and earlier focal window |

### `data/processed/pr_nlp_features.csv` — 1.4 MB
NLP scores aggregated to the PR level. Joined with `pr_dataset_structural.csv` on `(repo, pr_number)` for regression.

| Column | Type | Description |
|---|---|---|
| `mean_toxicity` | float [0,1] | Average toxicity across received comments |
| `max_toxicity` | float [0,1] | Worst single comment toxicity (main operationalisation) |
| `max_toxicity_x100` | float | max_toxicity × 100 (for coefficient readability) |
| `any_toxic_comment_005` | 0/1 | Binary flag: any comment ≥ 0.05 |
| `any_toxic_comment_010` | 0/1 | Binary flag: any comment ≥ 0.10 |
| `mean_sentiment` | float [-1,1] | P(positive) − P(negative) averaged over comments |
| `mean_anger` | float [0,1] | Average anger score from 7-emotion classifier |
| `mean_politeness` | float [0,1] | Average lexical politeness marker density |

### `data/processed/comments_scored.csv` — 232 MB *(moved)*
Individual comment-level scores before PR aggregation. Useful for inspection or alternative aggregation strategies.

### `data/processed/text_score_map.csv` — 78 MB *(moved)*
Cache of NLP scores keyed by (deduplicated) comment text. Re-used by `04_score_text.py` to avoid re-running models on already-seen text.

---

## Cleaning Decisions

| Decision | Rationale |
|---|---|
| Bot accounts filtered by login substring (e.g. `[bot]`, `-bot`, `ci-`) | Bot comments are not human communication |
| Comments shorter than 20 characters discarded | Too short to score meaningfully |
| Code blocks, inline code, URLs, @mentions stripped before NLP | Models were trained on natural language; code tokens produce noise |
| Only *received* comments scored (author excluded) | We measure the tone of feedback received, not what the author wrote |
| Duplicate comment bodies scored once then broadcast | Reduces inference cost 3–10× with no loss |
