# How to Read the Results

---

## File: `results/descriptive_stats.txt`

Reports summary statistics for all variables across the 12,098 focal PRs and then broken down by repository.

**What to look at:**
- `mean_toxicity` vs `max_toxicity`: mean is near zero (0.001), max is still tiny (0.003). The distribution is extremely right-skewed — most PRs have no detectable toxicity.
- `any_toxic_comment_005 = 0.0098`: only ~1% of PRs have any comment above the 0.05 threshold.
- `mean_sentiment = -0.044`: slightly negative on average — expected in critical code review.
- `prior_pr_count`: mean 65.7, SD 96.1 — very wide spread, confirms the need to log-transform this in regressions.

---

## File: `results/regression_results.txt`

Contains logistic regression output for all models. Here is how to read each table.

### Reading a coefficient table

```
                          Coef.  Std.Err.    z       P>|z|    [0.025  0.975]
mean_sentiment            2.104   0.122    17.31   3.8e-67   1.866   2.342
```

| Column | Meaning |
|---|---|
| `Coef.` | Log-odds coefficient. Positive = increases probability of outcome. Negative = decreases it. |
| `Std.Err.` | Standard error (HC3 robust). Smaller = more precise estimate. |
| `z` | Coef / Std.Err. Large absolute value = far from zero. |
| `P>|z|` | p-value. Below 0.05 = statistically significant at 95% confidence. |
| `[0.025  0.975]` | 95% confidence interval. If it does not cross zero, the effect is significant. |

**Converting to probability:** logistic regression coefficients are log-odds, not probabilities. A coefficient of +2.10 for sentiment means that a one-unit increase in mean_sentiment multiplies the odds of merging by e^2.10 ≈ 8.2. To get predicted probabilities use: P = 1 / (1 + e^(−linear_predictor)).

### Model list and what each answers

| Model label | RQ | Outcome |
|---|---|---|
| `RQ5 — PR merge outcome` | RQ1 | `merged` |
| `RQ1 — Contributor retention (30/90/180 days)` | RQ2 | `returned_30/90/180` |
| `RQ3 — Maintainer politeness moderation` | RQ4 | `returned_90` with interaction |
| `RQ4 — Newcomer sensitivity` | RQ3 | `returned_90` with newcomer interaction |
| Binary toxicity models | Robustness | Merge and return with `any_toxic_comment_005/010` |

*(Note: the script numbering of RQs is inverted from the presentation — script RQ5 = presentation RQ1, script RQ1/3/4 = presentation RQ2/3/4.)*

### Key results summary

**Merge outcome (RQ1)**

| Variable | Coef | p | Interpretation |
|---|---|---|---|
| mean_sentiment | +2.10 | <10⁻⁶⁰ | Strongest signal. Positive tone → higher merge probability |
| mean_anger | +4.40 | <10⁻²⁰ | Technical assertiveness, not hostility — signals engaged review |
| mean_politeness | −1.42 | <10⁻¹² | Polite hedging marks soft rejection, not approval |
| max_toxicity_x100 | +0.031 | 0.026 | Small positive effect — does not drive the story |
| log_prior_pr_count | +0.296 | <10⁻⁵² | More experienced authors get merged more often |

**90-day return (RQ2)**

| Variable | Coef | p | Interpretation |
|---|---|---|---|
| log_prior_pr_count | +1.51 | <10⁻¹⁷⁹ | Dominant predictor by far |
| mean_anger | +5.54 | <10⁻¹⁰ | Same engagement signal as in merge |
| max_toxicity_x100 | +0.034 | 0.25 | **Not significant** |
| mean_sentiment | +0.020 | 0.89 | **Not significant** |
| mean_politeness | −0.385 | 0.17 | **Not significant** |

Pseudo-R² = 0.498 — the model explains half the variance in return, almost entirely driven by prior experience.

**Politeness moderation (RQ4)**

| Variable | Coef | p | Interpretation |
|---|---|---|---|
| max_toxicity_x100 × mean_politeness | +0.81 | 0.003 | Polite context buffers the effect of isolated harsh comments |

**Newcomer sensitivity (RQ3)**

| Variable | Coef | p | Interpretation |
|---|---|---|---|
| max_toxicity_x100 × is_newcomer | +0.029 | 0.57 | **Not significant** — newcomers not more sensitive to toxicity |

### Robustness checks

Replacing continuous max toxicity with binary flags:
- `any_toxic_comment_005` in return model: coef +1.38, p < 0.001 — significant
- `any_toxic_comment_010` in return model: coef +0.77, p = 0.15 — not significant

The result changes with threshold. This confirms that toxicity is too sparse and operationalization-sensitive to support a stable "toxicity drives people away" narrative.

---

## Figures

### `01_toxicity_by_repo.png`
Box plot of max toxicity (log scale) per repository. Confirms that all repos are near zero with only extreme outliers. Kubernetes has the widest tail.

### `02_sentiment_distribution.png`
Two panels: left shows sentiment distribution (slightly negative mean, continuous spread); right shows max toxicity (wall at zero). Illustrates why the two signals require different operationalisation strategies.

### `03_toxicity_vs_merged.png`
Strip plot of max toxicity split by merge outcome. Merged PRs carry the high-toxicity outliers — raw data evidence that toxicity is not negatively associated with merge.

### `05_newcomer_vs_experienced.png`
Two panels: left shows similar toxicity exposure for both groups; right shows return rate at 90 days — ~92% for experienced contributors, ~28% for newcomers. Visual evidence that the return gap is about history, not tone.

### `06a_coef_merge.png`
Coefficient plot for the merge logistic regression. Error bars are 95% confidence intervals. Variables are sorted by absolute coefficient size. Sentiment and anger dominate; politeness is the largest negative effect.

### `06b_coef_retention90.png`
Coefficient plot for the 90-day return logistic regression. Prior PR count dwarfs all communication signals. Communication variables cluster near zero with wide confidence intervals.
