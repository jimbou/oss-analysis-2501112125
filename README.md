# Communication Tone in Open Source Pull Requests
## Quantitative Analysis of Open Source Software — Final Project
**Dimitrios Stamatios Bouras | 2501112125 | Spring 2026**

---

## The Story

This project investigates whether the *way* reviewers communicate in a pull request discussion is associated with two outcomes: whether the PR gets merged, and whether the author comes back to contribute again.

The intuition behind the question is simple: if you receive harsh or discouraging feedback on your first contribution, are you less likely to return? And does positive discussion lead to more PRs being accepted? These questions matter for understanding what shapes open source community health.

### What We Found

The answer is more nuanced than the original intuition suggests.

**Communication tone matters for merge, but not much for return.**

- **Sentiment** is the strongest predictor of merge outcome (coef +2.10, p < 10⁻⁶⁰). PRs with more positive discussion are significantly more likely to be merged.
- **Prior contributor experience** dominates return probability (coef +1.51 at 90 days, p < 10⁻¹⁷⁹). Whether someone comes back is driven almost entirely by how embedded they already were in the project — not by the tone of any single review.
- **Overt toxicity is extremely rare** in these three repositories. Only 1% of PRs have any comment above a 0.05 toxicity threshold. These are well-moderated communities, and this sparsity limits what toxicity alone can explain.
- **Anger associates positively with merge** — counterintuitive until you realise the emotion model was trained on social media. In code review, assertive technical language ("this is completely broken") scores as anger but signals engaged, decisive review.
- **Politeness associates negatively with merge** — polite hedging ("maybe consider...", "not sure if...") is the register reviewers use for soft rejection, not approval.
- **Politeness buffers toxicity**: the interaction between overall discussion politeness and max toxicity is significant (p = 0.003). A single harsh comment in an otherwise civil discussion lands differently than the same comment in a hostile thread.
- **Newcomers are not disproportionately driven away by harsh language** — the newcomer × toxicity interaction is not significant. The large return-rate gap between newcomers (~28%) and experienced contributors (~92%) is explained by history, not by differential sensitivity to tone.

### The Broader Lesson

The biggest risk to losing contributors is not any single harsh PR review — it is failing to onboard contributors deeply enough to begin with. And when applying off-the-shelf NLP models to code review, the outputs carry real information, but not the information the models were designed to capture. Domain context is essential.

---

## Repository Structure

```
artifacts/
├── README.md               ← this file: story and navigation
├── PIPELINE.md             ← how to run the full analysis from scratch
├── DATA.md                 ← data collection design, files, and schema
├── RESULTS.md              ← how to read the regression output and figures
│
├── scripts/                ← all five pipeline scripts + config
├── data/
│   ├── raw/                ← pull_requests.csv, pr_comments.csv
│   └── processed/          ← pr_dataset_structural.csv, pr_nlp_features.csv,
│                              comments_scored.csv, text_score_map.csv
├── results/
│   ├── descriptive_stats.txt
│   ├── regression_results.txt
│   └── figures/            ← the six graphs used in the analysis
│
└── presentations/
    ├── part1_prep/         ← first presentation (slides + PDF)
    ├── part2_midterm/      ← midterm presentation (slides + PDF + script)
    └── part3_final/        ← final presentation (slides + PDF + script)
```

---

## Figures Quick Reference

| File | What it shows | Used in |
|---|---|---|
| `01_toxicity_by_repo.png` | Max toxicity distribution per repository | Midterm |
| `02_sentiment_distribution.png` | Sentiment distribution + toxicity sparsity | Final |
| `03_toxicity_vs_merged.png` | Toxicity by merge outcome — raw data | Final |
| `05_newcomer_vs_experienced.png` | Toxicity exposure + return rate by experience | Final |
| `06a_coef_merge.png` | Logistic regression coefficients for merge model | Final |
| `06b_coef_retention90.png` | Logistic regression coefficients for 90-day return | Final |
