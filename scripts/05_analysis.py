"""
Script 05 — Statistical analysis and visualization.

Reads:
  data/processed/pr_dataset_structural.csv
  data/processed/pr_nlp_features.csv
  data/processed/comments_scored.csv       (for RQ2 temporal analysis)

Produces (in results/):
  descriptive_stats.txt
  regression_results.txt
  figures/01_toxicity_by_repo.png
  figures/02_sentiment_distribution.png
  figures/03_toxicity_vs_merged.png
  figures/04_toxicity_vs_returned90.png
  figures/05_newcomer_vs_experienced.png
  figures/06a_coef_merge.png
  figures/06b_coef_retention90.png
  figures/07_temporal_sentiment.png

Analyses performed
──────────────────
RQ1  Contributor retention (30 / 90 / 180 days)
     Logistic regression:
     returned_N ~ max_toxicity_x100 + sentiment + anger + politeness
     + history + controls + repo fixed effects

RQ2  Temporal emotional change before departure
     For leavers (returned_180 == 0): compare mean sentiment of their own
     comments in their last 3 PRs vs earlier PRs (Mann-Whitney U test).

RQ3  Maintainer politeness moderation
     Interaction term: max_toxicity_x100 × mean_politeness in the retention model.

RQ4  Newcomer sensitivity
     Interaction term: max_toxicity_x100 × is_newcomer.

RQ5  PR merge outcome
     Logistic regression:
     merged ~ max_toxicity_x100 + sentiment + anger + politeness
     + history + controls + repo fixed effects

Robustness checks
─────────────────
1. Author fixed-effects models (C(author)) for:
   - RQ5 merge outcome
   - RQ1 90-day retention

2. Binary toxicity indicators:
   - any_toxic_comment_005 = 1[max_toxicity >= 0.05]
   - any_toxic_comment_010 = 1[max_toxicity >= 0.10]

Usage:
    python scripts/05_analysis.py
"""

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from config import RETENTION_HORIZONS, PROCESSED_DIR, RESULTS_DIR

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

STRUCT_FILE = PROCESSED_DIR / "pr_dataset_structural.csv"
NLP_FILE = PROCESSED_DIR / "pr_nlp_features.csv"
SCORED_FILE = PROCESSED_DIR / "comments_scored.csv"
STATS_OUT = RESULTS_DIR / "descriptive_stats.txt"
REGR_OUT = RESULTS_DIR / "regression_results.txt"
FIG_DIR = RESULTS_DIR / "figures"

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

# Main toxicity variable used in plots.
TOX_PLOT_VAR = "max_toxicity"
# Rescaled toxicity variable used in regressions.
TOX_REG_VAR = "max_toxicity_x100"

# Main controls for baseline models.
_CONTROLS = (
    "log_num_comments_received + num_participants + "
    "log_prior_pr_count + log_first_response_delay_hours + "
    "is_newcomer + C(repo)"
)

# Controls for author fixed-effects robustness models.
# We omit is_newcomer because author FE absorbs much of the contributor-specific baseline.
_CONTROLS_FE = (
    "log_num_comments_received + num_participants + "
    "log_prior_pr_count + log_first_response_delay_hours + "
    "C(repo) + C(author)"
)

_NLP_VARS = (
    f"{TOX_REG_VAR} + mean_sentiment + mean_anger + mean_politeness"
)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Returns (df, scored_comments) where df is the merged analysis dataset.
    """
    if not STRUCT_FILE.exists():
        log.error("Missing %s. Run 03_build_dataset.py first.", STRUCT_FILE)
        sys.exit(1)
    if not NLP_FILE.exists():
        log.error("Missing %s. Run 04_score_text.py first.", NLP_FILE)
        sys.exit(1)

    structural = pd.read_csv(STRUCT_FILE)
    nlp = pd.read_csv(NLP_FILE)

    df = structural.merge(nlp, on=["repo", "pr_number"], how="inner")

    # Coerce types.
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
    df["merged"] = pd.to_numeric(df["merged"], errors="coerce").astype(int)

    for n in RETENTION_HORIZONS:
        col = f"returned_{n}"
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Log transforms for skewed variables.
    df["log_prior_pr_count"] = np.log1p(df["prior_pr_count"])
    df["log_num_comments_received"] = np.log1p(df["num_comments_received"])
    df["log_first_response_delay_hours"] = np.log1p(
        pd.to_numeric(df["first_response_delay_hours"], errors="coerce").clip(lower=0)
    )

    # Rescaled / derived toxicity variables for interpretability.
    df["max_toxicity_x100"] = df["max_toxicity"] * 100.0
    df["any_toxic_comment_005"] = (df["max_toxicity"] >= 0.05).astype(int)
    df["any_toxic_comment_010"] = (df["max_toxicity"] >= 0.10).astype(int)

    scored_cmts = None
    if SCORED_FILE.exists():
        scored_cmts = pd.read_csv(SCORED_FILE)
        scored_cmts["created_at"] = pd.to_datetime(
            scored_cmts["created_at"], utc=True, errors="coerce"
        )

    return df, scored_cmts


# ── Descriptive statistics ────────────────────────────────────────────────────

def write_descriptive_stats(df: pd.DataFrame, sink) -> None:
    cols = [
        "merged",
        "mean_toxicity", "max_toxicity", "max_toxicity_x100",
        "any_toxic_comment_005", "any_toxic_comment_010",
        "mean_sentiment", "mean_anger", "mean_politeness",
        "num_comments_received", "num_participants",
        "first_response_delay_hours",
        "prior_pr_count", "is_newcomer",
        "returned_30", "returned_90", "returned_180",
    ]
    cols = [c for c in cols if c in df.columns]
    desc = df[cols].describe(percentiles=[0.25, 0.5, 0.75]).T

    print("=" * 70, file=sink)
    print("DESCRIPTIVE STATISTICS", file=sink)
    print("=" * 70, file=sink)
    print(f"Total observations: {len(df)}", file=sink)
    print(f"Repositories: {df['repo'].nunique()}", file=sink)
    print(f"Unique PR authors: {df['author'].nunique()}", file=sink)
    print("", file=sink)
    print(desc.to_string(), file=sink)
    print("", file=sink)

    print("── By repository ──", file=sink)
    repo_summary = df.groupby("repo").agg(
        n=("pr_number", "count"),
        pct_merged=("merged", "mean"),
        mean_max_toxicity=("max_toxicity", "mean"),
        pct_toxic_005=("any_toxic_comment_005", "mean"),
        pct_toxic_010=("any_toxic_comment_010", "mean"),
        mean_sentiment=("mean_sentiment", "mean"),
        pct_newcomer=("is_newcomer", "mean"),
    )
    repo_summary["pct_merged"] = repo_summary["pct_merged"].map("{:.1%}".format)
    repo_summary["pct_toxic_005"] = repo_summary["pct_toxic_005"].map("{:.1%}".format)
    repo_summary["pct_toxic_010"] = repo_summary["pct_toxic_010"].map("{:.1%}".format)
    repo_summary["pct_newcomer"] = repo_summary["pct_newcomer"].map("{:.1%}".format)
    print(repo_summary.to_string(), file=sink)
    print("", file=sink)


# ── Figures ───────────────────────────────────────────────────────────────────

def fig_toxicity_by_repo(df: pd.DataFrame) -> None:
    plot_df = df.copy()
    plot_df["log_max_toxicity_x1000"] = np.log1p(plot_df["max_toxicity"] * 1000)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=plot_df, x="repo", y="log_max_toxicity_x1000", ax=ax, width=0.5)
    ax.set_title("Distribution of max toxicity by repository")
    ax.set_xlabel("Repository")
    ax.set_ylabel("log(1 + max_toxicity × 1000)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_toxicity_by_repo.png")
    plt.close(fig)


def fig_sentiment_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, title in zip(
        axes,
        ["mean_sentiment", "max_toxicity_x100"],
        ["Mean sentiment score", "Max toxicity × 100"],
    ):
        sns.histplot(df[col].dropna(), bins=40, kde=True, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(col)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_sentiment_distribution.png")
    plt.close(fig)


def fig_toxicity_vs_outcome(df: pd.DataFrame, outcome: str, label: str, fname: str) -> None:
    plot_df = df[[outcome, "max_toxicity_x100"]].dropna().copy()
    plot_df[outcome] = plot_df[outcome].astype(int)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=plot_df, x=outcome, y="max_toxicity_x100", ax=ax, width=0.4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No (0)", "Yes (1)"])
    ax.set_title(f"Max toxicity by {label}")
    ax.set_xlabel(label)
    ax.set_ylabel("Max toxicity × 100")
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname)
    plt.close(fig)


def fig_newcomer_vs_experienced(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left panel: rescaled toxicity by newcomer status.
    plot_left = df[["is_newcomer", "max_toxicity_x100"]].dropna().copy()
    sns.boxplot(data=plot_left, x="is_newcomer", y="max_toxicity_x100", ax=axes[0], width=0.4)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Experienced", "Newcomer"])
    axes[0].set_title("Max toxicity received")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Max toxicity × 100")

    # Right panel: mean return rate by newcomer status with CI.
    plot_right = df[["is_newcomer", "returned_90"]].dropna().copy()
    summary = plot_right.groupby("is_newcomer")["returned_90"].agg(["mean", "count", "std"]).reset_index()
    summary["se"] = np.sqrt(summary["mean"] * (1 - summary["mean"]) / summary["count"])
    summary["lo"] = (summary["mean"] - 1.96 * summary["se"]).clip(lower=0)
    summary["hi"] = (summary["mean"] + 1.96 * summary["se"]).clip(upper=1)

    axes[1].bar(summary["is_newcomer"], summary["mean"], yerr=1.96 * summary["se"], capsize=4)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Experienced", "Newcomer"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Returned within 90 days")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Mean return probability")

    fig.suptitle("Newcomer vs. experienced contributor comparison")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_newcomer_vs_experienced.png")
    plt.close(fig)


def fig_coefficient_plot(result, title: str, fname: str) -> None:
    """Plot odds ratios with 95% confidence intervals."""
    params = result.params.drop("Intercept", errors="ignore")

    # Hide FE dummies to keep plot readable.
    mask = ~params.index.str.startswith("C(repo)")
    mask &= ~params.index.str.startswith("C(author)")
    params = params[mask]

    conf = result.conf_int().loc[params.index]
    or_ = np.exp(params)
    or_lo = np.exp(conf[0])
    or_hi = np.exp(conf[1])

    fig, ax = plt.subplots(figsize=(7, max(3, 0.4 * len(params))))
    y_pos = range(len(params))
    ax.errorbar(
        or_,
        y_pos,
        xerr=[or_ - or_lo, or_hi - or_],
        fmt="o",
        capsize=4,
    )
    ax.axvline(1.0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(params.index)
    ax.set_xlabel("Odds ratio (95% CI)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname)
    plt.close(fig)


# ── Regression helpers ────────────────────────────────────────────────────────

def authors_with_multiple_prs(df: pd.DataFrame, min_prs: int = 2) -> pd.DataFrame:
    counts = df["author"].value_counts()
    keep = counts[counts >= min_prs].index
    return df[df["author"].isin(keep)].copy()


def run_logit(formula: str, data: pd.DataFrame, label: str, sink) -> object | None:
    """
    Fit logistic regression with HC3 robust SEs, print summary, return result.
    """
    log.info("Fitting: %s", label)
    try:
        result = smf.logit(formula, data=data).fit(cov_type="HC3", disp=False)
    except Exception as exc:
        print("=" * 70, file=sink)
        print(f"MODEL: {label}", file=sink)
        print(formula, file=sink)
        print("-" * 70, file=sink)
        print(f"Model failed: {exc}", file=sink)
        print("", file=sink)
        log.error("Model failed for %s: %s", label, exc)
        return None

    n_used = int(result.nobs)
    print("=" * 70, file=sink)
    print(f"MODEL: {label}  (N={n_used})", file=sink)
    print(formula, file=sink)
    print("-" * 70, file=sink)
    print(result.summary2().tables[1].to_string(), file=sink)
    print(f"Pseudo-R² (McFadden): {result.prsquared:.4f}", file=sink)
    print("", file=sink)
    return result


def check_vif(df: pd.DataFrame, predictors: list[str], sink) -> None:
    """Print VIFs to flag multicollinearity."""
    sub = df[predictors].dropna()
    if sub.empty:
        return
    X = sub.values
    print("── VIF check ──", file=sink)
    for i, col in enumerate(predictors):
        try:
            vif = variance_inflation_factor(X, i)
        except Exception:
            vif = float("nan")
        print(f"  {col:35s} VIF = {vif:.2f}", file=sink)
    print("", file=sink)


# ── RQ2: Temporal analysis ────────────────────────────────────────────────────

def rq2_temporal_analysis(df: pd.DataFrame, scored_cmts: pd.DataFrame | None, sink) -> None:
    """
    Compare sentiment of a contributor's own comments in their last 3 PRs
    (before departure) against their earlier PRs.
    """
    print("=" * 70, file=sink)
    print("RQ2: TEMPORAL EMOTIONAL CHANGE BEFORE DEPARTURE", file=sink)
    print("=" * 70, file=sink)

    if scored_cmts is None:
        print("scored_cmts not available — skipping RQ2.\n", file=sink)
        return

    author_max_return = (
        df.groupby("author")["returned_180"]
        .max()
        .reset_index()
        .rename(columns={"returned_180": "any_return"})
    )
    leavers = set(author_max_return[author_max_return["any_return"] == 0]["author"])
    log.info("RQ2: %d leavers identified.", len(leavers))

    if len(leavers) < 10:
        print(f"Too few leavers ({len(leavers)}) for reliable analysis.\n", file=sink)
        return

    leaver_prs = (
        df[df["author"].isin(leavers)][["author", "repo", "pr_number", "created_at"]]
        .copy()
    )
    leaver_prs["created_at"] = pd.to_datetime(leaver_prs["created_at"], utc=True)
    leaver_prs = leaver_prs.sort_values(["author", "repo", "created_at"])

    leaver_prs["pr_rank"] = leaver_prs.groupby(["author", "repo"]).cumcount() + 1
    leaver_prs["pr_count"] = leaver_prs.groupby(["author", "repo"])["pr_rank"].transform("max")
    leaver_prs = leaver_prs[leaver_prs["pr_count"] >= 4]

    if leaver_prs.empty:
        print("Not enough multi-PR leavers for temporal analysis.\n", file=sink)
        return

    leaver_prs["phase"] = np.where(
        leaver_prs["pr_rank"] > leaver_prs["pr_count"] - 3,
        "last_3",
        "earlier",
    )

    pr_scored = leaver_prs.merge(
        scored_cmts[["repo", "pr_number", "author", "sentiment"]],
        on=["repo", "pr_number"],
        suffixes=("_pr", ""),
    )

    own_cmts = pr_scored[pr_scored["author"] == pr_scored["author_pr"]].dropna(subset=["sentiment"])

    if own_cmts.empty:
        print("No scored own-comments for leavers — skipping RQ2.\n", file=sink)
        return

    early = own_cmts[own_cmts["phase"] == "earlier"]["sentiment"]
    late = own_cmts[own_cmts["phase"] == "last_3"]["sentiment"]

    if len(early) == 0 or len(late) == 0:
        print("Insufficient earlier/late sentiment observations for RQ2.\n", file=sink)
        return

    stat, pval = stats.mannwhitneyu(late, early, alternative="two-sided")

    print(f"  Leavers with ≥4 PRs: {leaver_prs['author'].nunique()}", file=sink)
    print(f"  Earlier phase — N={len(early)}, mean={early.mean():.4f}, median={early.median():.4f}", file=sink)
    print(f"  Last-3 phase  — N={len(late)},  mean={late.mean():.4f}, median={late.median():.4f}", file=sink)
    print(f"  Mann-Whitney U = {stat:.1f}, p = {pval:.4f}", file=sink)

    if pval < 0.05:
        direction = "more negative" if late.mean() < early.mean() else "more positive"
        print(f"  Significant: leavers' final comments were {direction} (p<0.05).", file=sink)
    else:
        print("  Not significant (p≥0.05).", file=sink)
    print("", file=sink)

    fig, ax = plt.subplots(figsize=(6, 4))
    plot_df = own_cmts[["phase", "sentiment"]].copy()
    sns.boxplot(data=plot_df, x="phase", y="sentiment", ax=ax, order=["earlier", "last_3"], width=0.4)
    ax.set_title("Leavers' own comment sentiment: earlier vs last 3 PRs")
    ax.set_xlabel("Phase")
    ax.set_ylabel("Sentiment score")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_temporal_sentiment.png")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    df, scored_cmts = load_data()
    log.info("Analysis dataset: %d rows, %d columns.", *df.shape)

    with open(STATS_OUT, "w") as f:
        write_descriptive_stats(df, f)
    log.info("Descriptive stats → %s", STATS_OUT)

    log.info("Generating figures …")
    fig_toxicity_by_repo(df)
    fig_sentiment_distribution(df)
    fig_toxicity_vs_outcome(df, "merged", "PR merged", "03_toxicity_vs_merged.png")
    fig_toxicity_vs_outcome(df, "returned_90", "Returned (90 days)", "04_toxicity_vs_returned90.png")
    fig_newcomer_vs_experienced(df)

    with open(REGR_OUT, "w") as f:
        # VIF check on main NLP variables.
        nlp_cols = [TOX_REG_VAR, "mean_sentiment", "mean_anger", "mean_politeness"]
        check_vif(df, [c for c in nlp_cols if c in df.columns], f)

        # ── RQ5: PR merge outcome ─────────────────────────────────────────────
        r_merge = run_logit(
            f"merged ~ {_NLP_VARS} + {_CONTROLS}",
            df,
            "RQ5 — PR merge outcome",
            f,
        )
        if r_merge is not None:
            fig_coefficient_plot(
                r_merge,
                "RQ5: Odds ratios for PR merge",
                "06a_coef_merge.png",
            )

        # ── RQ1: Contributor retention (30 / 90 / 180 days) ─────────────────
        for n in RETENTION_HORIZONS:
            col = f"returned_{n}"
            if col not in df.columns:
                continue
            run_logit(
                f"{col} ~ {_NLP_VARS} + {_CONTROLS}",
                df,
                f"RQ1 — Contributor retention ({n} days)",
                f,
            )

        r90 = run_logit(
            f"returned_90 ~ {_NLP_VARS} + {_CONTROLS}",
            df,
            "RQ1 — Retention 90d (for plot)",
            f,
        )
        if r90 is not None:
            fig_coefficient_plot(
                r90,
                "RQ1: Odds ratios for 90-day retention",
                "06b_coef_retention90.png",
            )

        # ── RQ3: Politeness moderation ───────────────────────────────────────
        run_logit(
            f"returned_90 ~ {TOX_REG_VAR} * mean_politeness + "
            f"mean_sentiment + mean_anger + {_CONTROLS}",
            df,
            "RQ3 — Maintainer politeness moderation",
            f,
        )

        # ── RQ4: Newcomer sensitivity ────────────────────────────────────────
        run_logit(
            f"returned_90 ~ {TOX_REG_VAR} * is_newcomer + "
            f"mean_sentiment + mean_anger + mean_politeness + {_CONTROLS}",
            df,
            "RQ4 — Newcomer sensitivity (interaction)",
            f,
        )

        # ── Robustness: binary toxicity indicators ───────────────────────────
        run_logit(
            "merged ~ any_toxic_comment_005 + mean_sentiment + mean_anger + "
            f"mean_politeness + {_CONTROLS}",
            df,
            "RQ5 — PR merge outcome (binary toxicity >= 0.05)",
            f,
        )

        run_logit(
            "returned_90 ~ any_toxic_comment_005 + mean_sentiment + mean_anger + "
            f"mean_politeness + {_CONTROLS}",
            df,
            "RQ1 — Retention 90d (binary toxicity >= 0.05)",
            f,
        )

        run_logit(
            "merged ~ any_toxic_comment_010 + mean_sentiment + mean_anger + "
            f"mean_politeness + {_CONTROLS}",
            df,
            "RQ5 — PR merge outcome (binary toxicity >= 0.10)",
            f,
        )

        run_logit(
            "returned_90 ~ any_toxic_comment_010 + mean_sentiment + mean_anger + "
            f"mean_politeness + {_CONTROLS}",
            df,
            "RQ1 — Retention 90d (binary toxicity >= 0.10)",
            f,
        )

        # ── Robustness: author fixed effects ─────────────────────────────────
        df_fe = authors_with_multiple_prs(df, min_prs=2)
        print("=" * 70, file=f)
        print("ROBUSTNESS CHECKS — AUTHOR FIXED EFFECTS", file=f)
        print("=" * 70, file=f)
        print(f"Authors with >=2 PRs: {df_fe['author'].nunique()}", file=f)
        print(f"Observations in FE sample: {len(df_fe)}", file=f)
        print("", file=f)

        run_logit(
            f"merged ~ {_NLP_VARS} + {_CONTROLS_FE}",
            df_fe,
            "RQ5 — PR merge outcome (author fixed effects)",
            f,
        )

        run_logit(
            f"returned_90 ~ {_NLP_VARS} + {_CONTROLS_FE}",
            df_fe,
            "RQ1 — Retention 90d (author fixed effects)",
            f,
        )

        # ── RQ2: Temporal analysis ───────────────────────────────────────────
        rq2_temporal_analysis(df, scored_cmts, f)

    log.info("Regression results → %s", REGR_OUT)
    log.info("Figures saved to %s/", FIG_DIR)
    log.info("Analysis complete.")


if __name__ == "__main__":
    main()