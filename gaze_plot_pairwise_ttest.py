import os
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display
from scipy import stats
from statsmodels.stats.multitest import multipletests

import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import chi2



# Figure of pairwise comparison per metrics 

METRIC_SUFFIXES: List[str] = [
    "gaze_duration",
    "fixation_duration",
    "fixation_count",
    "saccade_length",
    "saccade_count",
    "pupil_diameter",
    "pupil_diameter_fixation",
]

LOG_EPS = 1e-6
DEFAULT_NO_ANSWER = "no_answer"
DEFAULT_REASONING_AOIS: Set[str] = {"answer_reasoning", "reasoning", "reasoning_aoi"}
DEFAULT_REASONING_CONDITIONS: Set[str] = {"correct_reasoning", "incorrect_reasoning"}


def _detect_aois_and_metrics(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    aois: Set[str] = set()
    metrics: Set[str] = set()
    for col in df.columns:
        for suf in METRIC_SUFFIXES:
            if col.endswith("_" + suf):
                aois.add(col[: -(len(suf) + 1)])
                metrics.add(suf)
    return sorted(aois), sorted(metrics)


def _pick_conditions(
    available_conditions: Sequence[str],
    plot_conditions_mode: int,
    no_answer_label: str,
    plot_condition_order: Optional[Sequence[str]],
) -> List[str]:
    if plot_conditions_mode not in (3, 4):
        raise ValueError("plot_conditions_mode must be 3 or 4")

    available = [str(c) for c in available_conditions if pd.notna(c)]
    avail_set = set(available)

    if plot_condition_order is not None:
        ordered = [c for c in plot_condition_order if c in avail_set]
    else:
        ordered = sorted(avail_set - {no_answer_label})
        if plot_conditions_mode == 4 and no_answer_label in avail_set:
            ordered.append(no_answer_label)

    if plot_conditions_mode == 3:
        ordered = [c for c in ordered if c != no_answer_label]

    return ordered


def _melt_one_metric(df: pd.DataFrame, aois: List[str], metric: str, use_log10: bool) -> pd.DataFrame:
    required = {"participant_id", "trial_num", "classified_condition"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    metric_cols = [f"{a}_{metric}" for a in aois if f"{a}_{metric}" in df.columns]
    if not metric_cols:
        raise ValueError(f"No columns found for metric={metric} (expected <AOI>_{metric})")

    tmp = df[["participant_id", "trial_num", "classified_condition"] + metric_cols].copy()
    long_df = tmp.melt(
        id_vars=["participant_id", "trial_num", "classified_condition"],
        value_vars=metric_cols,
        var_name="AOI_col",
        value_name="value",
    )
    long_df["AOI"] = long_df["AOI_col"].str.replace("_" + metric, "", regex=False)
    long_df = long_df.drop(columns=["AOI_col"])
    long_df["participant_id"] = long_df["participant_id"].astype(str)
    long_df = long_df.dropna(subset=["value", "classified_condition", "AOI"]).copy()

    if use_log10:
        long_df["value"] = np.log10(long_df["value"].astype(float) + LOG_EPS)

    return long_df


def _subject_means(long_df: pd.DataFrame) -> pd.DataFrame:
    subj = (
        long_df.groupby(["participant_id", "AOI", "classified_condition"], as_index=False)["value"]
        .mean()
        .rename(columns={"value": "subj_mean"})
    )
    return subj


def _summarize_group(subj_means: pd.DataFrame) -> pd.DataFrame:
    summary = (
        subj_means.groupby(["AOI", "classified_condition"], as_index=False)
        .agg(mean=("subj_mean", "mean"), sd=("subj_mean", "std"), n=("subj_mean", "count"))
    )
    summary["sem"] = summary["sd"] / np.sqrt(summary["n"].clip(lower=1))
    return summary


def _paired_ttests_per_aoi(
    subj_means: pd.DataFrame,
    reasoning_aois: Set[str],
    reasoning_conditions: Set[str],
    conditions_to_compare: Set[str],
    alpha: float,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for aoi in sorted(subj_means["AOI"].unique()):
        sub = subj_means[subj_means["AOI"] == aoi]
        all_conds = sorted(sub["classified_condition"].unique())
        # Key rule:
        # - reasoning AOIs: ONLY compare reasoning conditions (e.g., correct vs incorrect reasoning)
        # - non-reasoning AOIs: compare the selected plotting conditions (3-mode or 4-mode)
        allowed_set = reasoning_conditions if aoi in reasoning_aois else conditions_to_compare
        allowed = [c for c in all_conds if c in allowed_set]
        if len(allowed) < 2:
            continue

        wide = sub.pivot_table(index="participant_id", columns="classified_condition", values="subj_mean")

        for c1, c2 in combinations(allowed, 2):
            if c1 not in wide.columns or c2 not in wide.columns:
                continue
            paired = wide[[c1, c2]].dropna()
            if len(paired) < 2:
                continue

            diff = paired[c1] - paired[c2]
            if np.allclose(diff.values, 0):
                t_stat, p_val = 0.0, 1.0
            else:
                t_stat, p_val = stats.ttest_rel(paired[c1], paired[c2])
                if not np.isfinite(p_val):
                    t_stat, p_val = 0.0, 1.0

            mean1 = float(paired[c1].mean())
            mean2 = float(paired[c2].mean())
            mean_diff = float(diff.mean())
            sd_diff = float(diff.std(ddof=1)) if len(diff) > 1 else np.nan
            n = int(len(diff))
            cohens_dz = mean_diff / sd_diff if sd_diff and np.isfinite(sd_diff) and sd_diff != 0 else 0.0

            rows.append(
                {
                    "AOI": aoi,
                    "group1": c1,
                    "group2": c2,
                    "t_stat": float(t_stat),
                    "p_raw": float(p_val),
                    "n": n,
                    "mean1": mean1,
                    "mean2": mean2,
                    "mean_diff": mean_diff,
                    "cohens_dz": float(cohens_dz),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(
            columns=[
                "AOI",
                "group1",
                "group2",
                "t_stat",
                "p_raw",
                "p_fdr",
                "reject_fdr",
                "n",
                "mean1",
                "mean2",
                "mean_diff",
                "cohens_dz",
            ]
        )

    out["p_fdr"] = 1.0
    out["reject_fdr"] = False
    valid = out["p_raw"].notna() & np.isfinite(out["p_raw"])
    if valid.any():
        reject, p_fdr, _, _ = multipletests(out.loc[valid, "p_raw"].values, alpha=alpha, method="fdr_bh")
        out.loc[valid, "p_fdr"] = p_fdr
        out.loc[valid, "reject_fdr"] = reject

    return out


def _p_to_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _draw_sig_bracket(ax, x1, x2, y, h, text, lw=1.2):
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=lw, c="k")
    ax.text((x1 + x2) / 2, y + h, text, ha="center", va="bottom")


def _plot_bar_sem_with_sig(
    means_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    metric: str,
    alpha: float,
    p_col: str,
    use_log10: bool,
    sig_label: str,
    cond_order: Sequence[str],
):
    aoi_order = sorted(means_df["AOI"].unique())
    cond_order = [c for c in cond_order if c in set(means_df["classified_condition"].unique())]
    width = 0.8 / max(1, len(cond_order))
    x_pos = np.arange(len(aoi_order))

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, cond in enumerate(cond_order):
        sub = means_df[means_df["classified_condition"] == cond].set_index("AOI").reindex(aoi_order)
        heights = sub["mean"].fillna(0)
        errs = sub["sem"].fillna(0)
        ax.bar(x_pos + i * width, heights, width=width, yerr=errs, capsize=3, label=cond)

    ax.set_xticks(x_pos + (len(cond_order) - 1) * width / 2)
    ax.set_xticklabels(aoi_order, rotation=20)
    ylabel = f"{metric} (log10 scale, subj mean ± SEM)" if use_log10 else f"{metric} (subj mean ± SEM)"
    ax.set_ylabel(ylabel)
    ax.set_title(f"{metric}: AOI × condition (participant means, {sig_label} sig)")
    ax.legend(title="classified_condition", bbox_to_anchor=(1.02, 1), loc="upper left")

    y_tops = (
        means_df.assign(top=means_df["mean"] + means_df["sem"]).groupby("AOI")["top"].max().to_dict()
    )

    if pairwise_df is not None and not pairwise_df.empty and p_col in pairwise_df.columns:
        sig = pairwise_df[pairwise_df[p_col] < alpha].copy()
        y_min, y_max = ax.get_ylim()
        level_gap = 0.06 * (y_max - y_min) if y_max > y_min else 1.0

        for aoi in aoi_order:
            sub_sig = sig[sig["AOI"] == aoi].sort_values(p_col)
            for k, (_, row) in enumerate(sub_sig.iterrows()):
                g1, g2 = row["group1"], row["group2"]
                if g1 not in cond_order or g2 not in cond_order:
                    continue

                i1, i2 = cond_order.index(g1), cond_order.index(g2)
                x_base = x_pos[aoi_order.index(aoi)]
                x1 = x_base + i1 * width
                x2 = x_base + i2 * width
                base_y = float(y_tops.get(aoi, 0.0)) + (k + 1) * level_gap
                _draw_sig_bracket(ax, x1, x2, base_y, h=level_gap * 0.3, text=_p_to_stars(float(row[p_col])))

    fig.tight_layout()
    return fig, ax


def analyze_pairwise_ttests_per_trial(
    input_csv: str,
    metric: str,
    outdir: Optional[str] = None,
    alpha: float = 0.05,
    save_figs: bool = True,
    save_tables: bool = True,
    use_log10: bool = False,
    reasoning_aois: Optional[Set[str]] = None,
    reasoning_conditions: Optional[Set[str]] = None,
    plot_use_corrected_p: bool = True,
    plot_conditions_mode: int = 3,
    no_answer_label: str = DEFAULT_NO_ANSWER,
    plot_condition_order: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    df = pd.read_csv(input_csv)
    aois, metrics_all = _detect_aois_and_metrics(df)
    if metric not in metrics_all and not any(c.endswith("_" + metric) for c in df.columns):
        raise ValueError(f"No columns found for metric={metric} (expected <AOI>_{metric})")

    long_df = _melt_one_metric(df, aois, metric, use_log10=use_log10)

    plot_conditions = _pick_conditions(
        long_df["classified_condition"].dropna().unique(),
        plot_conditions_mode=plot_conditions_mode,
        no_answer_label=no_answer_label,
        plot_condition_order=plot_condition_order,
    )
    if not plot_conditions:
        raise ValueError("No conditions left after filtering")

    long_df = long_df[long_df["classified_condition"].isin(plot_conditions)].copy()

    reasoning_aois = set(reasoning_aois) if reasoning_aois is not None else set(DEFAULT_REASONING_AOIS)
    reasoning_conditions = set(reasoning_conditions) if reasoning_conditions is not None else set(DEFAULT_REASONING_CONDITIONS)

    subj_means = _subject_means(long_df)
    means_df = _summarize_group(subj_means)
    pairwise_df = _paired_ttests_per_aoi(
        subj_means,
        reasoning_aois,
        reasoning_conditions,
        conditions_to_compare=set(plot_conditions),
        alpha=alpha,
    )

    print(f"AOIs detected: {aois}")
    print(f"Metric: {metric}")
    print(f"Plot conditions ({plot_conditions_mode}): {plot_conditions}")

    if not pairwise_df.empty:
        sort_col = "p_fdr" if plot_use_corrected_p else "p_raw"
        display(pairwise_df.sort_values(["AOI", sort_col], na_position="last"))

    p_col_for_plot = "p_fdr" if plot_use_corrected_p else "p_raw"
    sig_label = "FDR" if plot_use_corrected_p else "raw p"

    fig_bar, _ = _plot_bar_sem_with_sig(
        means_df,
        pairwise_df,
        metric,
        alpha=alpha,
        p_col=p_col_for_plot,
        use_log10=use_log10,
        sig_label=sig_label,
        cond_order=plot_conditions,
    )
    plt.show(fig_bar)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        log_suffix = "_log10" if use_log10 else ""
        sig_suffix = "FDR" if plot_use_corrected_p else "raw"
        if save_figs:
            fig_bar.savefig(os.path.join(outdir, f"{metric}{log_suffix}__bar_sem_sig_{sig_suffix}.png"), dpi=200)
        if save_tables:
            means_df.to_csv(os.path.join(outdir, f"{metric}{log_suffix}__participant_means.csv"), index=False)
            pairwise_df.to_csv(os.path.join(outdir, f"{metric}{log_suffix}__pairwise_paired_ttests.csv"), index=False)

    return {"means_df": means_df, "pairwise_df": pairwise_df, "figs": {"bar_sem_sig": fig_bar}}







