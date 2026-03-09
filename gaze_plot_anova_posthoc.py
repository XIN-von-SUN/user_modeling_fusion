import os
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import chi2


DEFAULT_REASONING_CONDITIONS = {"correct_reasoning", "incorrect_reasoning"}

### overall MixLMM and posthoc comparison

# Balanced, visually clean, colorblind-friendly palettes
PRETTY_PALETTES = {
    "okabe_ito": [
        "#0072B2", "#009E73", "#D55E00", "#CC79A7", 
        "#F0E442", "#56B4E9", "#000000",
    ],
    "tol_bright": [
        "#4477AA", "#228833", "#CC6677", "#DDCC77", 
        "#AA3377", "#BBBBBB",
    ],
    "pastel_fresh": [
        "#7DB3FF", "#6DD39E", "#FF9F80", "#FFD66B", "#B495E1",
    ],
    "sunset_soft": [
        "#5B8FF9", "#5AD8A6", "#F6BD16", "#E8684A", "#6DC8EC",
    ],
    "lab": [
        '#7F7F7F', '#4E79A7', '#E15759'
    ]
}


# ===================== PLOT CONFIG (edit here) =====================
FIGSIZE = (21, 10)  # Figure size in inches (width, height)
FONT_SCALE = 1.0  # Global seaborn font scaling factor

BASE_FONT_SIZE = 2  # Reserved for manual global scaling if needed
AXES_LABEL_SIZE = 30  # Axis-label font size
AXES_TITLE_SIZE = 30  # Axes-title font size
SUPTITLE_SIZE = 30  # Reserved for figure suptitle font size

TICK_LABEL_SIZE = 28  # Tick-label font size
LEGEND_FONT_SIZE = 30  # Legend font size
STAT_BOX_FONT_SIZE = 28  # Font size of the per-AOI stats box
STAR_FONT_SIZE = 40  # Significance star font size

# Vertical offset of the per-AOI stats box.
# Defined as a fraction of max_height; larger values place it higher.
STAT_BOX_Y_OFFSET = 0.08

# --- Adjustable significance-bracket parameters ---
SIG_LINE_WIDTH = 2.4  # Bracket line width
SIG_TICK_LENGTH = 0.025  # Vertical tick length as a fraction of max_height
SIG_BRACKET_SPACING = 0.10  # Layer spacing between multiple brackets

# Vertical lift of the bracket above the bar top, as a fraction of max_height.
# Values around 0.12-0.20 are common; larger values place brackets higher.
sig_line_height = 0.07

# Vertical offset of stars relative to the bracket, as a fraction of max_height.
# Positive: above the bracket. Negative: closer to or below the bracket.
sig_star_offset = -0.040

TITLE_PAD = 60  # Distance between the title and plotting area

# --- Bar layout ---
BAR_WIDTH = 0.70  # Total width of each AOI group
BAR_OFFSET = -0.0  # Horizontal shift of the entire bar group

# Optional per-feature y-axis limits (override auto-scaling)
Y_AXIS_LIMITS: Dict[str, Tuple[float, float]] = {
    # "fixation_count": (0, 70),
    # "fixation_duration": (0, 14000),
    # "saccade_count": (0, 110),
    # "saccade_length": (0, 100),
    # "pupil_diameter": (0, 4.5),
    "ttff_ms": (0, 80000),

}


DEFAULT_FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")


def _sanitize_filename(name: str) -> str:
    name = str(name).strip().replace(" ", "_")
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    cleaned = "".join(ch if ch in allowed else "_" for ch in name)
    return cleaned or "figure"

def _resolve_palette(palette, n_colors):
    if isinstance(palette, (list, tuple)):
        colors = list(palette)
    elif isinstance(palette, str) and palette in PRETTY_PALETTES:
        colors = PRETTY_PALETTES[palette]
    else:
        colors = sns.color_palette(palette, n_colors=max(n_colors, 3))
    
    if len(colors) < n_colors:
        reps = (n_colors + len(colors) - 1) // len(colors)
        colors = (colors * reps)[:n_colors]
    else:
        colors = colors[:n_colors]
    return colors

# # Data path
# DATA_PATH = "../user_data/user_data_final/cleaned_gaze/eye_metrics_per_trial_all_participants_with_conditions.csv"

def create_gaze_anaglysis_all(
    DATA_PATH,
    feature,
    plot_mode: str = "reasoning",  # "reasoning" (v1) or "advice" (v2)
    aois=("claim_evidence","answer_aoi", "reasoning_aoi", "response_aoi"),
    aoi_labels=("Claim Evidence", "Answer", "Reasoning AOI", "Response AOI"),
    conditions=("no_reasoning", "correct_reasoning", "incorrect_reasoning"),
    palette="Set2",
    figsize=FIGSIZE,
    font_scale: float = FONT_SCALE,
    show_values=False,
    aoi_bg_colors=None,
    show_stats=True,  # Whether to display omnibus statistics.
    fdr_method='fdr_bh',  # FDR method: `fdr_bh` (Benjamini-Hochberg) or `fdr_by` (Benjamini-Yekutieli).
    show_pairwise=True,  # Whether to show pairwise comparisons.
    pairwise_p_mode: str = "fdr",  # Significance mode for pairwise stars: `fdr` (default) or `raw`.
    features: Optional[Sequence[str]] = None,  # Plot multiple features in one call, one figure per feature.
    y_axis_limits: Optional[Dict[str, Tuple[float, float]]] = None,  # Optional override for global Y_AXIS_LIMITS.
    auto_save: bool = False,
    save_dir: str = DEFAULT_FIGURES_DIR,
    save_name: Optional[str] = None,
    save_format: str = "png",
    save_dpi: int = 360,
    close_after_save: bool = False,
):
    """
    Create grouped bar plots with cleaner styling and AOI background shading.
    Includes FDR correction and uses corrected p-values for significance.
    """
    # Allow plotting multiple features in one call.
    # - If `features` is provided, it takes precedence.
    # - If `feature` itself is a list/tuple/set (non-str), treat it as multiple features.
    if features is None and isinstance(feature, (list, tuple, set)):
        features = [str(f) for f in feature]

    if features is not None:
        results: Dict[str, Tuple[plt.Figure, plt.Axes]] = {}
        for f in features:
            per_feature_save_name = None
            if save_name is not None:
                per_feature_save_name = f"{save_name}_{f}"
            fig_ax = create_gaze_anaglysis_all(
                DATA_PATH=DATA_PATH,
                feature=str(f),
                plot_mode=plot_mode,
                aois=aois,
                aoi_labels=aoi_labels,
                conditions=conditions,
                palette=palette,
                figsize=figsize,
                font_scale=font_scale,
                show_values=show_values,
                aoi_bg_colors=aoi_bg_colors,
                show_stats=show_stats,
                fdr_method=fdr_method,
                show_pairwise=show_pairwise,
                pairwise_p_mode=pairwise_p_mode,
                features=None,
                y_axis_limits=y_axis_limits,
                auto_save=auto_save,
                save_dir=save_dir,
                save_name=per_feature_save_name,
                save_format=save_format,
                save_dpi=save_dpi,
                close_after_save=close_after_save,
            )
            results[str(f)] = fig_ax
        return results

    plot_mode = str(plot_mode).lower().strip()
    if plot_mode == "v1":
        plot_mode = "reasoning"
    if plot_mode == "v2":
        plot_mode = "advice"
    if plot_mode not in {"reasoning", "advice"}:
        raise ValueError("plot_mode must be 'reasoning' (v1) or 'advice' (v2)")

    # If caller didn't override defaults, switch to advice(v2) spec automatically.
    if plot_mode == "advice":
        # conditions: no_answer instead of no_reasoning
        if tuple(conditions) == ("no_reasoning", "correct_reasoning", "incorrect_reasoning"):
            conditions = ("no_answer", "correct_reasoning", "incorrect_reasoning")

        # AOIs/labels:
        # Advice = answer_aoi + reasoning_aoi
        # Answer = answer_aoi
        if tuple(aois) == ("claim_evidence", "answer_aoi", "reasoning_aoi", "response_aoi"):
            aois = ("claim_evidence", "advice_aoi", "response_aoi", "answer_aoi")

        if tuple(aoi_labels) == ("Claim Evidence", "Answer", "Reasoning AOI", "Response AOI"):
            aoi_labels = ("Claim Evidence", "Advice", "Response AOI", "Answer")

    sns.set_theme(style="white", context="notebook", font_scale=font_scale)
    plt.rcParams.update(
        {
            "axes.labelsize": AXES_LABEL_SIZE,
            "axes.titlesize": AXES_TITLE_SIZE,
            "xtick.labelsize": TICK_LABEL_SIZE,
            "ytick.labelsize": TICK_LABEL_SIZE,
            "legend.fontsize": LEGEND_FONT_SIZE,
        }
    )
    df = pd.read_csv(DATA_PATH)
    colors = _resolve_palette(palette, n_colors=len(conditions))

    if aoi_bg_colors is None:
        aoi_bg_colors = ['#F8FAFB', '#F2F5F7', '#EDF1F4', '#E8ECF0']
        aoi_bg_colors = aoi_bg_colors[:len(aois)]

    # Compute means, SEMs, and sample counts.
    means = np.zeros((len(aois), len(conditions)))
    errors = np.zeros((len(aois), len(conditions)))
    sample_counts = np.zeros((len(aois), len(conditions)))

    for i, aoi in enumerate(aois):
        if aoi == "advice_aoi":
            col_answer = f"answer_aoi_{feature}"
            col_reasoning = f"reasoning_aoi_{feature}"
            if col_answer not in df.columns or col_reasoning not in df.columns:
                continue
            for j, cond in enumerate(conditions):
                sub_vals = df.loc[df["classified_condition"] == cond, [col_answer, col_reasoning]]
                vals = sub_vals.sum(axis=1, min_count=1).dropna()
                sample_counts[i, j] = len(vals)
                if len(vals) > 0:
                    means[i, j] = vals.mean()
                    errors[i, j] = vals.std(ddof=1) / np.sqrt(len(vals))
        else:
            col = f"{aoi}_{feature}"
            if col not in df.columns:
                continue
            for j, cond in enumerate(conditions):
                vals = df.loc[df["classified_condition"] == cond, col].dropna()
                sample_counts[i, j] = len(vals)
                if len(vals) > 0:
                    means[i, j] = vals.mean()
                    errors[i, j] = vals.std(ddof=1) / np.sqrt(len(vals))

    # Statistical analysis: participant-level mixed linear model.
    lmm_results = {}
    raw_p_values = []
    aoi_indices = []
    pairwise_results = {}  # Store pairwise-comparison results.
    
    if show_stats:
        for i, aoi in enumerate(aois):
            if aoi == "advice_aoi":
                col_answer = f"answer_aoi_{feature}"
                col_reasoning = f"reasoning_aoi_{feature}"
                if col_answer not in df.columns or col_reasoning not in df.columns:
                    continue
            else:
                col = f"{aoi}_{feature}"
                if col not in df.columns:
                    continue
            
            # Prepare data; requires a `participant_id` column.
            if 'participant_id' not in df.columns:
                print(f"Warning: 'participant_id' column not found. Skipping LMM analysis.")
                break
            
            # 提取并聚合到participant × condition层面
            if aoi == "advice_aoi":
                sub = df[['participant_id', 'classified_condition', col_answer, col_reasoning]].copy()
                sub['value'] = sub[col_answer] + sub[col_reasoning]
                sub = sub[['participant_id', 'classified_condition', 'value']]
            else:
                sub = df[['participant_id', 'classified_condition', col]].copy()
                sub = sub.rename(columns={col: 'value'})
            sub = sub.dropna(subset=['participant_id', 'classified_condition', 'value'])

            # advice(v2): Advice & Answer stats exclude no_answer
            if plot_mode == "advice" and aoi in {"advice_aoi", "answer_aoi"}:
                sub = sub[sub['classified_condition'] != 'no_answer']
            
            # 特殊处理：reasoning_aoi只比较correct vs incorrect reasoning
            if aoi == "reasoning_aoi":
                sub = sub[sub['classified_condition'].isin(['correct_reasoning', 'incorrect_reasoning'])]
            
            if sub.empty or sub['classified_condition'].nunique() < 2:
                continue
            
            # 聚合到participant-level (每个participant × condition的均值)
            agg = sub.groupby(
                ['participant_id', 'classified_condition'], 
                as_index=False
            )['value'].mean()
            
            # 对于reasoning_aoi，实际条件数应该是2（correct vs incorrect）
            n_conditions = agg['classified_condition'].nunique()
            n_participants = agg['participant_id'].nunique()
            
            try:
                # 拟合Null model (只有随机截距)
                null_model = smf.mixedlm(
                    'value ~ 1', 
                    data=agg, 
                    groups=agg['participant_id']
                )
                null_fit = null_model.fit(method='lbfgs', reml=False, maxiter=200, disp=False)
                
                # 拟合Alternative model (加入condition固定效应)
                alt_model = smf.mixedlm(
                    'value ~ C(classified_condition)', 
                    data=agg, 
                    groups=agg['participant_id']
                )
                alt_fit = alt_model.fit(method='lbfgs', reml=False, maxiter=200, disp=False)
                
                # Likelihood Ratio Test
                lrt_stat = 2.0 * (alt_fit.llf - null_fit.llf)
                df_diff = n_conditions - 1
                p_val = 1.0 - chi2.cdf(lrt_stat, df_diff)
                
                lmm_results[i] = {
                    'statistic': lrt_stat,
                    'df': df_diff,
                    'p_value': p_val,
                    'n_participants': n_participants,
                    'test_type': 'LRT'
                }
                raw_p_values.append(p_val)
                aoi_indices.append(i)
                
            except Exception as e:
                # 如果LMM失败，尝试简单的ANOVA作为备选
                try:
                    groups = [
                        agg[agg['classified_condition'] == cond]['value'].values
                        for cond in agg['classified_condition'].unique()
                    ]
                    f_stat, p_val = stats.f_oneway(*groups)
                    lmm_results[i] = {
                        'statistic': f_stat,
                        'df': n_conditions - 1,
                        'p_value': p_val,
                        'n_participants': n_participants,
                        'test_type': 'ANOVA_fallback'
                    }
                    raw_p_values.append(p_val)
                    aoi_indices.append(i)
                except:
                    continue
            
            # Pairwise比较 - 严格按照指定的conditions进行，使用配对t检验
            if show_pairwise and n_conditions >= 2:
                # 使用指定的conditions列表，而不是数据中实际存在的条件
                specified_conditions = []
                if aoi == "reasoning_aoi":
                    # reasoning_aoi特殊处理：只做 reasoning 条件之间的比较
                    # 目标：不产生任何涉及 no_reasoning / no_answer 的对比与显著性
                    specified_conditions = [c for c in conditions if c in DEFAULT_REASONING_CONDITIONS]
                elif plot_mode == "advice" and aoi in {"advice_aoi", "answer_aoi"}:
                    # advice(v2): Advice/Answer pairwise exclude no_answer
                    specified_conditions = [c for c in conditions if c != "no_answer"]
                else:
                    # 其他AOI：使用完整的条件列表
                    specified_conditions = list(conditions)
                
                pairwise_results[i] = []
                
                for cond1, cond2 in combinations(specified_conditions, 2):
                    # 创建宽格式数据以进行配对t检验
                    wide = agg.pivot_table(
                        index='participant_id', 
                        columns='classified_condition', 
                        values='value'
                    )
                    
                    # 检查这两个条件是否都存在且有配对数据
                    if cond1 not in wide.columns or cond2 not in wide.columns:
                        print(f"Warning: Missing condition data for {cond1} vs {cond2} in {aoi}. Skipping comparison.")
                        continue
                    
                    # 提取配对数据（同时拥有两个条件数据的participants）
                    paired_data = wide[[cond1, cond2]].dropna()
                    
                    if len(paired_data) < 2:
                        print(f"Warning: Insufficient paired data for {cond1} vs {cond2} in {aoi} (n={len(paired_data)}). Skipping comparison.")
                        continue
                    
                    # 配对t检验
                    try:
                        # 检查是否所有差值都为0（避免除零错误）
                        diff = paired_data[cond1] - paired_data[cond2]
                        if np.allclose(diff.values, 0):
                            t_stat, p_val = 0.0, 1.0
                        else:
                            t_stat, p_val = stats.ttest_rel(paired_data[cond1], paired_data[cond2])
                            # 检查结果是否有限
                            if not np.isfinite(p_val):
                                t_stat, p_val = 0.0, 1.0
                    except Exception as e:
                        print(f"Warning: t-test failed for {cond1} vs {cond2} in {aoi}: {e}")
                        t_stat, p_val = 0.0, 1.0
                    
                    pairwise_results[i].append({
                        'condition1': cond1,
                        'condition2': cond2,
                        't_statistic': float(t_stat),
                        'p_value': float(p_val),
                        'n_pairs': len(paired_data),  # 添加配对数量信息
                        'mean1': float(paired_data[cond1].mean()),
                        'mean2': float(paired_data[cond2].mean()),
                        'mean_diff': float(diff.mean())
                    })
    
    # 进行FDR校正 (主效应)
    if show_stats and raw_p_values:
        # 使用statsmodels的multipletests函数进行FDR校正
        reject, pvals_corrected, alpha_sidak, alpha_bonf = multipletests(
            raw_p_values, 
            alpha=0.05, 
            method=fdr_method
        )
        
        # 将校正后的p值更新到结果中
        for idx, aoi_idx in enumerate(aoi_indices):
            lmm_results[aoi_idx]['p_value_raw'] = lmm_results[aoi_idx]['p_value']
            lmm_results[aoi_idx]['p_value_corrected'] = pvals_corrected[idx]
            lmm_results[aoi_idx]['fdr_rejected'] = reject[idx]
        
        print(f"\n{'='*60}")
        print(f"MAIN EFFECTS ANALYSIS (LMM/ANOVA)")
        print(f"{'='*60}")
        print(f"FDR correction method: {fdr_method}")
        print(f"Number of tests: {len(raw_p_values)}")
        
        for idx, aoi_idx in enumerate(aoi_indices):
            result = lmm_results[aoi_idx]
            aoi_name = aois[aoi_idx] if aoi_idx < len(aois) else f"AOI_{aoi_idx}"
            
            print(f"\n{aoi_name.upper()}:")
            print(f"  Test type: {result['test_type']}")
            print(f"  Statistic: {result['statistic']:.4f}")
            print(f"  df: {result['df']}")
            print(f"  Raw p-value: {result['p_value_raw']:.6f}")
            print(f"  FDR-corrected p-value: {result['p_value_corrected']:.6f}")
            print(f"  Significant after FDR: {'YES' if result['fdr_rejected'] else 'NO'}")
            print(f"  Participants: {result['n_participants']}")
        
        print(f"\nSummary: {sum(reject)}/{len(raw_p_values)} tests significant after FDR correction")
    
    # 进行FDR校正 (pairwise比较)
    if show_pairwise and pairwise_results:
        all_pairwise_p = []
        pairwise_indices = []
        
        for aoi_idx, comparisons in pairwise_results.items():
            for comp_idx, comp in enumerate(comparisons):
                all_pairwise_p.append(comp['p_value'])
                pairwise_indices.append((aoi_idx, comp_idx))
        
        if all_pairwise_p:
            reject_pw, pvals_corrected_pw, _, _ = multipletests(
                all_pairwise_p, 
                alpha=0.05, 
                method=fdr_method
            )
            
            pairwise_p_mode = str(pairwise_p_mode).lower().strip()
            if pairwise_p_mode not in {"fdr", "raw"}:
                raise ValueError("pairwise_p_mode must be 'fdr' or 'raw'")

            # 将校正后的p值更新到pairwise结果中，并根据选项决定显著性依据
            for idx, (aoi_idx, comp_idx) in enumerate(pairwise_indices):
                pairwise_results[aoi_idx][comp_idx]['p_value_corrected'] = pvals_corrected_pw[idx]
                if pairwise_p_mode == "raw":
                    p_for_sig = float(pairwise_results[aoi_idx][comp_idx]['p_value'])
                    pairwise_results[aoi_idx][comp_idx]['p_value_for_sig'] = p_for_sig
                    pairwise_results[aoi_idx][comp_idx]['significant'] = bool(p_for_sig < 0.05)
                else:
                    p_for_sig = float(pvals_corrected_pw[idx])
                    pairwise_results[aoi_idx][comp_idx]['p_value_for_sig'] = p_for_sig
                    pairwise_results[aoi_idx][comp_idx]['significant'] = bool(reject_pw[idx])
            
            print(f"\n{'='*60}")
            print(f"PAIRWISE COMPARISONS (paired t-tests)")
            print(f"{'='*60}")
            print(f"FDR correction method: {fdr_method}")
            print(f"Pairwise significance mode: {pairwise_p_mode}")
            print(f"Total pairwise comparisons: {len(all_pairwise_p)}")
            
            for aoi_idx, comparisons in pairwise_results.items():
                if comparisons:  # 只显示有比较的AOI
                    aoi_name = aois[aoi_idx] if aoi_idx < len(aois) else f"AOI_{aoi_idx}"
                    print(f"\n{aoi_name.upper()}:")
                    
                    for comp in comparisons:
                        cond1 = comp['condition1'].replace('_', ' ').title()
                        cond2 = comp['condition2'].replace('_', ' ').title()
                        t_stat = comp['t_statistic']
                        p_raw = comp['p_value']
                        p_corr = comp['p_value_corrected']
                        significant = comp['significant']
                        p_for_sig = comp.get('p_value_for_sig', p_corr if pairwise_p_mode == 'fdr' else p_raw)
                        n_pairs = comp.get('n_pairs', 'N/A')
                        mean1 = comp.get('mean1', 'N/A')
                        mean2 = comp.get('mean2', 'N/A')
                        mean_diff = comp.get('mean_diff', 'N/A')
                        
                        print(f"  {cond1} vs {cond2} (paired):")
                        print(f"    t-statistic: {t_stat:.4f}")
                        print(f"    n pairs: {n_pairs}")
                        if mean1 != 'N/A' and mean2 != 'N/A':
                            print(f"    Mean {cond1}: {mean1:.4f}")
                            print(f"    Mean {cond2}: {mean2:.4f}")
                            print(f"    Mean difference: {mean_diff:.4f}")
                        print(f"    Raw p-value: {p_raw:.6f}")
                        print(f"    FDR-corrected p-value: {p_corr:.6f}")
                        if pairwise_p_mode == 'raw':
                            print(f"    Significant (raw p < 0.05): {'YES' if significant else 'NO'}")
                        else:
                            print(f"    Significant after FDR: {'YES' if significant else 'NO'}")
            
            if pairwise_p_mode == 'raw':
                significant_pw = sum(
                    int(comp.get('significant', False))
                    for comps in pairwise_results.values()
                    for comp in comps
                )
                print(f"\nSummary: {significant_pw}/{len(all_pairwise_p)} pairwise comparisons significant (raw p < 0.05)")
            else:
                significant_pw = sum(reject_pw)
                print(f"\nSummary: {significant_pw}/{len(all_pairwise_p)} pairwise comparisons significant after FDR correction")
            print(f"{'='*60}\n")

    # 作图
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(aoi_labels))
    group_width = float(BAR_WIDTH)
    width = group_width / max(len(conditions), 1)

    # 添加AOI背景色块
    for i in range(len(aoi_labels)):
        ax.axvspan(
            i - 0.42, i + 0.42, 
            facecolor=aoi_bg_colors[i], 
            alpha=0.6, zorder=0, linewidth=0
        )

    # 绘制条形图
    max_height = 0
    for j, cond in enumerate(conditions):
        offset = width * (j - (len(conditions) - 1) / 2) - float(BAR_OFFSET)
        bars = ax.bar(
            x + offset, means[:, j],
            width=width, yerr=errors[:, j],
            label=cond.replace("_", " ").title(),
            color=colors[j], edgecolor='white',
            linewidth=1.2, capsize=4,
            error_kw={"linewidth": 1.2, "ecolor": "#555555", 
                     "elinewidth": 1.2, "capthick": 1.2},
            zorder=3, alpha=0.9
        )
        
        # 记录最大高度
        for i in range(len(aois)):
            if sample_counts[i, j] > 0:
                max_height = max(max_height, means[i, j] + errors[i, j])
        
        # 为每个条形添加标记
        for i, bar in enumerate(bars):
            h = bar.get_height()
            bar_x = bar.get_x()
            bar_width = bar.get_width()
            
            # 如果没有数据，添加"N/A"标记
            if sample_counts[i, j] == 0:
                na_height = max_height * 0.15 if max_height > 0 else 1
                ax.bar(
                    bar_x + bar_width/2, na_height,
                    width=bar_width, color='none',
                    edgecolor=colors[j], linewidth=1.2,
                    hatch='///', alpha=0.5, zorder=2
                )
                ax.text(
                    bar_x + bar_width / 2, na_height * 0.5,
                    'N/A', ha="center", va="center",
                    fontsize=8, fontweight='bold',
                    color=colors[j], alpha=0.8
                )
            elif show_values:
                ax.text(
                    bar_x + bar_width / 2,
                    h + 0.02 * max_height,
                    f"{h:.1f}", ha="center", va="bottom",
                    fontsize=8, fontweight='medium',
                    color="#2C3E50",
                )

    # 存储每个AOI的最高bracket高度
    aoi_max_bracket_heights = {}
    
    # 添加pairwise比较的可视化标记
    def add_pairwise_brackets(ax, aoi_idx, comparisons, means_row, errors_row, conditions, x_pos, width, max_height):
        """添加pairwise比较的bracket和显著性标记"""
        if not comparisons:
            return 0  # 返回0表示没有brackets
        
        # 过滤出显著的比较
        significant_comparisons = [comp for comp in comparisons if comp.get('significant', False)]
        if not significant_comparisons:
            return 0  # 返回0表示没有显著的brackets
        
        # 条件名称到索引的映射
        cond_to_idx = {cond: idx for idx, cond in enumerate(conditions)}
        
        # 计算bracket的高度层级
        bracket_heights = []
        bracket_spacing = max_height * float(SIG_BRACKET_SPACING)  # bracket之间的间距
        
        # 找到当前AOI中最高的bar（包含error bar）
        max_bar_height = 0
        for idx in range(len(conditions)):
            if idx < len(means_row) and sample_counts[aoi_idx, idx] > 0:
                bar_top = means_row[idx] + errors_row[idx]
                max_bar_height = max(max_bar_height, bar_top)
        
        # 设置第一个bracket的基础高度（可用 sig_line_height 控制）
        base_bracket_height = max_bar_height + max_height * float(sig_line_height)
        
        # 按条件对之间的距离排序（距离近的先绘制，在下层）
        significant_comparisons.sort(key=lambda x: abs(
            cond_to_idx.get(x['condition1'], 0) - cond_to_idx.get(x['condition2'], 0)
        ))
        
        for level, comp in enumerate(significant_comparisons):
            cond1, cond2 = comp['condition1'], comp['condition2']
            # p used for star decision depends on pairwise_p_mode
            p_for_sig = float(comp.get('p_value_for_sig', comp.get('p_value_corrected', comp.get('p_value', 1.0))))
            
            # 获取条件索引
            idx1 = cond_to_idx.get(cond1)
            idx2 = cond_to_idx.get(cond2)
            
            if idx1 is None or idx2 is None:
                continue
            
            # 计算bar的x坐标
            offset1 = width * (idx1 - (len(conditions) - 1) / 2) - float(BAR_OFFSET)
            offset2 = width * (idx2 - (len(conditions) - 1) / 2) - float(BAR_OFFSET)
            x1 = x_pos + offset1
            x2 = x_pos + offset2
            
            # 计算bracket高度
            bracket_height = base_bracket_height + level * bracket_spacing
            
            # 确保bracket不会与最高的bar重叠
            bar_height1 = means_row[idx1] + errors_row[idx1] if idx1 < len(means_row) else 0
            bar_height2 = means_row[idx2] + errors_row[idx2] if idx2 < len(means_row) else 0
            min_bracket_height = max(bar_height1, bar_height2) + max_height * float(sig_line_height)
            bracket_height = max(bracket_height, min_bracket_height)
            
            # 绘制bracket
            line_width = float(SIG_LINE_WIDTH)
            color = '#2C3E50'
            
            # 水平线
            ax.plot([x1, x2], [bracket_height, bracket_height], 
                   color=color, linewidth=line_width, solid_capstyle='round')
            
            # 左竖线
            tick_length = max_height * float(SIG_TICK_LENGTH)
            ax.plot([x1, x1], [bracket_height, bracket_height - tick_length], 
                   color=color, linewidth=line_width, solid_capstyle='round')
            
            # 右竖线
            ax.plot([x2, x2], [bracket_height, bracket_height - tick_length], 
                   color=color, linewidth=line_width, solid_capstyle='round')
            
            # 添加显著性标记
            if p_for_sig < 0.001:
                sig_text = "***"
            elif p_for_sig < 0.01:
                sig_text = "**"
            elif p_for_sig < 0.05:
                sig_text = "*"
            else:
                sig_text = "ns"
            
            # 在bracket中心添加显著性标记
            center_x = (x1 + x2) / 2
            ax.text(
                center_x,
                bracket_height + max_height * float(sig_star_offset),
                sig_text,
                ha='center',
                va='bottom',
                fontsize=STAR_FONT_SIZE,
                fontweight='bold',
                color=color,
            )
            
            bracket_heights.append(bracket_height + max_height * 0.04)
        
        # 返回最高的bracket高度
        return max(bracket_heights) if bracket_heights else 0

    # 先计算每个AOI的bracket最高点
    if show_pairwise and pairwise_results:
        for aoi_idx, comparisons in pairwise_results.items():
            if aoi_idx < len(aoi_labels):  # 确保索引有效
                # 获取该AOI的数据
                means_row = means[aoi_idx, :]
                errors_row = errors[aoi_idx, :]

                max_bracket_height = add_pairwise_brackets(
                    ax, aoi_idx, comparisons, means_row, errors_row, 
                    conditions, x[aoi_idx], width, max_height
                )
                aoi_max_bracket_heights[aoi_idx] = max_bracket_height

    # 在每个AOI左上角显示LMM结果（统一高度，但基于所有AOI的最高点）
    if show_stats and lmm_results:
        # 首先找到所有AOI中的最高点
        global_max_element = 0
        
        for i in range(len(aois)):
            # 计算该AOI的最高bar（包括error）
            aoi_max_bar = 0
            for j in range(len(conditions)):
                if i < len(means) and sample_counts[i, j] > 0:
                    bar_height = means[i, j] + errors[i, j]
                    aoi_max_bar = max(aoi_max_bar, bar_height)
            
            # 获取该AOI的bracket最高点
            aoi_bracket_height = aoi_max_bracket_heights.get(i, 0)
            
            # 找到该AOI的最高元素
            aoi_max = max(aoi_max_bar, aoi_bracket_height)
            global_max_element = max(global_max_element, aoi_max)
        
        # 统一的统计框y位置：基于全局最高元素（可用 STAT_BOX_Y_OFFSET 调整）
        unified_stat_box_y = global_max_element + max_height * float(STAT_BOX_Y_OFFSET)
        
        # 绘制所有统计框在同一高度
        for i, result_dict in lmm_results.items():
            stat_val = result_dict['statistic']
            p_val_raw = result_dict.get('p_value_raw', result_dict['p_value'])
            p_val_corrected = result_dict.get('p_value_corrected', result_dict['p_value'])
            test_type = result_dict.get('test_type', 'LRT')
            
            # 格式化统计量标签
            if test_type == 'LRT':
                stat_label = f"χ² = {stat_val:.2f}"
            else:
                stat_label = f"F = {stat_val:.2f}"
            
            # 格式化p值
            if p_val_corrected < 0.001:
                p_str = "p < 0.001"
            else:
                p_str = f"p = {p_val_corrected:.3f}"
            
            # 根据校正后p值的显著性决定颜色
            if p_val_corrected < 0.05:
                text_color = '#C85250'  # 柔和的深红色
                box_color = 'white'
                edge_color = '#E8A5A4'  # 浅红色边框
            else:
                text_color = '#2C3E50'
                box_color = 'white'
                edge_color = '#D5DBDB'
            
            # 显示统计结果（统一高度）
            ax.text(
                i - 0.40, unified_stat_box_y,
                f"{stat_label}\n{p_str}",
                fontsize=STAT_BOX_FONT_SIZE, ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.4', 
                         facecolor=box_color, 
                         edgecolor=edge_color, 
                         alpha=0.9, linewidth=0.8),
                zorder=10,  # 确保在最上层
                color=text_color,
                fontweight='medium'
            )

    # 更新图表的y轴范围
    if show_stats and lmm_results:
        # 统计框高度估算
        stat_box_height = max_height * 0.10
        # Y轴上限：统计框底部 + 统计框高度 + 小边距
        final_max_height = unified_stat_box_y + stat_box_height + max_height * 0.02
    elif aoi_max_bracket_heights:
        # 只有brackets，没有统计框
        final_max_height = max(aoi_max_bracket_heights.values()) * 1.05
    else:
        # 既没有brackets也没有统计框
        final_max_height = max_height * 1.1
    
    limits_map = y_axis_limits if y_axis_limits is not None else Y_AXIS_LIMITS
    if isinstance(limits_map, dict) and str(feature) in limits_map:
        lo, hi = limits_map[str(feature)]
        ax.set_ylim(lo, hi)
    else:
        ax.set_ylim(0, final_max_height)

    # 分隔线
    for i in range(1, len(aoi_labels)):
        ax.axvline(i - 0.5, color="#BDC3C7", 
                  linestyle="-", linewidth=0.8, 
                  alpha=0.3, zorder=1)

    # 轴、标题、网格
    ax.set_facecolor("#FEFEFE")
    ax.set_xlabel("Area of Interest", fontsize=AXES_LABEL_SIZE, 
                 labelpad=8, color="#2C3E50", fontweight='medium')
    ax.set_ylabel(feature.replace("_", " ").title(), 
                 fontsize=AXES_LABEL_SIZE, color="#2C3E50", fontweight='medium')
    
    # 如果有FDR校正，在标题中注明
    title_suffix = f" (FDR-corrected)" if show_stats and raw_p_values else ""
    ax.set_title(f"{feature.replace('_', ' ').title()}{title_suffix}",
                fontsize=AXES_TITLE_SIZE, fontweight="bold", 
                pad=TITLE_PAD, color="#1A1A1A")
    
    ax.set_xticks(x)
    ax.set_xticklabels(aoi_labels, fontsize=TICK_LABEL_SIZE, 
                      color="#34495E", fontweight='medium')
    ax.tick_params(axis="y", labelsize=TICK_LABEL_SIZE, colors="#34495E", width=1)
    ax.tick_params(axis="x", width=1)
    ax.grid(axis="y", color="#E0E5E8", alpha=0.5, 
           linewidth=0.7, zorder=0, linestyle='-')
    ax.set_axisbelow(True)
    ax.margins(x=0.02)

    # 边框样式
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color("#95A5A6")
    ax.spines["bottom"].set_color("#95A5A6")
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    # 图例
    legend = ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.15),
        ncol=len(conditions), frameon=True, fontsize=LEGEND_FONT_SIZE,
        columnspacing=1.5, handlelength=1.5, handletextpad=0.6,
        edgecolor='#D5DBDB', fancybox=True, 
        shadow=False, framealpha=0.95
    )
    legend.get_frame().set_facecolor('#FAFAFA')
    legend.get_frame().set_linewidth(0.8)
    for text in legend.get_texts():
        text.set_color("#2C3E50")
        text.set_fontweight('medium')

    plt.subplots_adjust(bottom=0.22, top=0.93, left=0.10, right=0.98)

    if auto_save:
        os.makedirs(save_dir, exist_ok=True)
        if save_name is None:
            save_stem = _sanitize_filename(str(feature))
        else:
            save_stem = _sanitize_filename(save_name)
        save_format = str(save_format).lower().lstrip(".")
        out_path = os.path.join(save_dir, f"{save_stem}.{save_format}")
        fig.savefig(out_path, dpi=save_dpi, bbox_inches="tight")
        if close_after_save:
            plt.close(fig)
    
    return fig, ax