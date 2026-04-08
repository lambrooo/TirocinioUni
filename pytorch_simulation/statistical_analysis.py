"""
Statistical Analysis Module for Thesis Experiments
====================================================

Provides rigorous statistical analysis tools for comparing
Active Inference agents (Static vs Learning) and Q-Learning baseline.

Includes:
- T-tests (paired and independent)
- Confidence intervals
- Effect size (Cohen's d)
- ANOVA for multi-group comparisons
- Bootstrap confidence intervals
- LaTeX table generation for thesis
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import warnings


def compute_confidence_interval(
    data: np.ndarray, confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Compute confidence interval for a sample.

    Args:
        data: Sample data
        confidence: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)

    # t-distribution critical value
    t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)

    margin = t_crit * std_err
    return mean, mean - margin, mean + margin


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size.

    Interpretation:
    - |d| < 0.2: negligible
    - 0.2 <= |d| < 0.5: small
    - 0.5 <= |d| < 0.8: medium
    - |d| >= 0.8: large

    Args:
        group1: First group data
        group2: Second group data

    Returns:
        Cohen's d value
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def independent_ttest(
    group1: np.ndarray, group2: np.ndarray, alternative: str = "two-sided"
) -> Dict:
    """
    Perform independent samples t-test.

    Args:
        group1: First group data
        group2: Second group data
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        Dictionary with test results
    """
    # Check for equal variances (Levene's test)
    levene_stat, levene_p = stats.levene(group1, group2)
    equal_var = levene_p > 0.05

    # Perform t-test
    t_stat, p_value = stats.ttest_ind(
        group1, group2, equal_var=equal_var, alternative=alternative
    )

    # Effect size
    cohens_d = compute_cohens_d(group1, group2)

    # Confidence intervals
    mean1, ci1_low, ci1_high = compute_confidence_interval(group1)
    mean2, ci2_low, ci2_high = compute_confidence_interval(group2)

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "effect_size_interpretation": interpret_effect_size(cohens_d),
        "significant_at_05": p_value < 0.05,
        "significant_at_01": p_value < 0.01,
        "group1_mean": mean1,
        "group1_ci": (ci1_low, ci1_high),
        "group2_mean": mean2,
        "group2_ci": (ci2_low, ci2_high),
        "mean_difference": mean1 - mean2,
        "equal_variance_assumed": equal_var,
        "levene_p": levene_p,
    }


def paired_ttest(
    before: np.ndarray, after: np.ndarray, alternative: str = "two-sided"
) -> Dict:
    """
    Perform paired samples t-test.

    Useful for comparing first half vs second half of same simulation.

    Args:
        before: Pre-treatment data
        after: Post-treatment data
        alternative: 'two-sided', 'less', or 'greater'

    Returns:
        Dictionary with test results
    """
    if len(before) != len(after):
        raise ValueError("Paired samples must have equal length")

    differences = after - before

    t_stat, p_value = stats.ttest_rel(before, after, alternative=alternative)

    # Effect size for paired data
    cohens_d = (
        np.mean(differences) / np.std(differences, ddof=1)
        if np.std(differences) > 0
        else 0
    )

    mean_diff, ci_low, ci_high = compute_confidence_interval(differences)

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "effect_size_interpretation": interpret_effect_size(cohens_d),
        "significant_at_05": p_value < 0.05,
        "significant_at_01": p_value < 0.01,
        "mean_difference": mean_diff,
        "difference_ci": (ci_low, ci_high),
        "before_mean": np.mean(before),
        "after_mean": np.mean(after),
    }


def one_way_anova(groups: Dict[str, np.ndarray]) -> Dict:
    """
    Perform one-way ANOVA for comparing multiple groups.

    Args:
        groups: Dictionary mapping group names to data arrays

    Returns:
        Dictionary with ANOVA results and post-hoc tests
    """
    group_names = list(groups.keys())
    group_data = [groups[name] for name in group_names]

    # ANOVA
    f_stat, p_value = stats.f_oneway(*group_data)

    # Effect size (eta-squared)
    all_data = np.concatenate(group_data)
    grand_mean = np.mean(all_data)

    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in group_data)
    ss_total = np.sum((all_data - grand_mean) ** 2)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0

    result = {
        "f_statistic": f_stat,
        "p_value": p_value,
        "eta_squared": eta_squared,
        "significant_at_05": p_value < 0.05,
        "n_groups": len(groups),
        "group_means": {name: np.mean(data) for name, data in groups.items()},
        "group_stds": {name: np.std(data, ddof=1) for name, data in groups.items()},
    }

    # Post-hoc pairwise comparisons (Tukey HSD would be better, using Bonferroni correction)
    if p_value < 0.05:
        n_comparisons = len(group_names) * (len(group_names) - 1) // 2
        bonferroni_alpha = 0.05 / n_comparisons

        pairwise = {}
        for i, name1 in enumerate(group_names):
            for name2 in group_names[i + 1 :]:
                t_result = independent_ttest(groups[name1], groups[name2])
                pairwise[f"{name1} vs {name2}"] = {
                    "p_value": t_result["p_value"],
                    "cohens_d": t_result["cohens_d"],
                    "significant_bonferroni": t_result["p_value"] < bonferroni_alpha,
                }

        result["pairwise_comparisons"] = pairwise
        result["bonferroni_alpha"] = bonferroni_alpha

    return result


def bootstrap_ci(
    data: np.ndarray,
    statistic: callable = np.mean,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.

    Non-parametric approach that doesn't assume normal distribution.

    Args:
        data: Sample data
        statistic: Function to compute statistic (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level

    Returns:
        Tuple of (observed_statistic, lower_bound, upper_bound)
    """
    observed = statistic(data)

    bootstrap_stats = []
    n = len(data)

    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(sample))

    alpha = (1 - confidence) / 2
    lower = np.percentile(bootstrap_stats, alpha * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha) * 100)

    return observed, lower, upper


def compare_survival_rates(
    group1_survived: int, group1_total: int, group2_survived: int, group2_total: int
) -> Dict:
    """
    Compare survival rates (e.g., budget > 0 at end of simulation).

    Uses Fisher's exact test for small samples.

    Args:
        group1_survived: Number survived in group 1
        group1_total: Total in group 1
        group2_survived: Number survived in group 2
        group2_total: Total in group 2

    Returns:
        Dictionary with test results
    """
    # Contingency table
    table = [
        [group1_survived, group1_total - group1_survived],
        [group2_survived, group2_total - group2_survived],
    ]

    # Fisher's exact test
    odds_ratio, p_value = stats.fisher_exact(table)

    # Survival rates
    rate1 = group1_survived / group1_total if group1_total > 0 else 0
    rate2 = group2_survived / group2_total if group2_total > 0 else 0

    return {
        "p_value": p_value,
        "odds_ratio": odds_ratio,
        "group1_rate": rate1,
        "group2_rate": rate2,
        "rate_difference": rate1 - rate2,
        "significant_at_05": p_value < 0.05,
    }


def analyze_experiment_results(
    results_df: pd.DataFrame,
    group_column: str = "config_label",
    metric_columns: List[str] = None,
) -> Dict:
    """
    Comprehensive statistical analysis of experiment results.

    Args:
        results_df: DataFrame with experiment results
        group_column: Column name for grouping
        metric_columns: Columns to analyze (default: auto-detect numeric)

    Returns:
        Dictionary with all statistical results
    """
    if metric_columns is None:
        metric_columns = results_df.select_dtypes(include=[np.number]).columns.tolist()
        metric_columns = [c for c in metric_columns if c not in ["run_id", "step"]]

    groups = results_df[group_column].unique()

    analysis = {
        "summary_statistics": {},
        "pairwise_comparisons": {},
        "anova_results": {},
    }

    # Summary statistics
    for metric in metric_columns:
        analysis["summary_statistics"][metric] = {}
        for group in groups:
            data = results_df[results_df[group_column] == group][metric].dropna().values
            if len(data) > 0:
                mean, ci_low, ci_high = compute_confidence_interval(data)
                analysis["summary_statistics"][metric][group] = {
                    "n": len(data),
                    "mean": mean,
                    "std": np.std(data, ddof=1),
                    "ci_95": (ci_low, ci_high),
                    "median": np.median(data),
                    "min": np.min(data),
                    "max": np.max(data),
                }

    # ANOVA for each metric (if more than 2 groups)
    if len(groups) >= 2:
        for metric in metric_columns:
            group_data = {}
            for group in groups:
                data = (
                    results_df[results_df[group_column] == group][metric]
                    .dropna()
                    .values
                )
                if len(data) >= 2:  # Need at least 2 samples
                    group_data[group] = data

            if len(group_data) >= 2:
                if len(group_data) == 2:
                    # Use t-test for 2 groups
                    g1, g2 = list(group_data.keys())
                    analysis["pairwise_comparisons"][metric] = {
                        f"{g1} vs {g2}": independent_ttest(
                            group_data[g1], group_data[g2]
                        )
                    }
                else:
                    # Use ANOVA for 3+ groups
                    analysis["anova_results"][metric] = one_way_anova(group_data)

    return analysis


def generate_latex_stats_table(
    analysis: Dict, metric: str, caption: str = None, label: str = None
) -> str:
    """
    Generate LaTeX table from statistical analysis.

    Args:
        analysis: Analysis dictionary from analyze_experiment_results()
        metric: Metric to generate table for
        caption: Table caption
        label: LaTeX label

    Returns:
        LaTeX table string
    """
    stats = analysis["summary_statistics"].get(metric, {})

    if not stats:
        return "% No data available for this metric"

    groups = list(stats.keys())

    latex = "\\begin{table}[htbp]\n\\centering\n"

    if caption:
        latex += f"\\caption{{{caption}}}\n"
    if label:
        latex += f"\\label{{{label}}}\n"

    latex += "\\begin{tabular}{lcccc}\n\\toprule\n"
    latex += "Agent & Mean & Std & 95\\% CI & N \\\\\n\\midrule\n"

    for group in groups:
        s = stats[group]
        ci = s["ci_95"]
        latex += f"{group} & {s['mean']:.2f} & {s['std']:.2f} & [{ci[0]:.2f}, {ci[1]:.2f}] & {s['n']} \\\\\n"

    latex += "\\bottomrule\n\\end{tabular}\n"

    # Add pairwise comparison if available
    if metric in analysis["pairwise_comparisons"]:
        comparisons = analysis["pairwise_comparisons"][metric]
        latex += "\n\\vspace{0.3cm}\n"
        latex += "\\begin{tabular}{lcccc}\n\\toprule\n"
        latex += "Comparison & t-stat & p-value & Cohen's d & Sig. \\\\\n\\midrule\n"

        for comp_name, comp_stats in comparisons.items():
            sig = (
                "$^{**}$"
                if comp_stats["significant_at_01"]
                else ("$^{*}$" if comp_stats["significant_at_05"] else "")
            )
            latex += f"{comp_name} & {comp_stats['t_statistic']:.3f} & {comp_stats['p_value']:.4f} & {comp_stats['cohens_d']:.3f} & {sig} \\\\\n"

        latex += "\\bottomrule\n"
        latex += (
            "\\multicolumn{5}{l}{\\footnotesize $^{*}p<0.05$, $^{**}p<0.01$} \\\\\n"
        )
        latex += "\\end{tabular}\n"

    latex += "\\end{table}"

    return latex


def print_analysis_report(analysis: Dict, title: str = "Statistical Analysis Report"):
    """
    Print a formatted analysis report to console.

    Args:
        analysis: Analysis dictionary from analyze_experiment_results()
        title: Report title
    """
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    # Summary statistics
    print("\n1. SUMMARY STATISTICS")
    print("-" * 40)

    for metric, groups in analysis["summary_statistics"].items():
        print(f"\n{metric}:")
        for group, stats in groups.items():
            ci = stats["ci_95"]
            print(f"  {group}:")
            print(f"    Mean: {stats['mean']:.4f} (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")
            print(f"    Std:  {stats['std']:.4f}, N = {stats['n']}")

    # Pairwise comparisons
    if analysis["pairwise_comparisons"]:
        print("\n2. PAIRWISE COMPARISONS (t-tests)")
        print("-" * 40)

        for metric, comparisons in analysis["pairwise_comparisons"].items():
            print(f"\n{metric}:")
            for comp_name, stats in comparisons.items():
                sig = (
                    "**"
                    if stats["significant_at_01"]
                    else ("*" if stats["significant_at_05"] else "")
                )
                print(f"  {comp_name}:")
                print(
                    f"    t = {stats['t_statistic']:.3f}, p = {stats['p_value']:.4f} {sig}"
                )
                print(
                    f"    Cohen's d = {stats['cohens_d']:.3f} ({stats['effect_size_interpretation']})"
                )

    # ANOVA results
    if analysis["anova_results"]:
        print("\n3. ANOVA RESULTS")
        print("-" * 40)

        for metric, stats in analysis["anova_results"].items():
            print(f"\n{metric}:")
            sig = (
                "**"
                if stats["p_value"] < 0.01
                else ("*" if stats["p_value"] < 0.05 else "")
            )
            print(
                f"  F({stats['n_groups'] - 1}, N-k) = {stats['f_statistic']:.3f}, p = {stats['p_value']:.4f} {sig}"
            )
            print(f"  eta^2 = {stats['eta_squared']:.4f}")

            if "pairwise_comparisons" in stats:
                print("  Post-hoc (Bonferroni-corrected):")
                for comp_name, comp in stats["pairwise_comparisons"].items():
                    sig = "*" if comp["significant_bonferroni"] else ""
                    print(
                        f"    {comp_name}: p = {comp['p_value']:.4f}, d = {comp['cohens_d']:.3f} {sig}"
                    )

    print("\n" + "=" * 60)
    print("* p < 0.05, ** p < 0.01")
    print("=" * 60)


def run_thesis_statistical_analysis(results_df: pd.DataFrame, output_dir: str = "."):
    """
    Run full statistical analysis for thesis and save outputs.

    Args:
        results_df: DataFrame with all experiment results
        output_dir: Directory to save outputs

    Returns:
        Dictionary with all analysis results
    """
    import os

    os.makedirs(output_dir, exist_ok=True)

    # Perform analysis
    analysis = analyze_experiment_results(
        results_df,
        group_column="config_label",
        metric_columns=["final_budget", "avg_efficiency", "overheating_steps"],
    )

    # Print report
    print_analysis_report(analysis)

    # Generate LaTeX tables
    metrics_info = [
        ("final_budget", "Final Budget Comparison", "tab:budget"),
        ("avg_efficiency", "System Efficiency Comparison", "tab:efficiency"),
    ]

    for metric, caption, label in metrics_info:
        latex = generate_latex_stats_table(analysis, metric, caption, label)
        filepath = os.path.join(output_dir, f"stats_{metric}.tex")
        with open(filepath, "w") as f:
            f.write(latex)
        print(f"Saved LaTeX table: {filepath}")

    # Save analysis as JSON
    import json

    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_for_json(x) for x in obj]
        return obj

    json_path = os.path.join(output_dir, "statistical_analysis.json")
    with open(json_path, "w") as f:
        json.dump(convert_for_json(analysis), f, indent=2)
    print(f"Saved analysis JSON: {json_path}")

    return analysis


if __name__ == "__main__":
    # Example usage of the statistical functions
    print("Testing statistical analysis module...")

    # Create sample data
    np.random.seed(42)

    static_budget = np.random.normal(5000, 1000, 20)
    learning_budget = np.random.normal(7000, 1200, 20)
    qlearning_budget = np.random.normal(4500, 1500, 20)

    # T-test
    print("\n1. Independent T-Test (Static vs Learning):")
    result = independent_ttest(static_budget, learning_budget)
    print(f"   t = {result['t_statistic']:.3f}, p = {result['p_value']:.4f}")
    print(
        f"   Cohen's d = {result['cohens_d']:.3f} ({result['effect_size_interpretation']})"
    )
    print(f"   Significant: {result['significant_at_05']}")

    # ANOVA
    print("\n2. One-way ANOVA:")
    groups = {
        "Static": static_budget,
        "Learning": learning_budget,
        "Q-Learning": qlearning_budget,
    }
    anova_result = one_way_anova(groups)
    print(
        f"   F = {anova_result['f_statistic']:.3f}, p = {anova_result['p_value']:.4f}"
    )
    print(f"   eta^2 = {anova_result['eta_squared']:.4f}")

    # Bootstrap CI
    print("\n3. Bootstrap CI for Learning budget:")
    obs, low, high = bootstrap_ci(learning_budget, n_bootstrap=5000)
    print(f"   Mean: {obs:.2f}, 95% CI: [{low:.2f}, {high:.2f}]")

    # Create test DataFrame
    print("\n4. Full Analysis Report:")
    data = []
    for label, budget in [
        ("Static", static_budget),
        ("Learning", learning_budget),
        ("Q-Learning", qlearning_budget),
    ]:
        for i, b in enumerate(budget):
            data.append(
                {
                    "config_label": label,
                    "run_id": i,
                    "final_budget": b,
                    "avg_efficiency": np.random.uniform(0.5, 1.0),
                }
            )

    df = pd.DataFrame(data)
    analysis = analyze_experiment_results(df)
    print_analysis_report(analysis)

    # Generate LaTeX
    print("\n5. LaTeX Table:")
    latex = generate_latex_stats_table(
        analysis, "final_budget", "Budget Comparison", "tab:budget"
    )
    print(latex)

    print("\nStatistical analysis module test completed.")
