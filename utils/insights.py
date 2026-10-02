"""Deterministic, rule-based insight generation (no AI API required)."""

import pandas as pd

from utils import analysis as an
from utils.analysis import pretty

STRONG_R = 0.5
MAX_STAT_COLUMNS = 6


def generate_insights(df: pd.DataFrame, focus=None) -> dict:
    """Return {section title: [insight sentences]} computed from the data."""
    numeric = an.get_numeric_columns(df)
    sections = {}

    sections["Dataset overview"] = [an.summary_sentence(an.dataset_summary(df))]
    sections["Statistics"] = _statistics_insights(df, numeric)
    sections["Relationships"] = _relationship_insights(df, numeric, focus)
    sections["Categories"] = _category_insights(df, focus)
    sections["Data quality"] = _quality_insights(df)
    sections["Possible outliers"] = _outlier_insights(df, numeric)
    return sections


def _statistics_insights(df, numeric):
    if not numeric:
        return ["No numerical columns are available for statistics."]
    lines = []
    for col in numeric[:MAX_STAT_COLUMNS]:
        stats = an.column_stats(df, col)
        if stats is None:
            continue
        line = (
            f"The average {pretty(col)} is {stats['mean']:,.2f} "
            f"(median {stats['median']:,.2f}, range {stats['min']:,.2f} to {stats['max']:,.2f})."
        )
        if stats["std"] and abs(stats["mean"] - stats["median"]) / stats["std"] > 0.3:
            line += " The gap between mean and median suggests a skewed distribution."
        lines.append(line)
    if len(numeric) > MAX_STAT_COLUMNS:
        lines.append(f"Showing the first {MAX_STAT_COLUMNS} of {len(numeric)} numerical columns.")
    return lines


def _relationship_insights(df, numeric, focus):
    if len(numeric) < 2:
        return ["At least two numerical columns are needed to describe relationships."]
    corr = an.correlation_matrix(df)
    pairs = an.correlation_pairs(corr)
    if pairs.empty:
        return ["Correlations could not be calculated for this data."]

    pairs = pairs.reindex(pairs["Correlation"].abs().sort_values(ascending=False).index)
    lines = []
    strong = pairs[pairs["Correlation"].abs() >= STRONG_R].head(3)
    if strong.empty:
        top = pairs.iloc[0]
        lines.append(
            f"No pair of numerical variables reaches |r| >= {STRONG_R}. The strongest is "
            f"{pretty(top['Variable 1'])} and {pretty(top['Variable 2'])} "
            f"(r = {top['Correlation']:.2f})."
        )
    for _, row in strong.iterrows():
        lines.append(
            f"{pretty(row['Variable 1'])} and {pretty(row['Variable 2'])} show a "
            f"{an.correlation_phrase(row['Correlation'])} relationship (r = {row['Correlation']:.2f})."
        )

    weakest = pairs.iloc[-1]
    if abs(weakest["Correlation"]) < 0.2 and len(pairs) > len(strong):
        lines.append(
            f"{pretty(weakest['Variable 1'])} and {pretty(weakest['Variable 2'])} show little "
            f"linear relationship (r = {weakest['Correlation']:.2f})."
        )

    if focus in corr.columns:
        related = corr[focus].drop(focus).dropna()
        related = related.reindex(related.abs().sort_values(ascending=False).index).head(2)
        if not related.empty:
            parts = [f"{pretty(c)} (r = {r:.2f})" for c, r in related.items()]
            lines.append(
                f"{pretty(focus)} is most closely associated with " + " and ".join(parts) + "."
            )
    lines.append("Correlation describes association only and does not show cause and effect.")
    return lines


def _category_insights(df, focus):
    if focus is None:
        return ["No numerical column is available to compare across groups."]
    lines = []
    for cat in an.filterable_columns(df, max_unique=10)[:3]:
        groups = df.groupby(cat, observed=True)[focus].agg(["mean", "count"]).dropna()
        if len(groups) < 2:
            continue
        high, low = groups["mean"].idxmax(), groups["mean"].idxmin()
        high_mean, low_mean = groups.loc[high, "mean"], groups.loc[low, "mean"]
        line = (
            f"Among {pretty(cat)} groups, {high} has the highest average {pretty(focus)} "
            f"({high_mean:,.2f}) and {low} has the lowest ({low_mean:,.2f}), "
            f"a difference of {high_mean - low_mean:,.2f}."
        )
        if groups["count"].min() < 5:
            line += " Some groups are small, so treat this difference with caution."
        lines.append(line)
    return lines or ["No categorical column with 2 to 10 groups is available for comparison."]


def _quality_insights(df):
    concerns = [text for level, text in an.data_quality_report(df) if level == "warn"]
    return [f"{c}." for c in concerns] or ["No data-quality concerns were detected."]


def _outlier_insights(df, numeric):
    findings = []
    for col in numeric:
        values = df[col].dropna()
        if len(values) < 8:
            continue
        q1, q3 = values.quantile(0.25), values.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        count = int(((values < q1 - 1.5 * iqr) | (values > q3 + 1.5 * iqr)).sum())
        if count:
            findings.append((col, count, count / len(values) * 100))
    if not findings:
        return ["No potential outliers were detected using the 1.5 x IQR rule."]
    findings.sort(key=lambda item: item[1], reverse=True)
    return [
        f"{pretty(col)} has {an.plural(count, 'potential outlier')} ({pct:.1f}% of values) "
        "under the 1.5 x IQR rule. These may be unusual but valid values."
        for col, count, pct in findings[:3]
    ]


def insights_to_text(sections: dict, source_name: str = "dataset") -> str:
    lines = ["DataLens - Insights Summary", f"Source: {source_name}", ""]
    for title, items in sections.items():
        lines.append(title.upper())
        lines.extend(f"- {item}" for item in items)
        lines.append("")
    lines.append("These insights are descriptive and do not establish causation.")
    return "\n".join(lines)
