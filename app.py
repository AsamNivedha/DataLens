"""DataLens - turn your data into a story."""

import html
import io
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from utils import analysis as an
from utils import insights as ins
from utils import visualizations as viz

st.set_page_config(page_title="DataLens", layout="wide")

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_student_performance.csv"
SAMPLE_NAME = "student_performance.csv"
STEPS = ["Overview", "Explore", "Analyze", "Discover", "Export"]

st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 2rem; max-width: 1100px;}
    h1 {font-weight: 700; letter-spacing: -0.02em;}
    .dl-brand {
        font-size: 0.8rem; font-weight: 700; letter-spacing: 0.14em;
        text-transform: uppercase; color: #2F5D9E; margin: 0 0 0.3rem 0;
    }
    .dl-lead {font-size: 1.15rem; color: #4A5868; max-width: 40rem; margin-bottom: 1.5rem;}
    .dl-banner {padding-bottom: 0.6rem; margin-bottom: 0.8rem;}
    .dl-file {font-size: 1.5rem; font-weight: 650; color: #1F2933; line-height: 1.2;}
    .dl-meta {color: #6B7A8C; font-size: 0.95rem; margin-top: 0.15rem;}
    .dl-step {font-weight: 650; margin-bottom: 0.1rem;}
    .dl-label {
        font-size: 0.72rem; letter-spacing: 0.08em; color: #7A8797;
        font-weight: 600; margin: 1.2rem 0 0.2rem 0;
    }
    div[role="radiogroup"] {gap: 1.4rem; border-bottom: 1px solid #E3E8EF; padding-bottom: 0.6rem;}
    div[data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #E3E8EF;
        border-radius: 10px; padding: 14px 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- data loading
@st.cache_data(show_spinner=False)
def load_csv_bytes(raw: bytes):
    """Cached CSV parsing. Returns (DataFrame or None, error message or None)."""
    return an.load_csv(io.BytesIO(raw))


def set_dataset(df: pd.DataFrame, name: str, sample: bool = False) -> None:
    st.session_state["df"] = df
    st.session_state["source_name"] = name
    st.session_state["is_sample"] = sample
    st.session_state["step"] = STEPS[0]
    for key in list(st.session_state.keys()):
        if key.startswith("filter_") or key == "search_text":
            del st.session_state[key]


def use_sample() -> None:
    if not SAMPLE_PATH.exists():
        st.session_state["load_error"] = (
            "The sample file (data/sample_student_performance.csv) was not found in the repository."
        )
        return
    df, error = load_csv_bytes(SAMPLE_PATH.read_bytes())
    if error:
        st.session_state["load_error"] = error
    else:
        set_dataset(df, SAMPLE_NAME, sample=True)


def handle_upload(uploaded) -> None:
    upload_id = getattr(uploaded, "file_id", None) or (uploaded.name, uploaded.size)
    if st.session_state.get("last_upload") == upload_id:
        return
    st.session_state["last_upload"] = upload_id
    df, error = load_csv_bytes(uploaded.getvalue())
    if error:
        st.session_state["load_error"] = error
        return
    set_dataset(df, uploaded.name)
    st.rerun()


def reset_filters() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith("filter_"):
            st.session_state[key] = []
    st.session_state["search_text"] = ""


def go(step: str) -> None:
    st.session_state["step"] = step


# --------------------------------------------------------------------- helpers
def csv_bytes(df: pd.DataFrame, index: bool = False) -> bytes:
    return df.to_csv(index=index).encode("utf-8")


def show_figure(fig, filename: str, key: str) -> None:
    st.pyplot(fig)
    st.download_button("Download chart (PNG)", viz.fig_to_png(fig),
                       file_name=filename, mime="image/png", key=key)
    plt.close(fig)


def sidebar_label(text: str) -> None:
    st.sidebar.markdown(f'<p class="dl-label">{text}</p>', unsafe_allow_html=True)


def next_step(target: str, text: str, key: str) -> None:
    st.divider()
    st.button(f"{text} →", on_click=go, args=(target,), key=f"next_{key}")


def is_placeholder(line: str) -> bool:
    return line.startswith(("At least", "No categorical", "No numerical", "Correlations could not"))


def strongest_pair(df: pd.DataFrame):
    pairs = an.correlation_pairs(an.correlation_matrix(df))
    if pairs.empty:
        return None
    row = pairs.loc[pairs["Correlation"].abs().idxmax()]
    return row["Variable 1"], row["Variable 2"]


def dataset_header(raw: pd.DataFrame, df: pd.DataFrame) -> None:
    name = html.escape(st.session_state.get("source_name", "Dataset"))
    meta = f"{len(raw):,} rows · {raw.shape[1]:,} columns"
    if st.session_state.get("is_sample"):
        meta = "Sample data · " + meta
    if len(df) < len(raw):
        meta += f" · Showing {len(df):,} of {len(raw):,} rows after filters"
    st.markdown(
        f'<div class="dl-banner"><p class="dl-brand">DataLens</p>'
        f'<div class="dl-file">{name}</div><div class="dl-meta">{meta}</div></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------- landing
def page_landing() -> None:
    st.markdown('<p class="dl-brand">DataLens</p>', unsafe_allow_html=True)
    st.markdown("# Turn your data into a story.")
    st.markdown(
        '<p class="dl-lead">Upload a CSV to explore its structure, understand its quality, '
        "uncover relationships, and discover patterns that matter.</p>",
        unsafe_allow_html=True,
    )
    error = st.session_state.pop("load_error", None)
    if error:
        st.error(error)

    left, _ = st.columns([3, 2])
    with left:
        uploaded = st.file_uploader("Upload CSV", type=["csv"], key="uploader_main")
        if uploaded is not None:
            handle_upload(uploaded)
        st.caption("No file handy? Try a small example dataset.")
        st.button("Explore sample data", on_click=use_sample)

    st.write("")
    steps = [
        ("Understand", "See the shape of your data at a glance: size, column types and overall health."),
        ("Explore", "Browse records, search and filter, and see exactly what is missing."),
        ("Analyze", "Measure averages and spread, and see how variables move together."),
        ("Discover", "Read plain-language findings, then export what you need."),
    ]
    for col, (title, text) in zip(st.columns(4), steps):
        with col:
            st.markdown(f'<div class="dl-step">{title}</div>', unsafe_allow_html=True)
            st.caption(text)


# -------------------------------------------------------------------- overview
def step_overview(raw: pd.DataFrame, df: pd.DataFrame) -> None:
    info = an.dataset_summary(df)
    cols = st.columns(4)
    cols[0].metric("Rows", f"{info['rows']:,}")
    cols[1].metric("Columns", f"{info['columns']:,}")
    cols[2].metric("Numerical columns", info["numeric"])
    cols[3].metric("Categorical columns", info["categorical"])
    st.write(an.summary_sentence(info))
    if info["rows"] < 10:
        st.warning("This is a very small dataset, so statistics may be unreliable.")

    st.subheader("Data health")
    report = an.data_quality_report(raw)
    warnings = [text for level, text in report if level == "warn"]
    if not warnings:
        st.success(f"All {len(report)} health checks passed.")
    else:
        st.warning(f"{len(warnings)} of {len(report)} health checks need attention.")
        st.markdown("\n".join(f"- {w}" for w in warnings[:3]))
        if len(warnings) > 3:
            st.caption(f"{len(warnings) - 3} more under Explore → Data quality.")

    numeric = an.get_numeric_columns(df)
    if numeric:
        st.subheader("Key numbers")
        shown = numeric[:4]
        for col, name in zip(st.columns(len(shown)), shown):
            stats = an.column_stats(df, name)
            col.metric(f"Average {an.pretty(name)}", an.format_number(stats["mean"]) if stats else "n/a")

    st.subheader("What stands out")
    sections = ins.generate_insights(df, numeric[-1] if numeric else None)
    picks = [line for key in ("Relationships", "Categories")
             for line in sections[key][:1] if not is_placeholder(line)]
    if picks:
        st.markdown("\n".join(f"- {line}" for line in picks))
    else:
        st.caption("Nothing notable yet. The Discover step has the full picture.")

    next_step("Explore", "Explore the records and columns", "overview")


# --------------------------------------------------------------------- explore
def tab_dataset(df: pd.DataFrame) -> None:
    st.caption("Use the sidebar to search and filter. Your original data is never modified.")
    with st.expander("Choose columns"):
        chosen = st.multiselect("Columns to display", list(df.columns), default=list(df.columns))
    if chosen:
        st.dataframe(df[chosen], height=400)
        st.caption(f"{len(df):,} records shown.")
    else:
        st.info("Select at least one column to display.")
    st.subheader("Columns")
    st.dataframe(an.column_summary(df))


def tab_quality(raw: pd.DataFrame) -> None:
    st.caption("Checks run on the full dataset. DataLens never changes or removes your data.")
    report = an.data_quality_report(raw)
    attention = [t for level, t in report if level == "warn"]
    healthy = [t for level, t in report if level == "ok"]

    left, right = st.columns(2)
    with left:
        st.markdown("**Needs attention**")
        st.markdown("\n\n".join(f"⚠ {t}" for t in attention) if attention else "Nothing to flag.")
    with right:
        st.markdown("**Looking good**")
        st.markdown("\n\n".join(f"✓ {t}" for t in healthy) if healthy else "No checks passed.")

    missing = raw.isna().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False).round(1)
    if not missing.empty:
        st.subheader("Missing values by column")
        fig = viz.bar_chart(missing.rename(index=an.pretty), "Missing Values by Column", "Column", "Missing (%)")
        show_figure(fig, "datalens_missing_values.png", "dl_missing_png")

    duplicates = int(raw.duplicated().sum())
    if duplicates:
        with st.expander(f"View {duplicates:,} duplicated rows"):
            st.dataframe(raw[raw.duplicated(keep=False)])


def step_explore(raw: pd.DataFrame, df: pd.DataFrame) -> None:
    tab_a, tab_b = st.tabs(["Dataset", "Data quality"])
    with tab_a:
        tab_dataset(df)
    with tab_b:
        tab_quality(raw)
    next_step("Analyze", "Analyze statistics and relationships", "explore")


# --------------------------------------------------------------------- analyze
def tab_statistics(df: pd.DataFrame) -> None:
    numeric = an.get_numeric_columns(df)
    if not numeric:
        st.warning("No numerical columns were found, so statistics cannot be calculated.")
        return
    col = st.selectbox("Column", numeric, format_func=an.pretty)
    stats = an.column_stats(df, col)
    if stats is None:
        st.warning(f"{an.pretty(col)} has no values in the current selection.")
        return

    st.metric(f"Average {an.pretty(col)}", an.format_number(stats["mean"]))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median", an.format_number(stats["median"]))
    c2.metric("Minimum", an.format_number(stats["min"]))
    c3.metric("Maximum", an.format_number(stats["max"]))
    c4.metric("Standard deviation", an.format_number(stats["std"]))
    st.caption(f"Based on {stats['count']:,} values; {stats['missing']:,} missing values were ignored.")

    st.subheader("All numerical columns")
    st.dataframe(an.descriptive_table(df, numeric))


def tab_correlations(df: pd.DataFrame) -> None:
    corr = an.correlation_matrix(df)
    if corr.shape[0] < 2:
        st.warning("At least two numerical columns with valid values are needed to calculate correlations.")
        return
    st.caption("Correlation measures the strength and direction of a linear relationship. "
               "It does not establish causation.")
    show_figure(viz.heatmap(corr), "datalens_correlation_heatmap.png", "dl_heat_png")

    pairs = an.correlation_pairs(corr)
    if pairs.empty:
        return
    top = pairs.loc[pairs["Correlation"].idxmax()]
    bottom = pairs.loc[pairs["Correlation"].idxmin()]
    c1, c2 = st.columns(2)
    with c1:
        if top["Correlation"] > 0:
            st.metric("Strongest positive correlation", f"{top['Correlation']:.2f}")
            st.caption(f"{an.pretty(top['Variable 1'])} and {an.pretty(top['Variable 2'])}")
        else:
            st.metric("Strongest positive correlation", "None")
    with c2:
        if bottom["Correlation"] < 0:
            st.metric("Strongest negative correlation", f"{bottom['Correlation']:.2f}")
            st.caption(f"{an.pretty(bottom['Variable 1'])} and {an.pretty(bottom['Variable 2'])}")
        else:
            st.metric("Strongest negative correlation", "None")
    with st.expander("All variable pairs, ranked by strength"):
        ranked = pairs.reindex(pairs["Correlation"].abs().sort_values(ascending=False).index)
        st.dataframe(ranked.round(3).reset_index(drop=True))


def tab_scatter(df: pd.DataFrame) -> None:
    nums = an.get_numeric_columns(df)
    if len(nums) < 2:
        st.info("Comparing two variables needs at least two numerical columns.")
        return
    pair = strongest_pair(df)
    x_default = nums.index(pair[0]) if pair and pair[0] in nums else 0
    y_default = nums.index(pair[1]) if pair and pair[1] in nums else len(nums) - 1
    if x_default == y_default:
        y_default = (x_default + 1) % len(nums)
    st.caption("The selection starts on the most strongly related pair of variables.")

    c1, c2, c3 = st.columns(3)
    x = c1.selectbox("X-axis", nums, index=x_default, format_func=an.pretty)
    y = c2.selectbox("Y-axis", nums, index=y_default, format_func=an.pretty)
    groups = ["None"] + an.filterable_columns(df, max_unique=10)
    group = c3.selectbox("Color by (optional)", groups,
                         format_func=lambda c: c if c == "None" else an.pretty(c))
    trend = st.checkbox("Show trend line", value=True)

    if x == y:
        st.warning("Choose two different columns for the X and Y axes.")
        return
    r, n = an.pearson_correlation(df, x, y)
    if n == 0:
        st.warning("These columns have no rows where both values are present.")
        return

    fig = viz.scatter_plot(df, x, y, None if group == "None" else group, trend)
    show_figure(fig, "datalens_scatter_plot.png", "dl_scatter_png")
    if r is None:
        st.info("A correlation could not be calculated (too few points or no variation).")
        return
    m1, m2 = st.columns(2)
    m1.metric("Pearson correlation (r)", f"{r:.3f}")
    m2.metric("Rows used", f"{n:,}")
    st.info(f"{an.describe_correlation(r)} Correlation alone does not show that one variable causes the other.")


def tab_groups(df: pd.DataFrame) -> None:
    cats = an.get_categorical_columns(df)
    nums = an.get_numeric_columns(df)
    if not cats:
        st.info("Group comparison needs at least one categorical (text) column.")
        return

    c1, c2, c3 = st.columns(3)
    cat = c1.selectbox("Group by", cats, format_func=an.pretty)
    aggs = ["Mean", "Median", "Sum", "Count"] if nums else ["Count"]
    agg = c3.selectbox("Measure", aggs)
    num = None
    if agg == "Count":
        c2.caption("Count does not need a numerical column.")
    else:
        num = c2.selectbox("Value", nums, format_func=an.pretty)

    values = an.aggregate_by_category(df, cat, num, agg).dropna()
    if values.empty:
        st.warning("There is no data to chart for this selection.")
        return

    total = len(values)
    top_n = 12
    if total > 5:
        top_n = st.slider("Maximum groups to display", 5, 30, 12)
    if total > top_n:
        st.warning(f"{an.pretty(cat)} has {total:,} groups. Showing the top {top_n} so the chart stays readable.")
        values = values.head(top_n)

    title, y_label = viz.describe_bar(cat, num, agg)
    fig = viz.bar_chart(values, title, an.pretty(cat), y_label)
    show_figure(fig, "datalens_bar_chart.png", "dl_bar_png")
    with st.expander("View chart data"):
        st.dataframe(values.rename(y_label).reset_index())


def step_analyze(raw: pd.DataFrame, df: pd.DataFrame) -> None:
    t1, t2, t3, t4 = st.tabs(["Statistics", "Correlations", "Compare variables", "Compare groups"])
    with t1:
        tab_statistics(df)
    with t2:
        tab_correlations(df)
    with t3:
        tab_scatter(df)
    with t4:
        tab_groups(df)
    next_step("Discover", "Read the insights", "analyze")


# -------------------------------------------------------------------- discover
STORY = [
    ("What stands out", ["Relationships", "Categories"]),
    ("The numbers", ["Dataset overview", "Statistics"]),
    ("Worth checking", ["Data quality", "Possible outliers"]),
]


def step_discover(raw: pd.DataFrame, df: pd.DataFrame) -> None:
    st.caption("Findings generated from your data. They describe patterns, not causes.")
    numeric = an.get_numeric_columns(df)
    focus = None
    if numeric:
        focus = st.selectbox("Focus on", numeric, index=len(numeric) - 1, format_func=an.pretty,
                             help="Group comparisons and relationships are described for this variable.")
    sections = ins.generate_insights(df, focus)
    for heading, keys in STORY:
        lines = [line for key in keys for line in sections.get(key, [])]
        if lines:
            st.subheader(heading)
            st.markdown("\n".join(f"- {line}" for line in lines))
    next_step("Export", "Export your results", "discover")


# ---------------------------------------------------------------------- export
def build_report(raw: pd.DataFrame, df: pd.DataFrame, sections: dict) -> str:
    name = st.session_state.get("source_name", "dataset")
    lines = [
        "DataLens analysis report",
        f"Dataset: {name}",
        f"Records analyzed: {len(df):,} of {len(raw):,}",
        f"Columns: {raw.shape[1]:,}",
        "",
        "DATA HEALTH",
    ]
    lines += [f"- {text}" for _, text in an.data_quality_report(raw)]
    numeric = an.get_numeric_columns(df)
    if numeric:
        lines += ["", "KEY STATISTICS", an.descriptive_table(df, numeric).to_string(index=False)]
    lines += ["", ins.insights_to_text(sections, name)]
    return "\n".join(lines)


def step_export(raw: pd.DataFrame, df: pd.DataFrame) -> None:
    numeric = an.get_numeric_columns(df)
    sections = ins.generate_insights(df, numeric[-1] if numeric else None)
    name = st.session_state.get("source_name", "dataset")

    st.subheader("Data")
    st.download_button("Download filtered dataset (CSV)", csv_bytes(df),
                       file_name="datalens_filtered_data.csv", mime="text/csv", key="dl_filtered")
    st.caption(f"{len(df):,} of {len(raw):,} records, with your current filters applied.")

    if numeric:
        st.subheader("Statistics and relationships")
        st.download_button("Download statistics summary (CSV)", csv_bytes(an.descriptive_table(df, numeric)),
                           file_name="datalens_statistics_summary.csv", mime="text/csv", key="dl_stats")
        corr = an.correlation_matrix(df)
        if corr.shape[0] >= 2:
            st.download_button("Download correlation matrix (CSV)", csv_bytes(corr.round(4), index=True),
                               file_name="datalens_correlation_matrix.csv", mime="text/csv", key="dl_corr")
            fig = viz.heatmap(corr)
            png = viz.fig_to_png(fig)
            plt.close(fig)
            st.download_button("Download correlation heatmap (PNG)", png,
                               file_name="datalens_correlation_heatmap.png", mime="image/png",
                               key="dl_heat_export")

    st.subheader("Findings")
    st.download_button("Download insights (TXT)", ins.insights_to_text(sections, name).encode("utf-8"),
                   
