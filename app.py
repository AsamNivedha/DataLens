"""DataLens - interactive CSV data-analysis dashboard built with Streamlit."""

import io
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from utils import analysis as an
from utils import insights as ins
from utils import visualizations as viz

st.set_page_config(page_title="DataLens", page_icon="📊", layout="wide")

SAMPLE_PATH = Path(__file__).parent / "data" / "sample_student_performance.csv"
SAMPLE_NAME = "Sample dataset (student performance)"
PAGES = [
    "Overview",
    "Data Explorer",
    "Statistics",
    "Visualizations",
    "Correlations",
    "Insights",
]

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; max-width: 1200px;}
    div[data-testid="stMetric"] {
        background: #F7F9FC; border: 1px solid #E3E8EF;
        border-radius: 10px; padding: 14px 16px;
    }
    .dl-title {font-size: 2.8rem; font-weight: 700; margin-bottom: 0;}
    .dl-tagline {color: #5B6B7F; font-size: 1.2rem; margin-bottom: 1.5rem;}
    .dl-label {
        font-size: 0.72rem; letter-spacing: 0.08em; color: #7A8797;
        font-weight: 600; margin: 1.2rem 0 0.2rem 0;
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


def set_dataset(df: pd.DataFrame, name: str) -> None:
    """Store a new dataset and clear filter state left over from the old one."""
    st.session_state["df"] = df
    st.session_state["source_name"] = name
    st.session_state.pop("load_error", None)
    for key in list(st.session_state.keys()):
        if key.startswith("filter_") or key == "search_text":
            del st.session_state[key]


def use_sample() -> None:
    if not SAMPLE_PATH.exists():
        st.session_state["load_error"] = (
            "The sample dataset file (data/sample_student_performance.csv) "
            "was not found in the repository."
        )
        return
    df, error = load_csv_bytes(SAMPLE_PATH.read_bytes())
    if error:
        st.session_state["load_error"] = error
    else:
        set_dataset(df, SAMPLE_NAME)


def reset_filters() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith("filter_"):
            st.session_state[key] = []
    st.session_state["search_text"] = ""


def handle_upload(uploaded) -> None:
    """Load an uploaded file once per new upload (not on every rerun)."""
    upload_id = (uploaded.name, uploaded.size)
    if st.session_state.get("last_upload") == upload_id:
        return
    st.session_state["last_upload"] = upload_id
    df, error = load_csv_bytes(uploaded.getvalue())
    if error:
        st.session_state["load_error"] = error
    else:
        set_dataset(df, uploaded.name)


# -------------------------------------------------------------------- sidebar
def sidebar_label(text: str) -> None:
    st.sidebar.markdown(f'<p class="dl-label">{text}</p>', unsafe_allow_html=True)


def sidebar_filters(df: pd.DataFrame):
    selections = {}
    for col in an.filterable_columns(df)[:5]:
        options = sorted(df[col].dropna().unique(), key=str)
        selections[col] = st.sidebar.multiselect(
            an.pretty(col), options, key=f"filter_{col}"
        )
    search = st.sidebar.text_input("Search all columns", key="search_text")
    return selections, search


def sidebar_about() -> None:
    sidebar_label("ABOUT")
    st.sidebar.caption(
        "DataLens turns raw CSV files into statistics, charts, correlations "
        "and plain-language insights."
    )
    st.sidebar.caption("Built with Python, Streamlit, Pandas, Matplotlib and Seaborn.")
    st.sidebar.caption("QSkill Python Development Internship - Task 1")


# ------------------------------------------------------------------- helpers
def show_figure(fig, filename: str, key: str) -> None:
    st.pyplot(fig)
    st.download_button(
        "Download chart as PNG",
        viz.fig_to_png(fig),
        file_name=filename,
        mime="image/png",
        key=key,
    )
    plt.close(fig)


def csv_bytes(df: pd.DataFrame, index: bool = False) -> bytes:
    return df.to_csv(index=index).encode("utf-8")


# ------------------------------------------------------------------- welcome
def page_welcome() -> None:
    st.markdown('<p class="dl-title">DataLens</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="dl-tagline">Turn raw CSV data into clear, actionable insights.</p>',
        unsafe_allow_html=True,
    )
    features = [
        ("Upload your own CSV", "Bring any tabular dataset, no setup needed."),
        ("Explore the dataset", "Preview rows, column types and data quality."),
        ("Analyze statistics", "Averages, medians, spread and percentiles."),
        ("Visualize relationships", "Bar charts and scatter plots you configure."),
        ("Discover correlations", "A heatmap plus strongest positive/negative pairs."),
        ("Generate insights", "Plain-language findings computed from your data."),
    ]
    for start in (0, 3):
        cols = st.columns(3)
        for col, (title, text) in zip(cols, features[start:start + 3]):
            with col:
                st.markdown(f"**{title}**")
                st.caption(text)
    st.write("")
    st.button("Use Sample Dataset", type="primary", on_click=use_sample)
    st.caption(
        "The sample is a synthetic demonstration dataset of student performance. "
        "To analyze your own file, open the sidebar and choose Upload CSV."
    )


# ------------------------------------------------------------------ overview
def page_overview(df: pd.DataFrame, raw: pd.DataFrame) -> None:
    st.header("Overview")
    st.caption(f"Data source: {st.session_state.get('source_name', 'Unknown')}")
    if len(df) < len(raw):
        st.caption(f"Filters active: showing {len(df):,} of {len(raw):,} records.")

    info = an.dataset_summary(df)
    row1 = st.columns(3)
    row1[0].metric("Rows", f"{info['rows']:,}")
    row1[1].metric("Columns", f"{info['columns']:,}")
    row1[2].metric("Numerical Columns", info["numeric"])
    row2 = st.columns(3)
    row2[0].metric("Categorical Columns", info["categorical"])
    row2[1].metric("Missing Values", f"{info['missing']:,}")
    row2[2].metric("Duplicate Rows", f"{info['duplicates']:,}")

    st.info(an.summary_sentence(info))
    if info["rows"] < 10:
        st.warning("This is a very small dataset, so statistics may be unreliable.")

    st.subheader("Dataset preview")
    st.dataframe(df.head(10))
    st.caption(f"Dimensions: {info['rows']:,} rows x {info['columns']:,} columns")


# ------------------------------------------------------------- data explorer
def page_explorer(df: pd.DataFrame, raw: pd.DataFrame) -> None:
    st.header("Data Explorer")
    tab_data, tab_columns, tab_quality = st.tabs(["Data", "Columns", "Data quality"])

    with tab_data:
        c1, c2, c3 = st.columns(3)
        c1.metric("Original records", f"{len(raw):,}")
        c2.metric("Filtered records", f"{len(df):,}")
        c3.metric("Columns", df.shape[1])
        st.caption("Use the filters in the sidebar to narrow the data. The original dataset is never modified.")
        chosen = st.multiselect("Columns to display", list(df.columns), default=list(df.columns))
        if chosen:
            st.dataframe(df[chosen], height=400)
            st.download_button(
                "Download filtered dataset (CSV)",
                csv_bytes(df[chosen]),
                file_name="datalens_filtered_data.csv",
                mime="text/csv",
                key="dl_filtered",
            )
        else:
            st.info("Select at least one column to display.")

    with tab_columns:
        st.dataframe(an.column_summary(df))
        st.caption(f"Dimensions: {df.shape[0]:,} rows x {df.shape[1]:,} columns")
        dups = int(df.duplicated().sum())
        st.write(f"Duplicate rows: **{dups:,}**")
        if dups:
            with st.expander("Show duplicated rows"):
                st.dataframe(df[df.duplicated(keep=False)])

    with tab_quality:
        st.caption("Computed on the original (unfiltered) dataset. Nothing is changed or removed.")
        lines = [
            f"{'✓' if level == 'ok' else '⚠'} {text}"
            for level, text in an.data_quality_report(raw)
        ]
        st.markdown("\n\n".join(lines))


# ---------------------------------------------------------------- statistics
def page_statistics(df: pd.DataFrame) -> None:
    st.header("Statistics")
    numeric = an.get_numeric_columns(df)
    if not numeric:
        st.warning("No numerical columns were found, so statistics cannot be calculated.")
        return

    col = st.selectbox("Select a numerical column", numeric, format_func=an.pretty)
    stats = an.column_stats(df, col)
    if stats is None:
        st.warning(f"{an.pretty(col)} contains no values in the current selection.")
        return

    st.metric(f"Average {an.pretty(col)}", an.format_number(stats["mean"]))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Median", an.format_number(stats["median"]))
    c2.metric("Minimum", an.format_number(stats["min"]))
    c3.metric("Maximum", an.format_number(stats["max"]))
    c4.metric("Standard deviation", an.format_number(stats["std"]))
    st.caption(
        f"Based on {stats['count']:,} non-missing values; "
        f"{stats['missing']:,} missing values were ignored."
    )

    st.subheader("Descriptive statistics")
    table = an.descriptive_table(df, numeric)
    st.dataframe(table)
    st.download_button(
        "Download statistics summary (CSV)",
        csv_bytes(table),
        file_name="datalens_statistics_summary.csv",
        mime="text/csv",
        key="dl_stats",
    )


# ------------------------------------------------------------ visualizations
def tab_bar_chart(df: pd.DataFrame) -> None:
    cats = an.get_categorical_columns(df)
    nums = an.get_numeric_columns(df)
    if not cats:
        st.info("Bar charts need at least one categorical (text) column.")
        return

    c1, c2, c3 = st.columns(3)
    cat = c1.selectbox("Categorical column", cats, format_func=an.pretty)
    aggs = ["Mean", "Median", "Sum", "Count"] if nums else ["Count"]
    agg = c3.selectbox("Aggregation", aggs)
    num = None
    if agg == "Count":
        c2.caption("Count does not need a numerical column.")
    else:
        num = c2.selectbox("Numerical column", nums, format_func=an.pretty)

    values = an.aggregate_by_category(df, cat, num, agg).dropna()
    if values.empty:
        st.warning("There is no data to chart for this selection.")
        return

    n_categories = len(values)
    top_n = 12
    if n_categories > 5:
        top_n = st.slider("Maximum categories to display", 5, 30, 12)
    if n_categories > top_n:
        st.warning(
            f"{an.pretty(cat)} has {n_categories:,} categories. "
            f"Showing the top {top_n} so the chart stays readable."
        )
        values = values.head(top_n)

    title, y_label = viz.describe_bar(cat, num, agg)
    fig = viz.bar_chart(values, title, an.pretty(cat), y_label)
    show_figure(fig, "datalens_bar_chart.png", "dl_bar_png")
    with st.expander("View chart data"):
        st.dataframe(values.rename(y_label).reset_index())


def tab_scatter(df: pd.DataFrame) -> None:
    nums = an.get_numeric_columns(df)
    if len(nums) < 2:
        st.info("Scatter plots need at least two numerical columns.")
        return

    c1, c2, c3 = st.columns(3)
    x = c1.selectbox("X-axis", nums, index=0, format_func=an.pretty)
    y = c2.selectbox("Y-axis", nums, index=len(nums) - 1, format_func=an.pretty)
    groups = ["None"] + an.filterable_columns(df, max_unique=10)
    group = c3.selectbox(
        "Group by (optional)", groups, format_func=lambda c: c if c == "None" else an.pretty(c)
    )
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
    st.info(
        f"{an.describe_correlation(r)} "
        "Correlation alone does not show that one variable causes the other."
    )


def page_visualizations(df: pd.DataFrame) -> None:
    st.header("Visualizations")
    tab_bar, tab_scatter_plot = st.tabs(["Bar chart", "Scatter plot"])
    with tab_bar:
        tab_bar_chart(df)
    with tab_scatter_plot:
        tab_scatter(df)


# -------------------------------------------------------------- correlations
def page_correlations(df: pd.DataFrame) -> None:
    st.header("Correlations")
    st.caption(
        "Correlation measures the strength and direction of a linear relationship. "
        "It does not establish causation."
    )
    corr = an.correlation_matrix(df)
    if corr.shape[0] < 2:
        st.warning(
            "At least two numerical columns with valid values are needed to calculate correlations."
        )
        return

    fig = viz.heatmap(corr)
    show_figure(fig, "datalens_correlation_heatmap.png", "dl_heatmap_png")

    pairs = an.correlation_pairs(corr)
    c1, c2 = st.columns(2)
    if not pairs.empty:
        top = pairs.loc[pairs["Correlation"].idxmax()]
        bottom = pairs.loc[pairs["Correlation"].idxmin()]
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

    st.download_button(
        "Download correlation matrix (CSV)",
        csv_bytes(corr.round(4), index=True),
        file_name="datalens_correlation_matrix.csv",
        mime="text/csv",
        key="dl_corr",
    )


# ------------------------------------------------------------------ insights
def page_insights(df: pd.DataFrame) -> None:
    st.header("Insights")
    st.caption(
        "Generated automatically from the current (filtered) data using Pandas. "
        "These are descriptive observations, not proof of cause and effect."
    )
    numeric = an.get_numeric_columns(df)
    focus = None
    if numeric:
        focus = st.selectbox(
            "Focus variable (used for group comparisons and relationships)",
            numeric,
            index=len(numeric) - 1,
            format_func=an.pretty,
        )

    sections = ins.generate_insights(df, focus)
    for title, lines in sections.items():
        st.subheader(title)
        st.markdown("\n".join(f"- {line}" for line in lines))

    st.download_button(
        "Download insights summary (TXT)",
        ins.insights_to_text(sections, st.session_state.get("source_name", "dataset")).encode("utf-8"),
        file_name="datalens_insights.txt",
        mime="text/plain",
        key="dl_insights",
    )


# ---------------------------------------------------------------------- main
def main() -> None:
    sidebar_label("DATA SOURCE")
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        handle_upload(uploaded)
    st.sidebar.button("Use Sample Dataset", on_click=use_sample, key="sidebar_sample")

    raw = st.session_state.get("df")

    sidebar_label("NAVIGATION")
    page = st.sidebar.radio(
        "Navigation", PAGES, label_visibility="collapsed", disabled=raw is None
    )

    filtered = raw
    if raw is not None:
        sidebar_label("FILTERS")
        selections, search = sidebar_filters(raw)
        filtered = an.apply_filters(raw, selections, search)
        st.sidebar.caption(f"Original records: {len(raw):,}")
        st.sidebar.caption(f"Filtered records: {len(filtered):,}")
        st.sidebar.button("Reset filters", on_click=reset_filters)
    sidebar_about()

    if st.session_state.get("load_error"):
        st.error(st.session_state["load_error"])

    if raw is None:
        page_welcome()
        return
    if filtered.empty:
        st.warning("No records match the current filters. Adjust or reset the filters in the sidebar.")
        return

    if page == "Overview":
        page_overview(filtered, raw)
    elif page == "Data Explorer":
        page_explorer(filtered, raw)
    elif page == "Statistics":
        page_statistics(filtered)
    elif page == "Visualizations":
        page_visualizations(filtered)
    elif page == "Correlations":
        page_correlations(filtered)
    else:
        page_insights(filtered)


main()
