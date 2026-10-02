"""Data loading, quality checks, filtering and statistics (all Pandas)."""

import pandas as pd

MAX_FILTER_CATEGORIES = 20
HIGH_MISSING_PCT = 20.0
AGG_FUNCTIONS = {"Mean": "mean", "Median": "median", "Sum": "sum"}


# ------------------------------------------------------------------ helpers
def pretty(col) -> str:
    """Display-friendly column name: Exam_Score -> Exam Score."""
    return str(col).replace("_", " ")


def format_number(value, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:,.{digits}f}"


def plural(n: int, word: str) -> str:
    return f"{n:,} {word}{'' if n == 1 else 's'}"


# ------------------------------------------------------------------ loading
def load_csv(source):
    """Read a CSV from a path or file-like object.

    Returns (DataFrame, None) on success or (None, error_message) on failure.
    """
    try:
        try:
            df = pd.read_csv(source)
        except UnicodeDecodeError:
            if hasattr(source, "seek"):
                source.seek(0)
            df = pd.read_csv(source, encoding="latin-1")
    except pd.errors.EmptyDataError:
        return None, "The file is empty. Please upload a CSV that contains a header row and data."
    except pd.errors.ParserError as exc:
        return None, f"The file could not be parsed as a CSV ({exc}). Check that it is a valid comma-separated file."
    except Exception as exc:  # noqa: BLE001 - show any read problem to the user
        return None, f"The file could not be read as a CSV: {exc}"

    df.columns = [str(c).strip() for c in df.columns]
    if df.shape[1] == 0 or df.shape[0] == 0:
        return None, "The CSV has no data rows. Please upload a file that contains at least one record."
    return df, None


# --------------------------------------------------------- column detection
def get_numeric_columns(df: pd.DataFrame) -> list:
    return [
        c
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
        and df[c].notna().any()
    ]


def get_categorical_columns(df: pd.DataFrame) -> list:
    numeric = set(get_numeric_columns(df))
    return [c for c in df.columns if c not in numeric]


def filterable_columns(df: pd.DataFrame, max_unique: int = MAX_FILTER_CATEGORIES) -> list:
    """Categorical columns with a sensible number of distinct values."""
    return [
        c
        for c in get_categorical_columns(df)
        if 2 <= df[c].nunique(dropna=True) <= max_unique
    ]


# ------------------------------------------------------------------ summary
def dataset_summary(df: pd.DataFrame) -> dict:
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "numeric": len(get_numeric_columns(df)),
        "categorical": len(get_categorical_columns(df)),
        "missing": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
    }


def summary_sentence(info: dict) -> str:
    return (
        f"Your dataset contains {plural(info['rows'], 'record')} across "
        f"{plural(info['columns'], 'column')}: {info['numeric']} numerical and "
        f"{info['categorical']} categorical."
    )


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isna().sum()
    return pd.DataFrame(
        {
            "Column": df.columns,
            "Type": [str(t) for t in df.dtypes],
            "Non-null": df.notna().sum().values,
            "Missing": missing.values,
            "Missing %": (missing / max(len(df), 1) * 100).round(1).values,
            "Unique values": df.nunique(dropna=True).values,
        }
    )


# ------------------------------------------------------------- data quality
def data_quality_report(df: pd.DataFrame) -> list:
    """Return a list of (level, message) where level is 'ok' or 'warn'."""
    if df.empty:
        return [("warn", "The dataset has no rows.")]

    report = []

    duplicates = int(df.duplicated().sum())
    if duplicates:
        report.append(("warn", f"{plural(duplicates, 'duplicate row')} detected ({duplicates / len(df):.1%} of rows)"))
    else:
        report.append(("ok", "No duplicate rows detected"))

    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        names = ", ".join(pretty(c) for c in empty_cols)
        report.append(("warn", f"Completely empty columns: {names}"))
    else:
        report.append(("ok", "No completely empty columns"))

    missing_pct = df.isna().mean() * 100
    partial = missing_pct[(missing_pct > 0) & (missing_pct < 100)]
    if partial.empty:
        report.append(("ok", "No missing values detected"))
    for col, pct in partial.items():
        if pct >= HIGH_MISSING_PCT:
            report.append(("warn", f"{pretty(col)} has a high share of missing values ({pct:.1f}%)"))
        else:
            report.append(("warn", f"{pretty(col)} contains {pct:.1f}% missing values"))

    constant_cols = [
        c for c in df.columns if c not in empty_cols and df[c].nunique(dropna=True) == 1
    ]
    if constant_cols:
        names = ", ".join(pretty(c) for c in constant_cols)
        report.append(("warn", f"Constant-value columns (no variation): {names}"))
    else:
        report.append(("ok", "No constant-value columns"))

    text_numeric = []
    for col in get_categorical_columns(df):
        if df[col].dtype != object:
            continue
        values = df[col].dropna()
        if values.empty:
            continue
        convertible = pd.to_numeric(values, errors="coerce").notna().mean()
        if convertible >= 0.9:
            text_numeric.append(col)
    if text_numeric:
        names = ", ".join(pretty(c) for c in text_numeric)
        report.append(("warn", f"Possible numeric columns stored as text: {names}"))
    else:
        report.append(("ok", "No numeric columns stored as text detected"))

    return report


# ---------------------------------------------------------------- filtering
def apply_filters(df: pd.DataFrame, selections: dict, search: str = "") -> pd.DataFrame:
    """Return a filtered copy; the original DataFrame is never modified."""
    out = df
    for col, values in selections.items():
        if values:
            out = out[out[col].isin(values)]
    if search and search.strip() and not out.empty:
        text = out.astype(str)
        mask = text.apply(
            lambda s: s.str.contains(search.strip(), case=False, regex=False, na=False)
        ).any(axis=1)
        out = out[mask]
    return out.copy()


# --------------------------------------------------------------- statistics
def column_stats(df: pd.DataFrame, col: str):
    series = df[col].dropna()
    if series.empty:
        return None
    return {
        "mean": series.mean(),
        "median": series.median(),
        "min": series.min(),
        "max": series.max(),
        "std": series.std(),
        "count": int(series.count()),
        "missing": int(df[col].isna().sum()),
    }


def descriptive_table(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    table = df[columns].describe().T.round(3)
    table = table.rename(
        columns={"count": "Count", "mean": "Mean", "std": "Std Dev", "min": "Min",
                 "25%": "25th percentile", "50%": "50th percentile (median)",
                 "75%": "75th percentile", "max": "Max"}
    )
    table.index.name = "Column"
    return table.reset_index()


def aggregate_by_category(df: pd.DataFrame, cat: str, num, agg: str) -> pd.Series:
    """Aggregate a numeric column (or count rows) per category, largest first."""
    grouped = df.groupby(cat, observed=True)
    if agg == "Count":
        result = grouped.size()
        result.name = "Count"
    else:
        result = grouped[num].agg(AGG_FUNCTIONS[agg])
    return result.sort_values(ascending=False)


# ------------------------------------------------------------- correlations
def pearson_correlation(df: pd.DataFrame, x: str, y: str):
    """Return (r, n_rows_used). r is None when it cannot be computed."""
    pair = df[[x, y]].dropna()
    n = len(pair)
    if n < 3 or pair[x].nunique() < 2 or pair[y].nunique() < 2:
        return None, n
    return float(pair[x].corr(pair[y])), n


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    corr = df.corr(numeric_only=True)
    return corr.dropna(how="all").dropna(axis=1, how="all")


def correlation_pairs(corr: pd.DataFrame) -> pd.DataFrame:
    cols = list(corr.columns)
    rows = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            r = corr.loc[a, b]
            if pd.notna(r):
                rows.append((a, b, float(r)))
    return pd.DataFrame(rows, columns=["Variable 1", "Variable 2", "Correlation"])


def correlation_phrase(r: float) -> str:
    """Short strength + direction phrase, e.g. 'moderately strong positive'."""
    size = abs(r)
    if size < 0.1:
        return "negligible"
    if size < 0.3:
        strength = "weak"
    elif size < 0.5:
        strength = "moderate"
    elif size < 0.7:
        strength = "moderately strong"
    else:
        strength = "strong"
    return f"{strength} {'positive' if r > 0 else 'negative'}"


def describe_correlation(r: float) -> str:
    phrase = correlation_phrase(r)
    if phrase == "negligible":
        return "The selected variables show almost no linear relationship."
    return f"The selected variables show a {phrase} relationship."
