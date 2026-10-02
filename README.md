# DataLens - Interactive CSV Data Analysis Dashboard

> Turn raw CSV data into clear, actionable insights.

## Project Overview

DataLens is a lightweight analytics web app built with Python and Streamlit. Upload any CSV file (or use the built-in sample) and DataLens walks you through understanding the dataset, checking its quality, exploring statistics, building charts, analyzing correlations, and reading automatically generated insights. No API keys, database, or external services are needed.

## QSkill Internship Task

**Task 1:** *Using the Pandas library, load a CSV file and perform basic data analysis tasks, such as calculating the average of a selected column. Additionally, use Matplotlib to create visualizations, including bar charts, scatter plots, and heatmaps, to analyze the data. Provide insights and observations based on the analysis and visualizations.*

## QSkill Requirement Mapping

- [x] CSV loading using Pandas (`utils/analysis.py` -> `load_csv`)
- [x] Basic data analysis (Overview, Data Explorer, data-quality checks)
- [x] Average calculation (Statistics page: "Average [Column Name]" for any numerical column)
- [x] Matplotlib bar chart (Visualizations -> Bar chart)
- [x] Matplotlib scatter plot (Visualizations -> Scatter plot, with Pearson correlation)
- [x] Correlation heatmap (Correlations page, Seaborn on a Matplotlib figure)
- [x] Insights and observations (Insights page, generated from the actual data)

## Features

- **Welcome screen** with one-click sample dataset and CSV upload
- **Overview dashboard** with metric cards, preview and an auto-generated summary
- **Data Explorer** with column selection, types, unique counts, missing values and duplicates
- **Data-quality checks:** missing values, duplicates, empty columns, constant columns, numbers stored as text, high missing percentages
- **Global filters** (categorical multiselect and text search) that drive every page, with original vs. filtered record counts
- **Statistics:** average, median, min, max, standard deviation and a full descriptive table
- **Bar chart builder:** choose category, numeric column and aggregation (mean/median/sum/count)
- **Scatter plot builder:** optional grouping, trend line and Pearson correlation with a careful interpretation
- **Correlation Center:** annotated heatmap plus strongest positive and negative pairs
- **Insights engine:** deterministic Pandas logic (no AI API) covering statistics, relationships, group differences, data quality and potential outliers
- **Downloads:** filtered CSV, statistics CSV, correlation matrix CSV, chart PNGs, insights TXT

## Technologies Used

- Python
- Streamlit
- Pandas
- Matplotlib
- Seaborn

## Project Structure

```text
DataLens/
├── app.py                  # Streamlit UI: sidebar, pages, session state
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   └── config.toml         # Theme and upload settings
├── data/
│   └── sample_student_performance.csv   # Synthetic demo dataset
└── utils/
    ├── __init__.py
        ├── analysis.py         # Loading, quality checks, statistics, filtering, correlations
            ├── visualizations.py   # Matplotlib / Seaborn figures
                └── insights.py         # Rule-based insight generation
                ```

                ## Installation

                ```bash
                git clone https://github.com/<your-username>/DataLens.git
                cd DataLens
                python -m venv .venv
                # macOS / Linux
                source .venv/bin/activate
                # Windows
                .venv\Scripts\activate
                pip install -r requirements.txt
                ```

                ## Running Locally

                ```bash
                streamlit run app.py
                ```

                ## How to Use

                1. Launch DataLens.
                2. Upload a CSV or click **Use Sample Dataset**.
                3. Review the **Overview**.
                4. Explore the data in **Data Explorer** (and check data quality).
                5. Analyze **Statistics** for any numerical column.
                6. Create **Visualizations** (bar chart and scatter plot).
                7. Review **Correlations**.
                8. Read the generated **Insights**.
                9. Download results from the relevant pages.

                Use the sidebar filters at any time; every page then analyzes the filtered data while the original stays untouched.

                ## Deployment

                1. Push this repository to GitHub (`app.py` and `requirements.txt` at the repository root).
                2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
                3. Click **Create app**, pick the repository and branch, and set the main file path to `app.py`.
                4. Click **Deploy**. No secrets or environment variables are required.

                ## Data Analysis Methodology

                - **Data loading:** `pandas.read_csv`, with a Latin-1 fallback for non-UTF-8 files and clear errors for empty or malformed files.
                - **Data-quality checks:** duplicates via `DataFrame.duplicated`, missing values via `isna`, constant and empty columns via `nunique`, and numeric-looking text via `pd.to_numeric(errors="coerce")`. Data is never silently modified.
                - **Descriptive statistics:** mean, median, min, max, standard deviation and quartiles using Pandas; missing values are ignored per column.
                - **Aggregation:** `groupby` with mean, median, sum or count.
                - **Correlation analysis:** Pearson correlation via `DataFrame.corr(numeric_only=True)`.
                - **Visualization:** Matplotlib figures (Seaborn only styles the heatmap), with readable layouts and capped category counts.
                - **Insight generation:** rule-based thresholds (for example |r| >= 0.5 for a "relatively strong" relationship, 1.5 x IQR for potential outliers) written in cautious, non-causal language.

                ## Sample Dataset

                `data/sample_student_performance.csv` is a **synthetic demonstration dataset** of 100 students. It is not real data. A few values are intentionally missing so the data-quality checks have something to report.

                ## Limitations

                - Correlation does not imply causation.
                - Automatically generated insights are descriptive, not conclusions.
                - Results depend on the quality of the uploaded data.
                - Only Pearson (linear) correlation is used.
                - Very large files may be slow on free hosting.
                - DataLens is intended for exploratory analysis, not high-stakes decision making.

                ## Future Improvements

                - More visualization types (histograms, box plots, line charts)
                - Advanced anomaly detection
                - Machine-learning models
                - Database support
                - More advanced filtering (numeric ranges, dates)
                - PDF/HTML report generation