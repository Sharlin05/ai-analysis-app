import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Dataset Analysis App", layout="wide")
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_theme(style="darkgrid")


def inject_css() -> None:
    st.markdown(
        """
        <style>
            :root {
                color-scheme: dark;
            }
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(109, 40, 217, 0.28), transparent 30%),
                    radial-gradient(circle at bottom right, rgba(34, 211, 238, 0.24), transparent 28%),
                    linear-gradient(135deg, #07111f 0%, #0f172a 55%, #111827 100%);
            }
            .block-container {
                padding-top: 1.8rem;
                padding-bottom: 2rem;
            }
            .sidebar .sidebar-content {
                background: linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(8, 15, 32, 0.96));
            }
            .hero-card, .glass-card {
                background: rgba(255, 255, 255, 0.08);
                backdrop-filter: blur(18px);
                border: 1px solid rgba(255, 255, 255, 0.14);
                border-radius: 24px;
                box-shadow: 0 20px 45px rgba(2, 6, 23, 0.28);
                padding: 1.2rem 1.3rem;
                margin-bottom: 1rem;
                position: relative;
                overflow: hidden;
            }
            .hero-card {
                padding: 1.25rem 1.4rem;
            }
            .hero-card::before, .hero-card::after {
                content: "";
                position: absolute;
                width: 180px;
                height: 180px;
                border-radius: 50%;
                filter: blur(30px);
                opacity: 0.45;
                animation: drift 8s ease-in-out infinite;
            }
            .hero-card::before {
                background: #6d28d9;
                top: -60px;
                right: -35px;
            }
            .hero-card::after {
                background: #22d3ee;
                bottom: -80px;
                left: -35px;
                animation-delay: 2s;
            }
            .pill {
                display: inline-flex;
                margin-bottom: 0.7rem;
                padding: 0.35rem 0.8rem;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.12);
                color: #fef3c7;
                border: 1px solid rgba(255, 255, 255, 0.2);
                font-size: 0.82rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.08em;
            }
            .hero-title {
                font-size: 2.1rem;
                font-weight: 800;
                line-height: 1.15;
                margin: 0;
                color: #f8fafc;
            }
            .hero-subtitle {
                color: #cbd5e1;
                font-size: 1rem;
                margin-top: 0.45rem;
                max-width: 760px;
            }
            .metric-card {
                border-radius: 20px;
                padding: 1rem;
                background: linear-gradient(145deg, rgba(255,255,255,0.12), rgba(255,255,255,0.05));
                box-shadow: 0 18px 40px rgba(2, 6, 23, 0.25);
                border: 1px solid rgba(255, 255, 255, 0.13);
                transform: perspective(900px) rotateX(4deg) rotateY(-3deg);
                transition: transform 0.25s ease, box-shadow 0.25s ease;
                margin-bottom: 0.7rem;
            }
            .metric-card:hover {
                transform: perspective(900px) rotateX(0deg) rotateY(0deg) translateY(-4px);
                box-shadow: 0 24px 50px rgba(2, 6, 23, 0.35);
            }
            .metric-label {
                color: #cbd5e1;
                font-size: 0.9rem;
                margin-bottom: 0.2rem;
            }
            .metric-value {
                font-size: 1.3rem;
                font-weight: 800;
                color: #ffffff;
            }
            .metric-subtext {
                color: #94a3b8;
                font-size: 0.84rem;
                margin-top: 0.2rem;
            }
            .section-title {
                font-size: 1.08rem;
                font-weight: 700;
                color: #e2e8f0;
                margin-bottom: 0.25rem;
            }
            .section-muted {
                color: #94a3b8;
                font-size: 0.95rem;
            }
            .floating {
                animation: float 3s ease-in-out infinite;
            }
            .stDataFrame, .stTable {
                border-radius: 16px;
                overflow: hidden;
            }
            .dashboard-card {
                background: rgba(15, 23, 42, 0.92);
                border: 1px solid rgba(148, 163, 184, 0.16);
                border-radius: 18px;
                padding: 1.2rem;
                margin-bottom: 1rem;
                box-shadow: 0 18px 50px rgba(15, 23, 42, 0.25);
            }
            .dashboard-title {
                color: #f8fafc;
                font-size: 1.2rem;
                font-weight: 700;
                margin-bottom: 0.35rem;
            }
            .dashboard-subtitle {
                color: #94a3b8;
                margin-bottom: 1rem;
            }
            .dashboard-panel {
                background: rgba(30, 41, 59, 0.9);
                border: 1px solid rgba(148, 163, 184, 0.1);
                border-radius: 16px;
                padding: 1rem;
            }
            .chart-option {
                background: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(148, 163, 184, 0.12);
                border-radius: 14px;
                padding: 0.9rem;
                margin-bottom: 0.75rem;
            }
            .chart-option h4 {
                margin: 0 0 0.35rem 0;
                color: #f8fafc;
                font-size: 1rem;
            }
            .chart-option p {
                margin: 0;
                color: #94a3b8;
                font-size: 0.9rem;
            }
            div[data-testid="stMetric"] {
                background: transparent;
            }
            @keyframes drift {
                0%, 100% { transform: translate3d(0, 0, 0) scale(1); }
                50% { transform: translate3d(8px, -12px, 0) scale(1.06); }
            }
            @keyframes float {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-6px); }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero_section() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="pill">💫 Modern data storytelling</div>
            <h1 class="hero-title">Understand Your Dataset With AI .... 👨‍💻👩‍💻</h1>
            <p class="hero-subtitle">Simplify data analysis with automated preprocessing suggestions, statistical summaries and interactive visualizations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_card(title: str, value: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card floating">
            <div class="metric-label">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtext">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_type_distribution_chart(df: pd.DataFrame):
    dtype_summary = df.dtypes.astype(str).value_counts().reset_index()
    dtype_summary.columns = ["dtype", "count"]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(dtype_summary["count"], labels=dtype_summary["dtype"], autopct="%1.1f%%", startangle=90)
    ax.set_title("Column Type Distribution")
    ax.axis("equal")
    return fig


def build_clustered_bar_chart(df: pd.DataFrame, col_a: str, col_b: str, top_n: int):
    top_a = df[col_a].value_counts(dropna=False).head(top_n).rename(col_a)
    top_b = df[col_b].value_counts(dropna=False).head(top_n).rename(col_b)
    combined = pd.concat([top_a, top_b], axis=1).fillna(0)
    combined.index = combined.index.astype(str)

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(combined))
    width = 0.35
    ax.bar(x - width / 2, combined[col_a], width, label=col_a)
    ax.bar(x + width / 2, combined[col_b], width, label=col_b)
    ax.set_xticks(x)
    ax.set_xticklabels(combined.index, rotation=30, ha="right")
    ax.set_title(f"Top {top_n} values: {col_a} vs {col_b}")
    ax.legend()
    return fig


def build_donut_chart(df: pd.DataFrame):
    dtype_summary = df.dtypes.astype(str).value_counts().reset_index()
    dtype_summary.columns = ["dtype", "count"]

    fig, ax = plt.subplots(figsize=(4, 4))
    wedges, texts, autotexts = ax.pie(
        dtype_summary["count"],
        labels=dtype_summary["dtype"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops={"width": 0.45, "edgecolor": "w"},
        textprops={"color": "#d38f58", "fontsize": 9},
    )
    ax.set_title("Column Type Distribution", color="#f8fafc", pad=14)
    ax.axis("equal")
    fig.patch.set_facecolor("#0f172a")
    return fig


def build_column_chart(df: pd.DataFrame, column: str, top_n: int):
    top_values = df[column].value_counts(dropna=False).head(top_n)
    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.bar(top_values.index.astype(str), top_values.values, color="#22d3ee")
    ax.set_title(f"Top {top_n} values for {column}", color="#f8fafc", pad=12)
    ax.set_ylabel("Count", color="#cbd5e1")
    ax.tick_params(axis="x", rotation=35, labelsize=9, colors="#cbd5e1")
    ax.tick_params(axis="y", labelcolor="#cbd5e1")
    fig.tight_layout()
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    return fig
    top_values = df[column].value_counts(dropna=False).head(top_n)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(top_values.index.astype(str), top_values.values, color="#2563eb")
    ax.set_title(f"Top {top_n} values for {column}")
    ax.set_ylabel("Count")
    ax.set_xticklabels(top_values.index.astype(str), rotation=35, ha="right")
    fig.tight_layout()
    return fig


def build_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str):
    fig, ax = plt.subplots(figsize=(5.25, 4))
    ax.scatter(df[x_col], df[y_col], alpha=0.6, edgecolors="w", s=50, color="#38bdf8")
    ax.set_title(f"Scatter plot: {x_col} vs {y_col}", color="#f8fafc", pad=12)
    ax.set_xlabel(x_col, color="#cbd5e1")
    ax.set_ylabel(y_col, color="#cbd5e1")
    ax.tick_params(axis="x", colors="#cbd5e1")
    ax.tick_params(axis="y", colors="#cbd5e1")
    ax.grid(alpha=0.2, color="#94a3b8")
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    return fig


def build_gauge_chart(df: pd.DataFrame):
    total_cells = df.size
    missing_cells = df.isna().sum().sum()
    missing_ratio = missing_cells / total_cells if total_cells else 0
    completeness = max(0.0, 1.0 - missing_ratio) * 100

    fig, ax = plt.subplots(figsize=(4.5, 2.25), subplot_kw={"aspect": "equal"})
    ax.add_patch(patches.Wedge((0, 0), 1, 180, 0, facecolor="#1e293b", edgecolor="none"))
    ax.add_patch(patches.Wedge((0, 0), 1, 180, 180 * completeness / 100, facecolor="#22c55e", edgecolor="none"))
    ax.text(0, -0.15, f"{completeness:.1f}% complete", ha="center", va="center", fontsize=14, fontweight="bold", color="#f8fafc")
    ax.text(0, -0.35, f"Missing: {missing_ratio * 100:.1f}%", ha="center", va="center", fontsize=10, color="#cbd5e1")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.5, 1.1)
    ax.axis("off")
    fig.patch.set_facecolor("#0f172a")
    return fig
    total_cells = df.size
    missing_cells = df.isna().sum().sum()
    missing_ratio = missing_cells / total_cells if total_cells else 0
    completeness = max(0.0, 1.0 - missing_ratio) * 100

    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw={"aspect": "equal"})
    ax.add_patch(patches.Wedge((0, 0), 1, 180, 0, facecolor="#e5e7eb", edgecolor="none"))
    ax.add_patch(patches.Wedge((0, 0), 1, 180, 180 * completeness / 100, facecolor="#22c55e", edgecolor="none"))
    ax.text(0, -0.2, f"{completeness:.1f}% complete", ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(0, -0.4, f"Missing: {missing_ratio * 100:.1f}%", ha="center", va="center", fontsize=12)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.5, 1.1)
    ax.axis("off")
    return fig


def build_correlation_heatmap(df: pd.DataFrame):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return None

    fig, ax = plt.subplots(figsize=(7, 5))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap")
    return fig


def recommend_ml_algorithms(df: pd.DataFrame):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    target_candidates = [c for c in df.columns if c.lower() in {"target", "label", "class", "y", "output", "result"}]
    target_col = target_candidates[0] if target_candidates else None

    if target_col is None:
        for col in df.columns:
            if col in cat_cols and df[col].nunique(dropna=False) <= 10:
                target_col = col
                break

    if target_col is not None:
        target_values = df[target_col].dropna()
        unique_count = target_values.nunique()
        if target_values.dtype == "object" or target_values.dtype == "category" or target_values.dtype == "bool" or unique_count <= 10:
            return {
                "problem_type": "Classification",
                "reason": f"Column '{target_col}' looks like a class label with {unique_count} distinct values.",
                "algorithms": [
                    "Logistic Regression",
                    "Decision Tree Classifier",
                    "Random Forest Classifier",
                    "XGBoost Classifier",
                ],
            }

        return {
            "problem_type": "Regression",
            "reason": f"Column '{target_col}' appears numeric, so it fits a regression-style prediction task.",
            "algorithms": [
                "Linear Regression",
                "Random Forest Regressor",
                "XGBoost Regressor",
                "Ridge or Lasso Regression",
            ],
        }

    if len(numeric_cols) >= 3:
        return {
            "problem_type": "Clustering / Pattern Detection",
            "reason": "The dataset has several numeric features and no obvious target column.",
            "algorithms": [
                "K-Means Clustering",
                "DBSCAN",
                "PCA for dimensionality reduction",
            ],
        }

    return {
        "problem_type": "General ML Starter",
        "reason": "The dataset is small or mixed, so a simple baseline model is a good starting point.",
        "algorithms": [
            "Random Forest",
            "Support Vector Machine",
            "Logistic Regression",
        ],
    }


def describe_dataset(df: pd.DataFrame) -> str:
    rows, cols = df.shape
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    text = [
        f"Dataset contains **{rows} rows** and **{cols} columns**.",
        f"There are **{len(num_cols)} numeric columns** and **{len(cat_cols)} categorical columns**.",
    ]

    if num_cols:
        numeric_preview = ", ".join(num_cols[:5])
        text.append(f"Numeric columns include: {numeric_preview}{'...' if len(num_cols) > 5 else ''}.")

    if cat_cols:
        categorical_preview = ", ".join(cat_cols[:5])
        text.append(f"Categorical columns include: {categorical_preview}{'...' if len(cat_cols) > 5 else ''}.")

    missing = df.isna().sum().sum()
    if missing > 0:
        text.append(f"There are **{missing} missing values** in the dataset.")
    else:
        text.append("There are **no missing values** in the dataset.")

    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        text.append(f"The dataset contains **{duplicate_count} duplicate rows**.")
    else:
        text.append("The dataset contains **no duplicate rows**.")

    return "\n\n".join(text)


@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)


def preprocess_data(df: pd.DataFrame, fill_missing: bool, scale_numeric: bool) -> pd.DataFrame:
    processed = df.copy()

    if fill_missing:
        numeric_cols = processed.select_dtypes(include=[np.number]).columns
        cat_cols = processed.select_dtypes(include=["object", "category", "bool"]).columns

        if len(numeric_cols) > 0:
            numeric_imputer = SimpleImputer(strategy="median")
            processed[numeric_cols] = numeric_imputer.fit_transform(processed[numeric_cols])

        if len(cat_cols) > 0:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            processed[cat_cols] = cat_imputer.fit_transform(processed[cat_cols])

    if scale_numeric:
        numeric_cols = processed.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            scaler = StandardScaler()
            processed[numeric_cols] = scaler.fit_transform(processed[numeric_cols])

    return processed


def column_summary(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column in df.select_dtypes(include=[np.number]).columns:
        summary = df[column].describe().to_frame().T
    else:
        top_values = df[column].value_counts(dropna=False).head(10)
        summary = top_values.rename_axis(column).reset_index(name="count")
    return summary


inject_css()
hero_section()

uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)

    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">Dataset Overview</div>
            <div class="section-muted">A quick glance at your data structure, quality, and shape.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(describe_dataset(df))

    metric_cols = st.columns(4)
    with metric_cols[0]:
        metric_card("Rows", f"{df.shape[0]:,}", "Observations")
    with metric_cols[1]:
        metric_card("Columns", f"{df.shape[1]:,}", "Variables")
    with metric_cols[2]:
        metric_card("Missing", f"{df.isna().sum().sum():,}", "Cells affected")
    with metric_cols[3]:
        duplicate_count = df.duplicated().sum()
        metric_card("Duplicates", f"{duplicate_count:,}", "Repeated rows")

    st.subheader("Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

    with st.expander("Dataset details"):
        st.write(df.info())
        st.write("### Missing values by column")
        st.dataframe(df.isna().sum().rename("missing_count").to_frame())

    st.sidebar.header("Recommendations")
    st.sidebar.markdown("Select the analysis areas you want to explore:")

    recommendation_options = {
        "Dataset Overview": True,
        "Data Quality Report": True,
        "Feature Analysis": True,
        "Statistical Summary": True,
        "Data Visualizations": True,
        "Correlation Analysis": True,
        "Preprocessing Recommendations": True,
        "AI Dataset Summary": False,
        "Key Insights": True,
        "Machine Learning Readiness": False,
    }

    selected_recommendations = {}
    for label, default in recommendation_options.items():
        selected_recommendations[label] = st.sidebar.checkbox(label, value=default, key=f"rec_{label.lower().replace(' ', '_')}")

    if "show_selected" not in st.session_state:
        st.session_state.show_selected = False

    if st.sidebar.button("Show selected recommendations", key="show_selected_btn"):
        st.session_state.show_selected = True

    show_selected = st.session_state.show_selected

    fill_missing = True
    scale_numeric = False
    remove_duplicates = True

    if remove_duplicates:
        before = df.shape[0]
        df = df.drop_duplicates().reset_index(drop=True)
        after = df.shape[0]
        st.sidebar.write(f"Removed {before - after} duplicate rows.")

    processed_df = preprocess_data(df, fill_missing=fill_missing, scale_numeric=scale_numeric)

    if show_selected:
        if selected_recommendations["Dataset Overview"]:
            st.markdown(
                """
                <div class="glass-card">
                    <div class="section-title">Dataset Overview</div>
                    <div class="section-muted">A quick summary of the cleaned dataset structure.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(describe_dataset(processed_df))

        if selected_recommendations["Data Quality Report"]:
            st.markdown(
                """
                <div class="glass-card">
                    <div class="section-title">Data Quality Report</div>
                    <div class="section-muted">Review the quality of the dataset after preprocessing.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.dataframe(processed_df.head(10), use_container_width=True)
            with st.expander("Missing values and details"):
                st.dataframe(processed_df.isna().sum().rename("missing_count").to_frame())

        if selected_recommendations["Feature Analysis"]:
            st.header("Feature Analysis")
            column_choice = st.selectbox("Choose a column to inspect", processed_df.columns)
            if column_choice:
                st.dataframe(column_summary(processed_df, column_choice))

        if selected_recommendations["Statistical Summary"]:
            st.header("Statistical Summary")
            numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                for col in numeric_cols[:5]:
                    values = processed_df[col].dropna()
                    if len(values) == 0:
                        continue
                    st.markdown(f"**{col}**: mean = {values.mean():.2f}, std = {values.std():.2f}, min = {values.min():.2f}, max = {values.max():.2f}")

        if selected_recommendations["Data Visualizations"]:
            st.markdown(
                """
                <div class="glass-card">
                    <div class="section-title">Data Visualizations</div>
                    <div class="section-muted">Visual insights from the processed dataset.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            kpi_cols = st.columns(4)
            with kpi_cols[0]:
                metric_card("Rows", f"{processed_df.shape[0]:,}", "Observations after cleaning")
            with kpi_cols[1]:
                metric_card("Columns", f"{processed_df.shape[1]:,}", "Features available")
            with kpi_cols[2]:
                metric_card("Missing", f"{processed_df.isna().sum().sum():,}", "Remaining empty cells")
            with kpi_cols[3]:
                metric_card("Numeric", f"{len(processed_df.select_dtypes(include=[np.number]).columns):,}", "Numeric fields")

            default_visualizations = ["Donut chart", "Clustered bar chart", "Column chart", "Scatter plot", "Gauge"]
            if "show_visualizations" not in st.session_state or not isinstance(st.session_state.show_visualizations, list):
                st.session_state.show_visualizations = []
            if "visualizations_ready" not in st.session_state:
                st.session_state.visualizations_ready = False

            st.markdown("<div class='section-muted'>Pick the charts you want, then click the button to render only those visuals.</div>", unsafe_allow_html=True)
            selected_visualizations = st.multiselect(
                "Choose visualizations",
                default_visualizations,
                default=st.session_state.show_visualizations,
                key="selected_visualizations",
            )

            if st.button("Show visualizations", key="show_visualizations_btn"):
                st.session_state.show_visualizations = selected_visualizations
                st.session_state.visualizations_ready = True

            top_n = st.slider("Select how many top values to display", min_value=3, max_value=10, value=5)
            all_cols = processed_df.columns.tolist()
            if all_cols:
                col_a, col_b = st.columns(2)
                with col_a:
                    first_col = st.selectbox("First column", all_cols, index=0, key="dashboard_col_a")
                with col_b:
                    second_choices = [c for c in all_cols if c != first_col]
                    second_col = st.selectbox(
                        "Second column",
                        second_choices if second_choices else all_cols,
                        index=0,
                        key="dashboard_col_b",
                    )
            else:
                first_col = None
                second_col = None

            if st.session_state.visualizations_ready:
                if "Donut chart" in st.session_state.show_visualizations:
                    st.markdown("<div class='dashboard-card'><div class='dashboard-title'>Donut Chart</div><div class='dashboard-subtitle'>Column type distribution for your dataset.</div></div>", unsafe_allow_html=True)
                    st.pyplot(build_donut_chart(processed_df), bbox_inches="tight")

                if "Clustered bar chart" in st.session_state.show_visualizations:
                    if not all_cols:
                        st.info("No columns are available to compare yet.")
                    else:
                        st.markdown("<div class='dashboard-card'><div class='dashboard-title'>Clustered Bar Chart</div><div class='dashboard-subtitle'>Compare top values between two columns.</div></div>", unsafe_allow_html=True)
                        st.pyplot(build_clustered_bar_chart(processed_df, first_col, second_col, top_n), bbox_inches="tight")

                if "Column chart" in st.session_state.show_visualizations:
                    if not all_cols:
                        st.info("No columns are available to compare yet.")
                    else:
                        st.markdown("<div class='dashboard-card'><div class='dashboard-title'>Column Chart</div><div class='dashboard-subtitle'>Top values for one selected column.</div></div>", unsafe_allow_html=True)
                        selected_col = st.selectbox("Choose column for column chart", all_cols, index=0, key="column_chart_col")
                        st.pyplot(build_column_chart(processed_df, selected_col, top_n), bbox_inches="tight")

                if "Scatter plot" in st.session_state.show_visualizations:
                    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_cols) < 2:
                        st.info("Not enough numeric columns for a scatter plot.")
                    else:
                        st.markdown("<div class='dashboard-card'><div class='dashboard-title'>Scatter Plot</div><div class='dashboard-subtitle'>Relationship between two numeric features.</div></div>", unsafe_allow_html=True)
                        scatter_a, scatter_b = st.columns(2)
                        with scatter_a:
                            x_col = st.selectbox("X-axis", numeric_cols, index=0, key="scatter_x_col")
                        with scatter_b:
                            y_choices = [c for c in numeric_cols if c != x_col]
                            y_col = st.selectbox(
                                "Y-axis",
                                y_choices if y_choices else numeric_cols,
                                index=0,
                                key="scatter_y_col",
                            )
                        st.pyplot(build_scatter_plot(processed_df, x_col, y_col), bbox_inches="tight")

                if "Gauge" in st.session_state.show_visualizations:
                    st.markdown("<div class='dashboard-card'><div class='dashboard-title'>Data Completeness Gauge</div><div class='dashboard-subtitle'>Missing values and dataset completeness.</div></div>", unsafe_allow_html=True)
                    st.pyplot(build_gauge_chart(processed_df), bbox_inches="tight")

                if not st.session_state.show_visualizations:
                    st.info("Select at least one visualization and click the button to display it.")
            else:
                st.info("Choose a chart type and click Show visualizations to render the dashboard.")

        if selected_recommendations["Correlation Analysis"]:
            st.header("Correlation Analysis")
            corr_fig = build_correlation_heatmap(processed_df)
            if corr_fig is not None:
                st.pyplot(corr_fig)
            else:
                st.info("Not enough numeric columns for correlation analysis.")

        if selected_recommendations["Preprocessing Recommendations"]:
            st.header("Preprocessing Recommendations")
            st.write("Recommended actions based on the current dataset:")
            st.write("- Fill missing values when needed")
            st.write("- Remove duplicate rows")
            st.write("- Scale numeric features for model readiness")

        if selected_recommendations["AI Dataset Summary"]:
            st.header("AI Dataset Summary")
            st.write(describe_dataset(processed_df))

        if selected_recommendations["Key Insights"]:
            st.header("Key Insights")
            st.write(f"- The dataset has {processed_df.shape[0]} rows and {processed_df.shape[1]} columns.")
            st.write(f"- Missing values remaining: {processed_df.isna().sum().sum()}")
            st.write(f"- Duplicate rows removed: {df.duplicated().sum()}")

        if selected_recommendations["Machine Learning Readiness"]:
            st.header("Machine Learning Readiness")
            readiness_score = max(0, 100 - processed_df.isna().sum().sum() * 5 - df.duplicated().sum() * 10)
            st.write(f"Readiness score: {min(readiness_score, 100)}%")

            ml_suggestion = recommend_ml_algorithms(processed_df)
            st.success(f"Suggested problem type: {ml_suggestion['problem_type']}")
            st.write(ml_suggestion["reason"])
            st.write("Recommended algorithms:")
            for algo in ml_suggestion["algorithms"]:
                st.write(f"- {algo}")
    else:
        st.info("Choose one or more recommendations and click the button to display them.")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Analysis Workflow ?...**")
    st.sidebar.markdown(
        "1. 📂 Upload your dataset\n"
        "2. 🔍 Explore the dataset overview\n"
        "3. 📊 Analyze data quality\n"
        "4. 📈 Visualize important patterns\n"
        "5. ^_^ Understand AI-generated insights\n"
        "6. 👩‍💻 Review preprocessing recommendations\n"
        "7. 🚀 Get your dataset ready for machine learning")
else:
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">🔔Let's Explore Your Data</div>
            <div class="section-muted">Upload a CSV or Excel file to explore your data with AI-powered analysis, interactive visualizations and intelligent preprocessing recommendations.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.title("🎓 Start Your Data Exploration Journey")
    st.write("📂 Upload a CSV file to start exploring your dataset.")
    st.write(
        "This app is designed to help beginners understand dataset structure, perform basic preprocessing and interpret the data with simple visualizations."
    )
    st.write("Supported file type: `.csv`. Example datasets: customer data, health records, sales data and more.")
