import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Dataset Analysis App", layout="wide")

st.title("Dataset Analysis and Explanation Web App")
st.markdown(
    "This app helps beginners understand a dataset, preprocess it, and generate easy-to-read summaries and visualizations."
)


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


uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)

    st.header("Dataset Overview")
    st.markdown(describe_dataset(df))

    st.subheader("Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

    with st.expander("Dataset details"):
        st.write(df.info())
        st.write("### Missing values by column")
        st.dataframe(df.isna().sum().rename("missing_count").to_frame())

    st.sidebar.header("Preprocessing Options")
    fill_missing = st.sidebar.checkbox("Fill missing values", value=True)
    scale_numeric = st.sidebar.checkbox("Scale numeric columns", value=False)
    remove_duplicates = st.sidebar.checkbox("Remove duplicate rows", value=True)

    if remove_duplicates:
        before = df.shape[0]
        df = df.drop_duplicates().reset_index(drop=True)
        after = df.shape[0]
        st.sidebar.write(f"Removed {before - after} duplicate rows.")

    processed_df = preprocess_data(df, fill_missing=fill_missing, scale_numeric=scale_numeric)

    st.header("Preprocessed Dataset")
    st.write("Use the options on the left to clean the data before analysis.")
    st.dataframe(processed_df.head(10), use_container_width=True)

    st.header("Beginner-Friendly Summary")
    numeric_cols = processed_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = processed_df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    if numeric_cols:
        st.subheader("Numeric column insights")
        for col in numeric_cols[:5]:
            values = processed_df[col].dropna()
            if len(values) == 0:
                continue
            st.markdown(f"**{col}**: mean = {values.mean():.2f}, std = {values.std():.2f}, min = {values.min():.2f}, max = {values.max():.2f}")

    if cat_cols:
        st.subheader("Categorical column insights")
        for col in cat_cols[:5]:
            top = processed_df[col].value_counts(dropna=False).head(3)
            st.markdown(f"**{col}** top values:")
            st.write(top)

    st.header("Column Explorer")
    column_choice = st.selectbox("Choose a column to inspect", processed_df.columns)
    if column_choice:
        st.dataframe(column_summary(processed_df, column_choice))

        if column_choice in numeric_cols:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(processed_df[column_choice].dropna(), kde=True, ax=ax)
            ax.set_title(f"Distribution of {column_choice}")
            st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(x=processed_df[column_choice].dropna(), ax=ax)
            ax.set_title(f"Boxplot of {column_choice}")
            st.pyplot(fig)
        else:
            fig, ax = plt.subplots(figsize=(8, 4))
            top_counts = processed_df[column_choice].value_counts(dropna=False).head(20)
            sns.barplot(x=top_counts.values, y=top_counts.index, ax=ax)
            ax.set_title(f"Frequency of {column_choice}")
            ax.set_xlabel("Count")
            st.pyplot(fig)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**How to use this app:**")
    st.sidebar.markdown(
        "1. Upload a CSV dataset.\n"
        "2. Review dataset overview and missing values.\n"
        "3. Use preprocessing options to clean the data.\n"
        "4. Explore columns and view beginner-friendly summaries."
    )
else:
    st.write("Upload a CSV file to start exploring your dataset.")
    st.write(
        "This app is designed to help beginners understand dataset structure, perform basic preprocessing, and interpret the data with simple visualizations."
    )
    st.write("Supported file type: `.csv`. Example datasets: customer data, health records, sales data, and more.")
