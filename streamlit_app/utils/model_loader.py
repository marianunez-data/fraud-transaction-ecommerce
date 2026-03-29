import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


@st.cache_resource
def load_model():
    """Load the champion model with metadata."""
    artifact = joblib.load(BASE_DIR / "models" / "champion_final.pkl")
    return artifact


@st.cache_data
def load_test_predictions():
    """Load precomputed test predictions for ROI simulator."""
    return pd.read_csv(BASE_DIR / "data" / "test_predictions.csv")


@st.cache_data
def load_baseline_results():
    """Load baseline model comparison results."""
    return pd.read_csv(BASE_DIR / "data" / "baseline_results.csv")


@st.cache_data
def load_test_metrics():
    """Load final test metrics."""
    import json
    with open(BASE_DIR / "reports" / "test_results.json") as f:
        return json.load(f)


@st.cache_data
def load_tuning_report():
    """Load Optuna tuning report."""
    import json
    with open(BASE_DIR / "reports" / "tuning_report.json") as f:
        return json.load(f)


@st.cache_data
def load_raw_dataset():
    """Load and preprocess the raw dataset for transaction scanner."""
    df = pd.read_csv(BASE_DIR / "data" / "fraud-prevention.csv")

    # Same cleaning as notebook
    df.columns = df.columns.str.strip().str.lower()
    for col in ["r", "monto", "q"]:
        df[col] = df[col].astype(str).str.replace(",", "", regex=False).astype(float)
    str_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip().str.lower()
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def preprocess_single_transaction(inputs, artifact):
    """
    Preprocess a single transaction dict into a feature vector
    matching the training pipeline.

    inputs: dict with raw feature values
    artifact: loaded model artifact with feature_names
    """
    feature_names = artifact["feature_names"]

    row = pd.DataFrame([inputs])

    # Row-level features
    row["c_is_null"] = row["c"].isnull().astype(int)
    row["k_is_null"] = row["k"].isnull().astype(int)
    row["c"] = row["c"].fillna(0)
    row["k"] = row["k"].fillna(0)
    row["log_monto"] = np.log1p(row["monto"])
    row["monto_bin"] = pd.cut(
        row["monto"], bins=[-0.01, 50, 200, 1000, 5000, np.inf],
        labels=[0, 1, 2, 3, 4]
    ).astype(float)

    sparse_features = ["a", "d", "e", "f", "g", "h", "i", "l", "o", "q", "r"]
    row["zero_count"] = (row[sparse_features] == 0).sum(axis=1)
    row["binary_sum"] = row[sparse_features].sum(axis=1)
    row["high_risk_flag_count"] = (row[["h", "o", "r", "q"]] != 0).sum(axis=1)
    row["activity_score"] = row["l"] + row["m"] + row["n"]
    row["risk_signal_sum"] = row["o"] + row["p"]

    # Interactions (post-impute)
    row["b_k_interaction"] = row["b"] * row["k"]
    row["b_k_ratio"] = row["b"] / (row["k"] + 0.001)
    row["i_e_interaction"] = row["i"] * row["e"]
    row["s_k_interaction"] = row["s"] * row["k"]

    # Cluster placeholder (use 0 — we can't run KMeans on a single row meaningfully)
    row["cluster_id"] = 0

    # OHE for j
    j_val = row["j"].values[0] if "j" in row.columns else "ar"
    row = row.drop(columns=["j"], errors="ignore")

    # Create all j_ columns as 0, then set the right one to 1
    for col in feature_names:
        if col.startswith("j_") and col not in row.columns:
            row[col] = 0

    # Map j value to OHE column
    j_col = f"j_j_{j_val}"
    if j_col in feature_names:
        row[j_col] = 1

    # Ensure all feature columns exist and are in the right order
    for col in feature_names:
        if col not in row.columns:
            row[col] = 0

    return row[feature_names]
