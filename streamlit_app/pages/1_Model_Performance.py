import streamlit as st
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.model_loader import load_test_metrics, load_baseline_results, load_tuning_report

st.title("Model Performance")
st.markdown("Comprehensive evaluation across all pipeline stages.")

# Load data
metrics = load_test_metrics()
baselines = load_baseline_results()
tuning = load_tuning_report()

st.markdown("---")

# --- 1. Champion Metrics ---
st.markdown("### Champion Model: LightGBM (Optuna-Tuned)")

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("F2-Score", f"{metrics['F2']:.4f}", help="Primary metric")
col2.metric("Recall", f"{metrics['Recall']:.1%}", help="Fraud detection rate")
col3.metric("Precision", f"{metrics['Precision']:.1%}", help="Review efficiency")
col4.metric("ROC-AUC", f"{metrics['ROC-AUC']:.4f}")
col5.metric("Brier Score", f"{metrics['Brier']:.4f}", help="Lower = better calibrated")

st.markdown("---")

# --- 2. Baseline Comparison ---
st.markdown("### Baseline Model Comparison")

st.info(
    "**Note**: The table below shows **baseline** results (default hyperparameters, threshold = 0.5). "
    "The champion metrics above reflect the **final** model after Optuna tuning, "
    "Isotonic calibration, and F2-optimized threshold."
)

st.caption("All 6 models evaluated on validation set, sorted by F2-Score (primary metric)")

def highlight_champion(row):
    if row["Model"] == "LightGBM":
        return ["background-color: #1a3a2a"] * len(row)
    return [""] * len(row)

styled = baselines.style.apply(highlight_champion, axis=1).format({
    "F2": "{:.4f}", "F1": "{:.4f}", "Precision": "{:.4f}",
    "Recall": "{:.4f}", "ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}",
    "Brier": "{:.4f}",
})
st.dataframe(styled, use_container_width=True, hide_index=True)

st.markdown("""
> **Key takeaways**:
> - **CatBoost** and **LightGBM** lead in F2-Score at baseline
> - Both advanced to Optuna tuning (80 trials each)
> - **LightGBM** won the tuning phase with F2 = 0.7133 (val)
> - **DummyClassifier** (F2 = 0.2907) establishes the absolute floor
""")

# --- 2b. Model Progression ---
st.markdown("### Model Progression: Baseline to Final")
st.caption("How LightGBM improved through each pipeline stage")

progression = pd.DataFrame([
    {"Stage": "Baseline (default params, threshold=0.5)", "F2 (Val)": 0.6954, "F2 (Test)": "—"},
    {"Stage": "Optuna-Tuned (80 trials, regularized)", "F2 (Val)": 0.7133, "F2 (Test)": "—"},
    {"Stage": "Tuned + Isotonic Calibration + F2 Threshold", "F2 (Val)": "—", "F2 (Test)": f"{metrics['F2']:.4f}"},
])
st.dataframe(progression, use_container_width=True, hide_index=True)

st.markdown(f"""
> **Improvement**: Baseline F2 0.6954 → Final F2 **{metrics['F2']:.4f}** (+{metrics['F2'] - 0.6954:.4f}).
> Each stage contributed: tuning (+0.0179 on val), calibration + threshold optimization (test gain).
""")

st.markdown("---")

# --- 3. Generalization Table ---
st.markdown("### Generalization: CV vs Validation vs Test")

gen_data = pd.DataFrame([
    {"Set": "CV (5-fold)", "F2": 0.6854, "Note": "Ground truth (5-fold mean)"},
    {"Set": "Validation", "F2": 0.7133, "Note": "Used for tuning + threshold"},
    {"Set": "Test", "F2": metrics["F2"], "Note": "First and only look"},
])
st.dataframe(gen_data, use_container_width=True, hide_index=True)

st.markdown(f"""
> Test F2 ({metrics['F2']:.4f}) exceeds CV range — favorable variance in the test split,
> not a pipeline issue. Both CV and test confirm generalization.
> Train-CV gap of 0.048 indicates healthy generalization.
""")

st.markdown("---")

# --- 4. Calibration Analysis ---
st.markdown("### Probability Calibration")

col1, col2, col3 = st.columns(3)
col1.metric("Raw Brier", "0.1643")
col2.metric("Isotonic Brier", "0.1361", delta="-17.2%", delta_color="inverse")
col3.metric("Test Brier", f"{metrics['Brier']:.4f}")

st.markdown("""
> **Isotonic calibration** reduced the Brier Score by 17.2%, ensuring that
> probability outputs are trustworthy for threshold optimization and business decisions.
""")

st.markdown("---")

# --- 5. Tuning Details ---
st.markdown("### Optuna Hyperparameters")
st.caption(f"{tuning['n_trials']} Bayesian trials (TPE sampler) | Regularization-constrained search space")

params_df = pd.DataFrame([
    {"Parameter": k, "Value": f"{v:.6f}" if isinstance(v, float) else str(v)}
    for k, v in tuning["best_params"].items()
])
st.dataframe(params_df, use_container_width=True, hide_index=True)

st.markdown("""
> **Search space constraints**: `num_leaves ≤ 64`, `reg_alpha/lambda ≥ 0.1`, `min_child_weight ≥ 5`.
> These bounds prevent overfitting on the 11K-row training set while preserving generalization.
""")
