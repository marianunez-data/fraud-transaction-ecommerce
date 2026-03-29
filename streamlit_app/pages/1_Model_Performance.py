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
    "Isotonic calibration, and F2-optimized threshold (0.19)."
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
> - **LightGBM** won the tuning phase with F2 = 0.7139 (val)
> - **DummyClassifier** (F2 = 0.2907) establishes the absolute floor
""")

# --- 2b. Model Progression ---
st.markdown("### Model Progression: Baseline to Final")
st.caption("How LightGBM improved through each pipeline stage")

progression = pd.DataFrame([
    {"Stage": "Baseline (default params, threshold=0.5)", "F2 (Val)": 0.6954, "F2 (Test)": "—"},
    {"Stage": "Optuna-Tuned (80 trials)", "F2 (Val)": 0.7139, "F2 (Test)": "—"},
    {"Stage": "Tuned + Isotonic Calibration + F2 Threshold (0.19)", "F2 (Val)": "—", "F2 (Test)": "0.7221"},
])
st.dataframe(progression, use_container_width=True, hide_index=True)

st.markdown("""
> **Improvement**: Baseline F2 0.6954 → Final F2 **0.7221** (+0.0267).
> Each stage contributed: tuning (+0.0185 on val), calibration + threshold optimization (test gain).
""")

st.markdown("---")

# --- 3. Generalization Table ---
st.markdown("### Generalization: CV vs Validation vs Test")

gen_data = pd.DataFrame([
    {"Set": "CV (5-fold)", "F2": 0.6786, "Note": "Ground truth (5-fold mean)"},
    {"Set": "Validation", "F2": 0.7139, "Note": "Used for tuning + threshold"},
    {"Set": "Test", "F2": 0.7221, "Note": "First and only look"},
])
st.dataframe(gen_data, use_container_width=True, hide_index=True)

st.markdown("""
> Test F2 (0.7221) exceeds CV range — favorable variance in the test split,
> not a pipeline issue. Both CV and test confirm generalization.
""")

st.markdown("---")

# --- 4. Calibration Analysis ---
st.markdown("### Probability Calibration")

col1, col2, col3 = st.columns(3)
col1.metric("Raw Brier", "0.1586")
col2.metric("Isotonic Brier", "0.1343", delta="-15.3%", delta_color="inverse")
col3.metric("Test Brier", f"{metrics['Brier']:.4f}")

st.markdown("""
> **Isotonic calibration** reduced the Brier Score by 15.3%, ensuring that
> probability outputs are trustworthy for threshold optimization and business decisions.
""")

st.markdown("---")

# --- 5. Overfitting Note ---
st.markdown("### Overfitting Analysis")

st.warning("""
**Train-CV gap: 0.10** — The model (num_leaves=150, max_depth=7) has enough capacity
to partially memorize training patterns on this 16K-row dataset.

**Root cause**: High num_leaves with moderate depth gives excessive flexibility.

**Production fix**: Constrain Optuna search space to `num_leaves <= 64` and increase
`reg_alpha` bounds to enforce stronger regularization.

**Impact**: The test F2 (0.7221) confirms generalization despite the gap. The model
works — but a tighter regularization would yield a more robust deployment.
""")

# --- 6. Tuning Details ---
st.markdown("### Optuna Hyperparameters")
st.caption(f"{tuning['n_trials']} Bayesian trials (TPE sampler)")

params_df = pd.DataFrame([
    {"Parameter": k, "Value": f"{v:.6f}" if isinstance(v, float) else str(v)}
    for k, v in tuning["best_params"].items()
])
st.dataframe(params_df, use_container_width=True, hide_index=True)
