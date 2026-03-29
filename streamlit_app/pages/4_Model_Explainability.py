import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

st.title("Model Explainability")
st.markdown("SHAP-based analysis of how the model makes fraud decisions.")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Global Importance", "Feature Deep Dive", "Decision Examples"])

with tab1:
    st.markdown("### Global Feature Importance (Mean |SHAP|)")
    st.markdown("""
    SHAP values quantify each feature's contribution to individual predictions.
    The bar plot shows the **average absolute impact** across all test set predictions.
    """)

    bar_path = BASE_DIR / "reports" / "shap_global_bar.png"
    if bar_path.exists():
        st.image(str(bar_path), caption="Global Feature Importance (Mean |SHAP|)",
                 use_container_width=True)
    else:
        st.warning("SHAP bar plot not found. Run the notebook to generate it.")

    st.markdown("---")

    st.markdown("### SHAP Value Distribution (Beeswarm)")
    st.markdown("""
    The beeswarm plot shows how feature **values** (color) affect model **output** (x-axis).
    Red = high feature value, Blue = low feature value.
    """)

    bee_path = BASE_DIR / "reports" / "shap_beeswarm.png"
    if bee_path.exists():
        st.image(str(bee_path), caption="SHAP Value Distribution",
                 use_container_width=True)
    else:
        st.warning("SHAP beeswarm plot not found. Run the notebook to generate it.")

    st.markdown("---")

    st.markdown("### Key Insights")
    st.markdown("""
    - **`b_k_ratio`** is the #1 predictor — the engineered B/K interaction captures
      the strongest fraud signal
    - Geographic origin (`j_` features) ranks high — region is a critical risk factor
    - **5 of the top 10 features are engineered** — feature engineering from EDA
      adds measurable predictive value
    - **`cluster_id`** in top 10 — behavioral archetypes capture multi-feature patterns
      that no single feature can represent
    - The model uses both **linear signals** (direct feature values) and **non-linear
      combinations** (interactions, ratios, cluster assignments)
    """)

with tab2:
    st.markdown("### SHAP Dependence Plots — Top 3 Features")
    st.markdown("""
    Dependence plots show how a specific feature's value affects the SHAP output,
    colored by the feature with the strongest interaction effect.
    """)

    dep_path = BASE_DIR / "reports" / "shap_dependence_top3.png"
    if dep_path.exists():
        st.image(str(dep_path), caption="SHAP Dependence — Top 3 Features",
                 use_container_width=True)
    else:
        st.warning("SHAP dependence plot not found. Run the notebook to generate it.")

with tab3:
    st.markdown("### Individual Decision Examples")
    st.markdown("""
    Waterfall plots show how each feature **pushes** the prediction higher (toward fraud)
    or lower (toward legitimate) for a specific transaction.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### High-Risk Transaction (True Positive)")
        st.caption("A correctly identified fraudulent transaction — features that triggered the flag")
        hr_path = BASE_DIR / "reports" / "shap_waterfall_high_risk.png"
        if hr_path.exists():
            st.image(str(hr_path), use_container_width=True)
        else:
            st.warning("High-risk waterfall not found.")

    with col2:
        st.markdown("#### Low-Risk Transaction (True Negative)")
        st.caption("A correctly cleared legitimate transaction — features that reduced risk")
        lr_path = BASE_DIR / "reports" / "shap_waterfall_low_risk.png"
        if lr_path.exists():
            st.image(str(lr_path), use_container_width=True)
        else:
            st.warning("Low-risk waterfall not found.")

    st.markdown("---")
    st.markdown("""
    > **Interpretation guide**: In waterfall plots, red bars push the prediction
    > toward fraud (positive SHAP), blue bars push toward legitimate (negative SHAP).
    > The final prediction is the sum of all feature contributions plus the base value.
    """)
