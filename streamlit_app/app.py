import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Fraud Detection Intelligence",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# Sidebar branding
st.sidebar.title("Fraud Detection")
st.sidebar.caption("E-Commerce Transaction Intelligence System")
st.sidebar.markdown("---")
st.sidebar.markdown("**Model**: LightGBM (Optuna-tuned)")
st.sidebar.markdown("**Calibration**: Isotonic")
st.sidebar.markdown("**Primary Metric**: F2-Score")
st.sidebar.markdown("---")
st.sidebar.markdown("Built by **Maria Camila Gonzalez Nuñez**")

# Main content
st.title("Fraud Detection Intelligence System")
st.markdown("### E-Commerce Transaction Analysis & Real-Time Fraud Scoring")

st.markdown("""
This system uses a **LightGBM model** trained on 16,879 anonymized e-commerce
transactions to identify fraudulent activity. The model was tuned with Optuna
(80 Bayesian trials), calibrated with Isotonic regression, and optimized for
**F2-Score** to prioritize fraud detection over false alarm minimization.
""")

# Key metrics row
col1, col2, col3, col4 = st.columns(4)
col1.metric("F2-Score", "0.7164", help="Primary metric — recall-weighted")
col2.metric("Recall", "85.0%", help="Fraud detection rate")
col3.metric("Precision", "44.0%", help="Review efficiency")
col4.metric("Net Savings", "$74,308", help="Per test batch (2,532 transactions)")

st.markdown("---")

# Architecture overview
st.markdown("### Pipeline Architecture")
st.code("""
Raw Data -> Cleaning -> EDA (11 analyses) -> Feature Engineering (15 features)
-> Stratified Split (70/15/15) -> 6 Baseline Models -> Optuna Tuning (top 2)
-> Cross-Validation -> Isotonic Calibration -> F2 Threshold Optimization
-> SHAP Explainability -> Business Impact Analysis
""", language=None)

# Quick navigation
st.markdown("### Explore the System")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("**Model Performance**")
    st.caption("Metrics, comparison tables, calibration curves")
with col2:
    st.markdown("**ROI Simulator**")
    st.caption("Adjust threshold, see financial impact in real-time")
with col3:
    st.markdown("**Transaction Scanner**")
    st.caption("Score individual transactions with explanation")
with col4:
    st.markdown("**Model Explainability**")
    st.caption("SHAP analysis, feature importance, decision logic")
