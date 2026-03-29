import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, fbeta_score
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.model_loader import load_test_predictions

st.title("ROI Simulator")
st.markdown("Adjust the decision threshold and business parameters to see real-time financial impact.")

# Load precomputed predictions
pred_df = load_test_predictions()
y_true = pred_df["y_true"].values
y_proba = pred_df["y_proba"].values

# --- SIDEBAR CONTROLS ---
st.sidebar.markdown("### Business Parameters")
avg_fraud_amount = st.sidebar.slider("Avg Fraud Amount ($)", 50, 500, 139, step=10)
review_cost = st.sidebar.slider("Review Cost per Flag ($)", 1, 50, 7)
friction_cost = st.sidebar.slider("Customer Friction Cost ($)", 0, 20, 3)
capacity_pct = st.sidebar.slider("Max Review Capacity (%)", 10, 100, 50, step=5)

st.sidebar.markdown("### Decision Threshold")
threshold = st.sidebar.slider("Classification Threshold", 0.05, 0.95, 0.19, step=0.01)

# --- COMPUTE METRICS AT CURRENT THRESHOLD ---
y_pred = (y_proba >= threshold).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

fraud_prevented = tp * avg_fraud_amount
fraud_missed = fn * avg_fraud_amount
false_alarm_cost = fp * (review_cost + friction_cost)
net_savings = fraud_prevented - false_alarm_cost
flagged_pct = (tp + fp) / len(y_true) * 100
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0

# --- METRICS CARDS ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Net Savings", f"${net_savings:,.0f}")
col2.metric("Fraud Detected", f"{tp}/{tp + fn} ({recall:.0%})")
col3.metric("False Alarms", f"{fp}")

over_capacity = flagged_pct > capacity_pct
col4.metric(
    "Flagged %", f"{flagged_pct:.1f}%",
    delta="Over capacity" if over_capacity else "Within capacity",
    delta_color="inverse" if over_capacity else "normal",
)

if over_capacity:
    st.warning(
        f"Review volume ({flagged_pct:.1f}%) exceeds team capacity ({capacity_pct}%). "
        f"Increase threshold or expand review team."
    )

st.markdown("---")

# --- ROI CURVE (interactive plotly) ---
st.markdown("### ROI vs Threshold")

thresholds_sweep = np.arange(0.05, 0.96, 0.01)
roi_values = []
f2_values = []
flagged_values = []

for t in thresholds_sweep:
    yp = (y_proba >= t).astype(int)
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_true, yp).ravel()
    roi_t = tp_t * (avg_fraud_amount - review_cost) - fp_t * (review_cost + friction_cost) - fn_t * avg_fraud_amount
    f2_t = fbeta_score(y_true, yp, beta=2) if yp.sum() > 0 else 0
    roi_values.append(roi_t)
    f2_values.append(f2_t)
    flagged_values.append((tp_t + fp_t) / len(y_true) * 100)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=thresholds_sweep, y=roi_values,
    name="Net ROI ($)", line=dict(color="#2ecc71", width=3),
    yaxis="y1",
))
fig.add_trace(go.Scatter(
    x=thresholds_sweep, y=f2_values,
    name="F2-Score", line=dict(color="#e74c3c", width=2, dash="dash"),
    yaxis="y2",
))

# Current threshold marker
fig.add_vline(x=threshold, line_dash="dot", line_color="#f39c12", line_width=2,
              annotation_text=f"Current: {threshold:.2f}", annotation_position="top")

# Capacity constraint band
capacity_threshold_idx = None
for i, fp_val in enumerate(flagged_values):
    if fp_val <= capacity_pct and (i == 0 or flagged_values[i - 1] > capacity_pct):
        capacity_threshold_idx = i

fig.update_layout(
    title="Financial Impact & F2-Score vs Decision Threshold",
    xaxis=dict(title="Threshold"),
    yaxis=dict(title="Net ROI ($)", title_font=dict(color="#2ecc71"), side="left"),
    yaxis2=dict(title="F2-Score", title_font=dict(color="#e74c3c"),
                overlaying="y", side="right", range=[0, 1]),
    height=450,
    template="plotly_dark",
    legend=dict(x=0.01, y=0.99),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# --- SCENARIO COMPARISON TABLE ---
st.markdown("### Scenario Comparison")

scenarios = []
for label, theta in [("Do Nothing", 1.0), ("Default (0.50)", 0.50),
                      (f"Current ({threshold:.2f})", threshold),
                      ("F2-Optimal (0.19)", 0.19)]:
    yp = (y_proba >= theta).astype(int)
    tn_s, fp_s, fn_s, tp_s = confusion_matrix(y_true, yp).ravel()
    roi_s = (tp_s * (avg_fraud_amount - review_cost)
             - fp_s * (review_cost + friction_cost)
             - fn_s * avg_fraud_amount)
    f2_s = fbeta_score(y_true, yp, beta=2) if yp.sum() > 0 else 0
    rec_s = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0
    prec_s = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0
    scenarios.append({
        "Scenario": label,
        "Threshold": f"{theta:.2f}",
        "Net ROI": f"${roi_s:,.0f}",
        "Recall": f"{rec_s:.1%}",
        "Precision": f"{prec_s:.1%}",
        "Flagged %": f"{(tp_s + fp_s) / len(y_true) * 100:.1f}%",
        "F2": f"{f2_s:.4f}",
    })

st.dataframe(pd.DataFrame(scenarios), use_container_width=True, hide_index=True)

st.markdown("---")

# --- CONFUSION MATRIX at current threshold ---
st.markdown("### Confusion Matrix")

col1, col2 = st.columns([1, 1])
with col1:
    cm_fig = go.Figure(data=go.Heatmap(
        z=[[tn, fp], [fn, tp]],
        x=["Predicted Legit", "Predicted Fraud"],
        y=["Actual Legit", "Actual Fraud"],
        text=[[f"TN: {tn}", f"FP: {fp}"], [f"FN: {fn}", f"TP: {tp}"]],
        texttemplate="%{text}",
        colorscale=[[0, "#1a1a2e"], [1, "#e74c3c"]],
        showscale=False,
    ))
    cm_fig.update_layout(
        title=f"Confusion Matrix (threshold = {threshold:.2f})",
        height=350, template="plotly_dark",
    )
    st.plotly_chart(cm_fig, use_container_width=True)

with col2:
    st.markdown("#### Financial Breakdown")
    st.markdown(f"""
    | Component | Amount |
    |-----------|--------|
    | Fraud Prevented (TP) | **+${fraud_prevented:,.0f}** |
    | Fraud Missed (FN) | **-${fraud_missed:,.0f}** |
    | False Alarm Cost (FP) | **-${false_alarm_cost:,.0f}** |
    | **Net Savings** | **${net_savings:,.0f}** |
    """)
