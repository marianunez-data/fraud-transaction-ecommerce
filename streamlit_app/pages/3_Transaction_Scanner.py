import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.model_loader import load_model, load_raw_dataset, preprocess_single_transaction

st.title("Transaction Scanner")
st.markdown("Enter transaction features to get a real-time fraud probability score.")

# Load model
artifact = load_model()
model = artifact["model"]
threshold = artifact["threshold"]

# Load dataset for random sampling
df = load_raw_dataset()

st.markdown("---")

mode = st.radio("Input Mode", ["Random Sample from Test Set", "Manual Entry"], horizontal=True)

if mode == "Random Sample from Test Set":
    if st.button("Get Random Transaction"):
        # Pick a random row
        idx = np.random.randint(0, len(df))
        row = df.iloc[idx]

        st.markdown("#### Transaction Features")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**Core Features**")
            st.write(f"Amount (monto): ${row['monto']:,.2f}")
            st.write(f"Region (j): {row['j'].upper()}")
            st.write(f"Feature B: {row['b']}")
            st.write(f"Feature S: {row['s']:.2f}")
        with col2:
            st.markdown("**Behavioral Signals**")
            st.write(f"Feature K: {row['k'] if pd.notna(row['k']) else 'NULL'}")
            st.write(f"Feature C: {row['c'] if pd.notna(row['c']) else 'NULL'}")
            st.write(f"Feature M: {row['m']}")
            st.write(f"Feature N: {row['n']}")
        with col3:
            st.markdown("**Risk Flags**")
            st.write(f"Feature H: {row['h']}")
            st.write(f"Feature O: {row['o']}")
            st.write(f"Feature R: {row['r']}")
            st.write(f"Feature Q: {row['q']}")

        # Build input dict
        feature_cols = [c for c in df.columns if c != "fraude"]
        inputs = {col: row[col] for col in feature_cols}

        # Preprocess and score
        X_single = preprocess_single_transaction(inputs, artifact)
        prob = model.predict_proba(X_single)[:, 1][0]
        actual = int(row["fraude"]) if "fraude" in row.index else None

        st.markdown("---")
        st.markdown("#### Fraud Assessment")

        # Risk tier
        if prob > 0.7:
            st.error(f"BLOCK — High confidence fraud | P(fraud) = {prob:.1%}")
        elif prob > threshold:
            st.warning(f"FLAG — Manual review required | P(fraud) = {prob:.1%}")
        else:
            st.success(f"CLEAR — Transaction approved | P(fraud) = {prob:.1%}")

        # Probability gauge
        col1, col2, col3 = st.columns(3)
        col1.metric("P(fraud)", f"{prob:.1%}")
        col2.metric("Threshold", f"{threshold:.2f}")
        decision = "BLOCK" if prob >= 0.70 else "FLAG" if prob >= threshold else "CLEAR"
        col3.metric("Decision", decision)

        if actual is not None:
            st.info(f"Ground truth: {'FRAUD' if actual == 1 else 'LEGITIMATE'}")

elif mode == "Manual Entry":
    st.markdown("#### Enter Transaction Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Transaction Details**")
        monto = st.number_input("Amount (monto)", 0.0, 50000.0, 100.0, step=10.0)
        region = st.selectbox("Region (j)", ["ar", "br", "mx", "uy", "es", "us",
                                              "cl", "co", "au", "ca", "de", "gb",
                                              "it", "fr", "pt", "pe", "py", "ec", "ve"])
        b_val = st.number_input("Feature B", 0.0, 30.0, 5.0, step=0.5)
        s_val = st.number_input("Feature S", 0.0, 200.0, 10.0, step=1.0)

    with col2:
        st.markdown("**Continuous Features**")
        k_val = st.number_input("Feature K (0 = NULL)", 0.0, 10.0, 0.0, step=0.1)
        k_is_null = st.checkbox("K is NULL", value=True)
        c_val = st.number_input("Feature C", 0.0, 500000.0, 30000.0, step=1000.0)
        c_is_null = st.checkbox("C is NULL", value=False)
        m_val = st.number_input("Feature M", 0.0, 50.0, 1.0, step=0.5)
        n_val = st.number_input("Feature N", 0.0, 50.0, 1.0, step=0.5)

    with col3:
        st.markdown("**Risk Flags & Others**")
        a_val = st.number_input("Feature A", 0.0, 100.0, 0.0, step=1.0)
        d_val = st.number_input("Feature D", 0.0, 10.0, 0.0, step=1.0)
        e_val = st.number_input("Feature E", 0.0, 100.0, 0.0, step=1.0)
        f_val = st.number_input("Feature F", 0.0, 10.0, 0.0, step=0.1)
        g_val = st.number_input("Feature G", 0.0, 10.0, 0.0, step=1.0)
        h_val = st.number_input("Feature H", 0.0, 10.0, 0.0, step=1.0)
        i_val = st.number_input("Feature I", 0.0, 100.0, 0.0, step=1.0)
        l_val = st.number_input("Feature L", 0.0, 50.0, 0.0, step=1.0)
        o_val = st.number_input("Feature O", 0.0, 100.0, 0.0, step=1.0)
        p_val = st.number_input("Feature P", 0.0, 100.0, 0.0, step=1.0)
        q_val = st.number_input("Feature Q", 0.0, 1000.0, 0.0, step=10.0)
        r_val = st.number_input("Feature R", 0.0, 10000.0, 0.0, step=100.0)

    if st.button("Score Transaction"):
        inputs = {
            "monto": monto, "j": region, "b": b_val, "s": s_val,
            "k": np.nan if k_is_null else k_val,
            "c": np.nan if c_is_null else c_val,
            "m": m_val, "n": n_val,
            "a": a_val, "d": d_val, "e": e_val, "f": f_val, "g": g_val,
            "h": h_val, "i": i_val, "l": l_val, "o": o_val, "p": p_val,
            "q": q_val, "r": r_val,
        }

        X_single = preprocess_single_transaction(inputs, artifact)
        prob = model.predict_proba(X_single)[:, 1][0]

        st.markdown("---")
        st.markdown("#### Fraud Assessment")

        if prob > 0.7:
            st.error(f"BLOCK — High confidence fraud | P(fraud) = {prob:.1%}")
        elif prob > threshold:
            st.warning(f"FLAG — Manual review required | P(fraud) = {prob:.1%}")
        else:
            st.success(f"CLEAR — Transaction approved | P(fraud) = {prob:.1%}")

        col1, col2, col3 = st.columns(3)
        col1.metric("P(fraud)", f"{prob:.1%}")
        col2.metric("Threshold", f"{threshold:.2f}")
        decision = "BLOCK" if prob >= 0.70 else "FLAG" if prob >= threshold else "CLEAR"
        col3.metric("Decision", decision)
