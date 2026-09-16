"""
Streamlit app: pick two Pokemon, see the predicted winner, win probability,
and a SHAP-based breakdown of what drove the call.

Run with: streamlit run app.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
import pandas as pd
from explain import explain_prediction

st.set_page_config(page_title="Pokemon Battle Predictor", page_icon="\u26a1", layout="centered")

@st.cache_data
def load_stats():
    return pd.read_csv("data/pokemon_stats.csv").set_index("name")

stats = load_stats()
names = sorted(stats.index.tolist())

st.title("\u26a1 Pokemon battle predictor")
st.caption("Pick two Pokemon to predict the winner")

col1, col_vs, col2 = st.columns([5, 1, 5])
with col1:
    p1_name = st.selectbox("Pokemon 1", names, index=names.index("Charizard") if "Charizard" in names else 0)
with col_vs:
    st.markdown("<div style='padding-top: 2.2rem; text-align:center;'>vs</div>", unsafe_allow_html=True)
with col2:
    default_2 = "Venusaur" if "Venusaur" in names else names[1]
    p2_name = st.selectbox("Pokemon 2", names, index=names.index(default_2))

predict_clicked = st.button("\u2694\ufe0f Predict winner", use_container_width=True, type="primary")

if predict_clicked:
    if p1_name == p2_name:
        st.warning("Pick two different Pokemon.")
    else:
        p1_row = stats.loc[p1_name].to_dict()
        p1_row["name"] = p1_name
        p2_row = stats.loc[p2_name].to_dict()
        p2_row["name"] = p2_name

        win_prob_p1, contributions = explain_prediction(p1_row, p2_row)
        win_prob_p2 = 1 - win_prob_p1

        c1, c2 = st.columns(2)
        with c1:
            st.metric(f"{p1_name} win probability", f"{win_prob_p1 * 100:.0f}%")
        with c2:
            st.metric(f"{p2_name} win probability", f"{win_prob_p2 * 100:.0f}%")

        winner = p1_name if win_prob_p1 >= 0.5 else p2_name
        st.success(f"Predicted winner: **{winner}**")

        st.markdown("---")
        st.markdown("**What drove this prediction**")
        for label, value in contributions:
            # value is in log-odds/probability space, oriented toward p1 winning
            favors = p1_name if value > 0 else p2_name
            st.write(f"{label}: favors **{favors}**")
            st.progress(min(abs(value) * 2, 1.0))  # scaled for a readable bar length

st.markdown("---")
with st.expander("How this works"):
    st.write(
        "The model was trained on simulated multi-turn battles (speed decides "
        "turn order, damage uses type effectiveness and STAB, HP depletes each "
        "turn) rather than a simple stat comparison. Predictions come from a "
        "gradient-boosted model (XGBoost/LightGBM) trained on differential "
        "features like speed advantage, type matchup, and estimated damage output. "
        "SHAP values above show which of those features pushed the prediction "
        "toward each Pokemon."
    )
