import streamlit as st

from src.section2 import ANALYSES

st.set_page_config(page_title="Section 2 — Club vs League", page_icon="⚽", layout="wide")
st.title("Section 2 — Cross-Network, Same Type: Club vs League (#16–18)")
st.caption("Compare the club and league views of the same network type, bridged by the "
           "P2 (club, season, window) → league mapping. Aggregation stays within one type "
           "so direction (and the finance reversal) is preserved.")

choice = st.sidebar.radio("Analysis", list(ANALYSES.keys()), key="s2_choice")
st.divider()
ANALYSES[choice]()
