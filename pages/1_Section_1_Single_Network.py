import streamlit as st

from src.section1 import ANALYSES

st.set_page_config(page_title="Section 1 — Single Network", page_icon="⚽", layout="wide")
st.title("Section 1 — Single-Network Analyses (#1–15)")
st.caption("Metrics on each network in isolation. Pick an analysis; controls for grain "
           "(club/league), top-X and Outside-System handling appear per analysis.")

choice = st.sidebar.radio("Analysis", list(ANALYSES.keys()), key="s1_choice")
st.divider()
ANALYSES[choice]()
