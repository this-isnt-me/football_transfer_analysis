import streamlit as st

from src.section4 import ANALYSES

st.set_page_config(page_title="Section 4 — All Networks Combined", page_icon="⚽", layout="wide")
st.title("Section 4 — All Four Networks Combined (#26–32)")
st.caption("Finance aligned to movement by reversing the finance layer (both run sell→buy). "
           "P1 links corresponding edges, P2 does club→league rollups. Heavy metrics use "
           "igraph + leidenalg/infomap on top-N subgraphs where needed (labelled estimates); "
           "community detection is club-level only.")

choice = st.sidebar.radio("Analysis", list(ANALYSES.keys()), key="s4_choice")
st.divider()
ANALYSES[choice]()
