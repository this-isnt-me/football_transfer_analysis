import streamlit as st

from src.section3 import ANALYSES

st.set_page_config(page_title="Section 3 — Movement vs Finance", page_icon="⚽", layout="wide")
st.title("Section 3 — Cross-Network, Same Granularity: Movement vs Finance (#19–25)")
st.caption("Movement and finance joined per deal via P1 (transfer_id). Movement runs "
           "sell→buy, finance buy→sell — the join aligns the flip. The ~22k unmatched "
           "(NULL-fee) moves are excluded from fee stats, never zeroed; fee stats use medians.")

choice = st.sidebar.radio("Analysis", list(ANALYSES.keys()), key="s3_choice")
st.divider()
ANALYSES[choice]()
