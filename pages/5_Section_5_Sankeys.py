import streamlit as st

from src.section5 import render

st.set_page_config(page_title="Section 5 — Temporal Sankeys", page_icon="⚽", layout="wide")
st.title("Section 5 — Temporal League-Level Sankeys (#33–37)")
st.caption("State-transition Sankeys: node per (league, stage), stages ordered by "
           "(season, window). Movement follows the player; finance follows the money "
           "(shown as-is) — the two are mirrored. Outside System is toggleable and pinned.")

render()
