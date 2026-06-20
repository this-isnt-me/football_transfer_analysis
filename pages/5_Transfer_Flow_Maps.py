from src.section5 import render
from src.theme import apply_chrome
import streamlit as st

st.set_page_config(page_title="Transfer Flow Maps", page_icon="⚽", layout="wide")
apply_chrome()
st.title("Transfer Flow Maps")
st.caption("Follow transfers between leagues over time. The movement map follows the players "
           "(selling → buying league); the finance map follows the money (paying → receiving "
           "league) — so the two are mirror images. Outside System can be toggled on or off.")

render()
