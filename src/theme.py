"""Brand theme: colour scheme, bento-box CSS, dark Plotly template, and logo.

One place for the app's look. ``apply_chrome()`` is called at the top of every
page (Home + the section pages); it pins the sidebar logo and injects the CSS.
Importing this module also registers + activates a dark Plotly template so every
chart matches the dark "bento" surfaces.

Colour scheme (fixed):
  * App background  #111214 (dark)
  * Highlights      #590606, #007480 (teal), #D1B49C (sand)
"""
from __future__ import annotations

from pathlib import Path

import plotly.io as pio
import streamlit as st

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
BG = "#111214"        # app background
CARD = "#1A1C20"      # bento surface (one step up from the background)
CARD_HI = "#202329"   # chips / inner cards
RED = "#590606"       # highlight 1
TEAL = "#007480"      # highlight 2
SAND = "#D1B49C"      # highlight 3
TEXT = "#E8E6E3"
MUTED = "#9AA0A6"
BORDER = "rgba(0,116,128,0.22)"

LOGO = Path(__file__).resolve().parent.parent / "logo" / "logo_only.png"

# Categorical colourway: brand-led but extended to >=12 legible-on-dark hues so
# many-series charts (11 leagues, 7 positions) still differentiate. Deep red
# #590606 reads as near-black on dark, so series use lighter brand-adjacent reds;
# the true highlight reds/teal are used for app chrome and accents.
COLORWAY = [
    "#33A0AC",  # teal (light)
    "#D1B49C",  # sand
    "#C0504D",  # brick red
    "#5FC7D1",  # pale teal
    "#E8CDB5",  # pale sand
    "#8C4B4B",  # muted red
    "#007480",  # teal (brand)
    "#B8906E",  # tan
    "#A63A3A",  # red
    "#9FE0E8",  # ice teal
    "#F0E0D0",  # cream
    "#732424",  # brick (dark)
]

# 11-colour league palette for the Sankeys (consistent with the colourway).
LEAGUE_PALETTE = COLORWAY[:11]


# --------------------------------------------------------------------------- #
# Plotly template (registered + activated on import)
# --------------------------------------------------------------------------- #
def _register_template() -> None:
    base = pio.templates["plotly_dark"]
    t = base.to_plotly_json()
    layout = t.setdefault("layout", {})
    layout["paper_bgcolor"] = "rgba(0,0,0,0)"   # transparent -> bento card shows through
    layout["plot_bgcolor"] = "rgba(0,0,0,0)"
    layout["colorway"] = COLORWAY
    layout["font"] = {"color": TEXT, "family": "Inter, Segoe UI, sans-serif", "size": 13}
    layout["title"] = {"font": {"color": SAND, "size": 16}}
    layout["legend"] = {"bgcolor": "rgba(0,0,0,0)"}
    grid = "rgba(255,255,255,0.07)"
    for ax in ("xaxis", "yaxis"):
        layout[ax] = {"gridcolor": grid, "zerolinecolor": "rgba(255,255,255,0.12)",
                      "linecolor": "rgba(255,255,255,0.18)"}
    pio.templates["mtnscot"] = t
    pio.templates.default = "mtnscot"


_register_template()


# --------------------------------------------------------------------------- #
# CSS — bento boxes + dark chrome on the fixed colour scheme
# --------------------------------------------------------------------------- #
_CSS = f"""
<style>
:root {{
  --bg:{BG}; --card:{CARD}; --card2:{CARD_HI};
  --red:{RED}; --teal:{TEAL}; --sand:{SAND};
  --text:{TEXT}; --muted:{MUTED}; --border:{BORDER};
}}
.stApp, [data-testid="stAppViewContainer"] {{ background:var(--bg); color:var(--text); }}
[data-testid="stHeader"] {{ background:rgba(17,18,20,0.6); backdrop-filter:blur(6px); }}
.block-container {{ padding-top:2rem; padding-bottom:3rem; max-width:1320px; }}

/* sidebar */
[data-testid="stSidebar"] {{ background:#16181C; border-right:1px solid var(--border); }}
[data-testid="stSidebarHeader"], [data-testid="stSidebar"] [data-testid="stLogo"] {{
  margin:0 auto; }}
[data-testid="stLogo"] {{ width:96px; height:auto; }}

/* headings */
h1 {{ color:var(--sand); font-weight:800; letter-spacing:-0.5px; }}
h2 {{ color:var(--text); }}
[data-testid="stHeading"] h3, h3 {{ color:#5FC7D1; font-weight:700; }}

/* BENTO: charts, tables and bordered containers become rounded cards */
[data-testid="stPlotlyChart"],
[data-testid="stDataFrame"],
[data-testid="stTable"] {{
  background:var(--card); border:1px solid var(--border); border-radius:16px;
  padding:10px 12px; box-shadow:0 8px 24px rgba(0,0,0,0.35);
}}
[data-testid="stVerticalBlockBorderWrapper"] {{
  background:var(--card); border:1px solid var(--border) !important;
  border-radius:18px; padding:1.1rem 1.25rem;
  box-shadow:0 8px 24px rgba(0,0,0,0.35);
}}

/* metric chips */
[data-testid="stMetric"] {{
  background:var(--card2); border-radius:14px; padding:0.7rem 0.95rem;
  border-left:3px solid var(--teal);
}}
[data-testid="stMetricValue"] {{ color:var(--sand); }}
[data-testid="stMetricLabel"] p {{ color:var(--muted); }}

/* layman "in plain English" callout */
.layman {{
  background:linear-gradient(90deg, rgba(0,116,128,0.16), rgba(209,180,156,0.06));
  border-left:4px solid var(--teal); border-radius:12px;
  padding:0.75rem 1.05rem; margin:0.1rem 0 1.1rem 0; color:#DAD7D2;
  font-size:0.98rem; line-height:1.45;
}}
.layman b {{ color:var(--sand); }}

/* tabs */
button[data-baseweb="tab"] [data-testid="stMarkdownContainer"] p {{ color:var(--muted); }}
button[data-baseweb="tab"][aria-selected="true"] [data-testid="stMarkdownContainer"] p {{
  color:#5FC7D1; }}
[data-baseweb="tab-highlight"] {{ background:var(--teal) !important; }}

/* misc */
hr {{ border-color:rgba(255,255,255,0.08); }}
a {{ color:#5FC7D1; }}
[data-testid="stSidebar"] .stRadio label p {{ color:var(--text); font-weight:500; }}
</style>
"""


def apply_chrome() -> None:
    """Pin the sidebar logo and inject the brand CSS. Call once per page run,
    before rendering content."""
    if LOGO.exists():
        st.logo(str(LOGO), size="large", icon_image=str(LOGO))
    st.markdown(_CSS, unsafe_allow_html=True)
