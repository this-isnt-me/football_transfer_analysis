"""Shared UI + plotting helpers for the section pages.

Single source for the recurring Streamlit controls (grain radio, top-X slider,
Outside-System exclusion) and the recurring Plotly idioms (ranked horizontal
bar, parity diagonal, top-N-by-total selection) that were previously copy-pasted
across ``section1``–``section4``. Each helper is a literal lift of the original
inline code, so rendered output is byte-for-byte unchanged; call sites that pass
custom labels/keys keep their exact text.

Pages read from the cached data/metrics layers, call a render fn, and render —
``section_page`` single-sources the multipage entry-file boilerplate.
"""
from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from . import metrics as M
from . import theme

# Brand colourways shared by every section (see src/theme.py). Brand-led, but
# extended to >=12 legible-on-dark hues so high-cardinality charts still
# differentiate. Importing theme also activates the dark Plotly template.
PALETTE = theme.COLORWAY
QUAL = theme.COLORWAY


# --------------------------------------------------------------------------- #
# Controls
# --------------------------------------------------------------------------- #
def grain_control(key: str, default: str = "club") -> str:
    """Club/league radio (horizontal). Key namespaced ``grain_<key>``."""
    return st.radio(
        "Grain", ["club", "league"], horizontal=True,
        index=0 if default == "club" else 1, key=f"grain_{key}",
    )


def topx(key: str, default: int = 20, lo: int = 5, hi: int = 40) -> int:
    """Top-X slider. Key namespaced ``topx_<key>``."""
    return st.slider("Top-X", lo, hi, default, key=f"topx_{key}")


def exclude_os(key: str, grain: str, default: bool = True, **_kw) -> bool:
    """Non-club nodes (OS1, Without Club, UnknownUnknown) are now removed from
    every metric *upstream* in the data layer (see ``metrics.club_edges``), so
    excluding them is no longer a user choice. Retained as a no-op — renders
    nothing — for call-site compatibility; returns True at club grain so any
    residual post-filter stays a harmless no-op."""
    return grain == "club"


def maybe_drop_os(df, grain, exclude, id_col="node"):
    """Compatibility shim: the catch-all nodes are already gone (filtered
    upstream), so this just returns the frame unchanged plus an empty ``os_row``
    (the old Outside-System reference row no longer applies)."""
    if grain == "club" and exclude:
        return M.drop_non_clubs(df, id_col), df.iloc[0:0]
    return df, df.iloc[0:0]


# --------------------------------------------------------------------------- #
# Plotting idioms
# --------------------------------------------------------------------------- #
def ranked_bar(df, x, y, title, color=None, orientation="h", height=480, hover_data=None):
    """Ranked bar with the shared styling (PALETTE, reversed y for horizontal)."""
    fig = px.bar(df, x=x, y=y, title=title, color=color,
                 color_discrete_sequence=PALETTE, orientation=orientation,
                 hover_data=hover_data)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=height)
    if orientation == "h":
        fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")


def add_parity_line(fig, m):
    """Add a dashed grey ``y = x`` parity diagonal from 0 to ``m``."""
    fig.add_trace(go.Scatter(x=[0, m], y=[0, m], mode="lines",
                             line=dict(dash="dash", color="grey"), showlegend=False))


def top_nodes_by(df, value, n, *, group=("node", "label"), use_abs=False):
    """Filter ``df`` to the ``n`` nodes with the largest summed ``value``.

    Mirrors the repeated ``groupby(node,label).sum() -> nlargest -> isin`` select.
    ``use_abs`` ranks by magnitude (for signed metrics like net spend)."""
    totals = df.groupby(list(group), observed=True)[value].sum()
    if use_abs:
        totals = totals.abs()
    keep = totals.reset_index().nlargest(n, value)["node"]
    return df[df["node"].isin(keep)]


# --------------------------------------------------------------------------- #
# Sidebar edge-filters (apply to every analysis on the page)
# --------------------------------------------------------------------------- #
def sidebar_filters():
    """Render the season / window / position filters in the sidebar and stash the
    selection in ``st.session_state['edge_filter']``. Defaults select everything
    (a no-op = full graph). When the selection changes we clear the data caches
    so every cached metric recomputes on the newly sliced data."""
    lo0, hi0 = M.season_bounds()
    with st.sidebar:
        with st.expander("🔎 Filter the data", expanded=False):
            st.caption("Narrow the transfers feeding every chart on this page. "
                       "Defaults show everything.")
            yr = st.slider("Seasons (year)", lo0, hi0, (lo0, hi0), key="flt_season",
                           help="Keep only transfers from these seasons.")
            wins = st.multiselect("Transfer window", M.WINDOWS, default=M.WINDOWS,
                                  format_func=str.capitalize, key="flt_windows",
                                  help="Summer and/or winter windows.")
            poss = st.multiselect("Player position", M.POSITIONS, default=M.POSITIONS,
                                  key="flt_positions", help="Keep only these positions.")
            st.caption("Filters persist as you navigate; they do not affect the "
                       "Transfer Flow Maps.")
    st.session_state["edge_filter"] = {
        "season": yr, "season_full": (lo0, hi0), "windows": wins, "positions": poss,
    }
    key = (tuple(yr), tuple(sorted(wins)), tuple(sorted(poss)))
    if st.session_state.get("_filter_key") != key:
        st.cache_data.clear()                # selection changed -> recompute on sliced data
        st.session_state["_filter_key"] = key


# --------------------------------------------------------------------------- #
# Page boilerplate
# --------------------------------------------------------------------------- #
def section_page(page_title: str, title: str, caption: str, analyses: dict, key: str,
                 explain: dict | None = None):
    """Standard multipage entry: page config + brand chrome, title/caption,
    sidebar analysis picker + data filters, a plain-English callout, then dispatch
    to the chosen ``render_*`` function (guarded against empty/over-filtered data)."""
    icon = str(theme.LOGO) if theme.LOGO.exists() else "⚽"
    st.set_page_config(page_title=page_title, page_icon=icon, layout="wide")
    theme.apply_chrome()
    st.title(title)
    st.caption(caption)
    st.caption(
        "ℹ️ Non-club nodes — Outside System, free agency (“Without Club”) and "
        "unknown-status moves — are filtered out of every metric here, so the figures "
        "reflect genuine club-to-club activity. They stay visible in **Transfer Flow Maps**."
    )
    choice = st.sidebar.radio("Pick an analysis", list(analyses.keys()), key=key)
    sidebar_filters()
    st.divider()
    if explain and choice in explain:
        st.markdown(f"<div class='layman'><b>In plain English —</b> {explain[choice]}</div>",
                    unsafe_allow_html=True)

    # Safeguard 1: the filters can empty the graph entirely.
    if not M.any_edges():
        st.warning("**No transfers match the current filters.** Widen the year range, "
                   "window or position selection in the sidebar to see this analysis.",
                   icon="🔎")
        return
    # Safeguard 2: a very narrow slice can leave a graph too sparse/disconnected for
    # a particular algorithm. Catch it, log it for devs, and guide the user.
    try:
        analyses[choice]()
    except Exception:  # noqa: BLE001 - last-resort guard against filter-induced edge cases
        import traceback
        traceback.print_exc()
        st.warning("**This view couldn't be built for the current filter selection** — "
                   "there may be too few transfers (e.g. a single window and position in "
                   "one season). Try widening the filters in the sidebar.", icon="⚠️")
