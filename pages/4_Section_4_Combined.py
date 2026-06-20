from src.section4 import ANALYSES
from src.ui import section_page

section_page(
    "Section 4 — All Networks Combined",
    "Section 4 — All Four Networks Combined (#26–32)",
    "Finance aligned to movement by reversing the finance layer (both run sell→buy). "
    "P1 links corresponding edges, P2 does club→league rollups. Heavy metrics use "
    "igraph + leidenalg/infomap on top-N subgraphs where needed (labelled estimates); "
    "community detection is club-level only.",
    ANALYSES, "s4_choice",
)
