from src.section1 import ANALYSES
from src.ui import section_page

section_page(
    "Section 1 — Single Network",
    "Section 1 — Single-Network Analyses (#1–15)",
    "Metrics on each network in isolation. Pick an analysis; controls for grain "
    "(club/league), top-X and Outside-System handling appear per analysis.",
    ANALYSES, "s1_choice",
)
