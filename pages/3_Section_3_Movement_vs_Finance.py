from src.section3 import ANALYSES
from src.ui import section_page

section_page(
    "Section 3 — Movement vs Finance",
    "Section 3 — Cross-Network, Same Granularity: Movement vs Finance (#19–25)",
    "Movement and finance joined per deal via P1 (transfer_id). Movement runs "
    "sell→buy, finance buy→sell — the join aligns the flip. The ~22k unmatched "
    "(NULL-fee) moves are excluded from fee stats, never zeroed; fee stats use medians.",
    ANALYSES, "s3_choice",
)
