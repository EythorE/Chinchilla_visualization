"""
extract_fig4_right.py
=====================
Extracts the isoFLOP scatter data from Figure 4 (right panel) of:

    Hoffmann et al. (2022) "Training Compute-Optimal Large Language Models"
    arXiv:2203.15556v1

Method: direct PDF vector-graphics parsing via PyMuPDF (fitz).
No rasterisation, no model inference — every data point is read from the
exact centroid and fill colour of the scatter marker paths in the content stream.

Output: chinchilla_fig4_right.csv
    flops_label    - isoFLOP curve label (e.g. "6e+18")
    compute_flops  - exact FLOPs value for that curve
    hex_color      - marker fill colour as hex
    n_params       - model size N (parameters)
    loss           - training loss
    n_tokens       - training tokens D = C / (6·N)

Dependencies: pip install pymupdf
"""

import math
import csv
import fitz  # PyMuPDF


# ── Configuration ─────────────────────────────────────────────────────────────

PDF_PATH   = "2203_15556v1.pdf"
PAGE_INDEX = 6   # 0-indexed; Figure 4 is on page 7

OUTPUT_CSV = "chinchilla_isoflopslices_fig4_right.csv"


# ── Axis calibration (derived from PDF grid-line positions) ──────────────────
#
# X-axis: log₁₀(N params) — vertical grey gridlines at exact x positions
#   x=363.939 → N = 10^8  (100M)
#   x=402.354 → N = 10^9  (1B)
#   x=440.768 → N = 10^10 (10B)
#   x=463.708 → N = 10^{10.60} (40B)
#
# Y-axis: loss (log scale) — horizontal grey gridlines at exact y positions
#   y=97.425  → loss = 5.00
#   y=131.926 → loss = 4.00
#   y=176.406 → loss = 3.00
#   y=239.096 → loss = 2.00
#
# Both calibrations verified to < 0.001 log units against labelled tick values.

# X: log10(N) = mx * x_pdf + bx
_x1, _lx1 = 363.939, 8.0   # 100M
_x2, _lx2 = 402.354, 9.0   # 1B
MX = (_lx2 - _lx1) / (_x2 - _x1)
BX = _lx1 - MX * _x1

# Y: loss = exp(my * y_pdf + by)  [log-scale on loss axis]
_y1, _l1 = 97.425,  math.log(5.0)
_y2, _l2 = 239.096, math.log(2.0)
MY = (_l2 - _l1) / (_y2 - _y1)
BY = _l1 - MY * _y1

def x_to_N(x_pdf):
    return 10 ** (MX * x_pdf + BX)

def y_to_loss(y_pdf):
    return math.exp(MY * y_pdf + BY)


# ── IsoFLOP legend colour mapping ────────────────────────────────────────────
#
# The legend lines (paths 370–378) carry the exact same RGB floats as the
# scatter markers. Each is matched to its FLOPs label by vertical position.
#
# Extracted directly from the legend paths on page 7.

LEGEND_COLORS = [
    # (R, G, B as [0,1] floats)  →  (label string, FLOPs value)
    ((0.661296, 0.884140, 0.741307), "6e+18", 6e18),
    ((0.375982, 0.809506, 0.676414), "1e+19", 1e19),
    ((0.242426, 0.705222, 0.678131), "3e+19", 3e19),
    ((0.204319, 0.591408, 0.663002), "6e+19", 6e19),
    ((0.206927, 0.482018, 0.638127), "1e+20", 1e20),
    ((0.224140, 0.364873, 0.609949), "3e+20", 3e20),
    ((0.254061, 0.248816, 0.503489), "6e+20", 6e20),
    ((0.221363, 0.164719, 0.330414), "1e+21", 1e21),
    ((0.144525, 0.088242, 0.162625), "3e+21", 3e21),
]

def _rgb_dist(c1, c2):
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5

def match_flops(fill_rgb):
    """Return (label, flops_value) for the nearest legend colour."""
    best = min(LEGEND_COLORS, key=lambda lc: _rgb_dist(fill_rgb, lc[0]))
    return best[1], best[2]


# ── Plot bounding box ─────────────────────────────────────────────────────────

PLOT_X0, PLOT_Y0, PLOT_X1, PLOT_Y1 = 352.4, 97.4, 479.2, 255.4


# ── Main extraction ───────────────────────────────────────────────────────────

def extract(pdf_path: str = PDF_PATH, output_csv: str = OUTPUT_CSV):
    doc   = fitz.open(pdf_path)
    page  = doc[PAGE_INDEX]
    paths = page.get_drawings()

    rows = []
    for p in paths:
        fill = p.get("fill")
        rect = p.get("rect")

        # Skip: no fill, white fill, or no bounding rect
        if not fill or fill == (1.0, 1.0, 1.0) or not rect:
            continue

        w  = rect.width
        h  = rect.height
        cx = (rect.x0 + rect.x1) / 2
        cy = (rect.y0 + rect.y1) / 2

        # Scatter markers are small (~3–6 px), roughly square (circle glyph)
        if not (w < 12 and h < 12 and abs(w - h) < 3):
            continue

        # Must be inside the right subplot
        if not (PLOT_X0 <= cx <= PLOT_X1 and PLOT_Y0 <= cy <= PLOT_Y1):
            continue

        flop_label, flops = match_flops(fill)
        N    = x_to_N(cx)
        loss = y_to_loss(cy)
        D    = flops / (6.0 * N)   # C = 6·N·D  (Kaplan et al. approximation)

        hex_color = "#{:02x}{:02x}{:02x}".format(
            round(fill[0] * 255), round(fill[1] * 255), round(fill[2] * 255)
        )

        rows.append({
            "flops_label":   flop_label,
            "compute_flops": flops,
            "hex_color":     hex_color,
            "n_params":      round(N),
            "loss":          round(loss, 6),
            "n_tokens":      round(D),
        })

    # Sort by compute budget then model size
    rows.sort(key=lambda r: (r["compute_flops"], r["n_params"]))

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["flops_label", "compute_flops", "hex_color",
                           "n_params", "loss", "n_tokens"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Extracted {len(rows)} points → {output_csv}")

    # Summary
    from collections import Counter
    counts = Counter(r["flops_label"] for r in rows)
    for label in ["6e+18","1e+19","3e+19","6e+19","1e+20","3e+20","6e+20","1e+21","3e+21"]:
        print(f"  {label}: {counts[label]} points")

    return rows


if __name__ == "__main__":
    import sys
    pdf  = sys.argv[1] if len(sys.argv) > 1 else PDF_PATH
    out  = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_CSV
    extract(pdf, out)
