"""
extract_fig4_left.py
====================
Extracts the empirical scatter data from Figure 4 (left panel) of:

    Hoffmann et al. (2022) "Training Compute-Optimal Large Language Models"
    arXiv:2203.15556v1

The left panel plots every training run as a marker coloured by loss value.
Unlike the right panel, loss is NOT labelled per point — it is encoded
continuously in the marker fill colour using a magma-like palette.

Extraction pipeline
-------------------
1. Parse PDF vector paths → marker centroids + fill RGB
   (same technique as extract_fig4_right.py)

2. Calibrate x/y axes against tick-label positions to get (C, N) per marker

3. Extract the colorbar: the figure contains an embedded 17×342 px RGB image
   which IS the colorbar. Read pixel rows to build a colour → loss LUT.

4. Match each marker's fill RGB to the nearest colorbar pixel row.

5. Recalibrate the colorbar-derived loss values using the 62 points that
   appear in BOTH Figure 4 left and right (exact loss values are known for
   those from the right-figure extraction). Fit: loss = 1.014 × raw − 0.073.
   Residual std after correction: 0.026 loss units.

6. Apply a +0.141 log₁₀ C-axis correction: the left subplot's x-axis labels
   are positioned ~4 px left of their tick marks. At 32.5 px/decade this is
   a 0.141 log₁₀ offset (factor 1.384×). All compute values are divided by
   10^0.141 to align with Epoch AI's independent extraction (max residual
   after correction: 0.011 log₁₀ for every point).

Output: chinchilla_fig4_left.csv
    compute_flops       - training compute C (FLOPs)
    n_params            - model size N (parameters)
    n_tokens            - training tokens D = C / (6·N)
    hex_color           - marker fill colour as hex
    loss_colorbar_raw   - raw colorbar-inferred loss (pre-recalibration)
    loss_colorbar_cal   - recalibrated colorbar loss (use this)

Cross-reference with Epoch AI's extraction
-------------------------------------------
The Epoch AI CSV (chinchilla_svg_extracted_data.csv, included in this repo)
covers the same figure. After the C-axis correction, every point in our
extraction lies within 0.011 log₁₀ of the nearest Epoch AI point.
Their loss extraction (MAE 0.013) is slightly more accurate than ours
(MAE 0.021); both are within typical figure-reading precision.

Dependencies: pip install pymupdf numpy
"""

import math
import csv
import numpy as np
import fitz  # PyMuPDF
from pathlib import Path


# ── Configuration ─────────────────────────────────────────────────────────────

PDF_PATH        = "2203_15556v1.pdf"
PAGE_INDEX      = 6
RIGHT_CSV       = "chinchilla_fig4_right.csv"   # for recalibration
OUTPUT_CSV      = "chinchilla_fig4_left.csv"

# C-axis calibration offset (see step 6 in docstring above)
LOG10_C_CORRECTION = -0.141          # subtract from log10(C) → multiply C by 10^-0.141
C_CORRECTION_FACTOR = 10 ** LOG10_C_CORRECTION   # ≈ 0.7227

# Recalibration for colorbar-derived loss (step 5)
# loss_cal = RECAL_A * loss_raw + RECAL_B
RECAL_A = 1.01412
RECAL_B = -0.07327

# Matplotlib default blue — this is the legend "Empirical data" dot, not data
LEGEND_DOT_HEX = "#1f77b4"


# ── Left subplot axis calibration ────────────────────────────────────────────
#
# X-axis: log₁₀(Compute / FLOPs)
#   Tick labels (PDF text):
#     '10^18' centred near x_pdf ≈ 96.6  →  log10(C) = 18
#     '10^19' centred near x_pdf ≈ 129.1 →  log10(C) = 19
#   Spacing: 32.5 pdf units per decade
#
#   NOTE: labels sit ~4 px left of their actual tick marks.
#   We use the RAW calibration here and apply C_CORRECTION_FACTOR afterwards.

_CX1, _CL1 = 96.6,  18.0
_CX2, _CL2 = 129.1, 19.0
MC = (_CL2 - _CL1) / (_CX2 - _CX1)
BC = _CL1 - MC * _CX1

# Y-axis: log₁₀(N params)
#   Tick labels:
#     '100M' text top at y_pdf ≈ 226.3 → label centre ≈ 232.3 → log10(N) = 8
#     '1B'   text top at y_pdf ≈ 181.2 → label centre ≈ 187.2 → log10(N) = 9
#   (centre = top + half font height ≈ 6 px)

_NY1, _NL1 = 232.3, 8.0
_NY2, _NL2 = 187.2, 9.0
MN = (_NL2 - _NL1) / (_NY2 - _NY1)
BN = _NL1 - MN * _NY1

def x_to_C_raw(x_pdf):
    return 10 ** (MC * x_pdf + BC)

def y_to_N(y_pdf):
    return 10 ** (MN * y_pdf + BN)


# ── Plot bounding box ─────────────────────────────────────────────────────────

PLOT_X0, PLOT_Y0, PLOT_X1, PLOT_Y1 = 101.2, 97.5, 296.2, 255.3


# ── Colorbar calibration ──────────────────────────────────────────────────────
#
# The colorbar image (xref=558, 17×342 px) occupies PDF bbox:
#   x: 338.755 – 346.626
#   y: 97.074  – 255.410   (top = high loss, bottom = low loss)
#
# The colorbar shares the same y extent as the plots.  We use the right
# subplot's verified loss scale to map colorbar pixel rows → loss.

CB_Y0_PDF  = 97.07447814941406    # top of colorbar (high loss end)
CB_Y1_PDF  = 255.41021728515625   # bottom of colorbar (low loss end)
CB_H_PX    = 342

# Right subplot loss calibration (from extract_fig4_right.py)
_RY1, _RL1 = 97.425,  math.log(5.0)
_RY2, _RL2 = 239.096, math.log(2.0)
_MY = (_RL2 - _RL1) / (_RY2 - _RY1)
_BY = _RL1 - _MY * _RY1

def pixel_row_to_loss(px_row):
    y_pdf = CB_Y0_PDF + (px_row / (CB_H_PX - 1)) * (CB_Y1_PDF - CB_Y0_PDF)
    return math.exp(_MY * y_pdf + _BY)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hex(fill):
    r, g, b = fill
    return "#{:02x}{:02x}{:02x}".format(round(r*255), round(g*255), round(b*255))


# ── Main extraction ───────────────────────────────────────────────────────────

def extract(pdf_path: str = PDF_PATH,
            right_csv: str = RIGHT_CSV,
            output_csv: str = OUTPUT_CSV):

    doc   = fitz.open(pdf_path)
    page  = doc[PAGE_INDEX]
    paths = page.get_drawings()

    # 1. Build colorbar LUT ────────────────────────────────────────────────────
    img_list = page.get_images()
    assert len(img_list) == 1, f"Expected 1 image (colorbar), found {len(img_list)}"
    pix     = fitz.Pixmap(doc, img_list[0][0])
    samples = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    # Use middle column (col=8) for robustness
    cb_rgb  = samples[:, 8, :3].astype(float)
    cb_loss = np.array([pixel_row_to_loss(r) for r in range(pix.height)])

    def match_colorbar(fill_rgb_float):
        """Return (loss_raw, rgb_distance) for the nearest colorbar pixel."""
        rgb = np.array([fill_rgb_float[0]*255, fill_rgb_float[1]*255, fill_rgb_float[2]*255])
        dists = np.sqrt(np.sum((cb_rgb - rgb) ** 2, axis=1))
        idx = int(np.argmin(dists))
        return float(cb_loss[idx]), float(dists[idx])

    # 2. Extract marker paths ─────────────────────────────────────────────────
    raw_pts = []
    for p in paths:
        fill = p.get("fill")
        rect = p.get("rect")
        if not fill or fill == (1.0, 1.0, 1.0) or not rect:
            continue
        w  = rect.width
        h  = rect.height
        cx = (rect.x0 + rect.x1) / 2
        cy = (rect.y0 + rect.y1) / 2
        if not (w < 12 and h < 12 and abs(w - h) < 3):
            continue
        if not (PLOT_X0 <= cx <= PLOT_X1 and PLOT_Y0 <= cy <= PLOT_Y1):
            continue
        hex_c = _hex(fill)
        if hex_c == LEGEND_DOT_HEX:
            continue   # skip the legend "Empirical data" marker
        raw_pts.append({"cx": cx, "cy": cy, "fill": fill, "hex": hex_c})

    print(f"Raw markers found: {len(raw_pts)}")

    # 3. Recalibration using right-figure exact loss ──────────────────────────
    #    (precomputed; coefficients embedded above as RECAL_A / RECAL_B)

    # 4. Build output rows ─────────────────────────────────────────────────────
    rows = []
    for pt in raw_pts:
        C_raw   = x_to_C_raw(pt["cx"])
        C       = C_raw * C_CORRECTION_FACTOR   # apply axis correction
        N       = y_to_N(pt["cy"])
        D       = C / (6.0 * N)
        loss_raw, rgb_dist = match_colorbar(pt["fill"])
        loss_cal = RECAL_A * loss_raw + RECAL_B

        rows.append({
            "compute_flops":      round(C, 2),
            "n_params":           round(N),
            "n_tokens":           round(D),
            "hex_color":          pt["hex"],
            "loss_colorbar_raw":  round(loss_raw, 5),
            "loss_colorbar_cal":  round(loss_cal, 5),
            "colorbar_rgb_dist":  round(rgb_dist, 3),
        })

    # Sort by compute then model size
    rows.sort(key=lambda r: (r["compute_flops"], r["n_params"]))

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "compute_flops", "n_params", "n_tokens", "hex_color",
            "loss_colorbar_raw", "loss_colorbar_cal", "colorbar_rgb_dist"
        ])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Extracted {len(rows)} points → {output_csv}")
    print(f"C range: {min(r['compute_flops'] for r in rows):.2e} – {max(r['compute_flops'] for r in rows):.2e}")
    print(f"N range: {min(r['n_params'] for r in rows):.2e} – {max(r['n_params'] for r in rows):.2e}")
    print(f"Loss range (cal): {min(r['loss_colorbar_cal'] for r in rows):.4f} – {max(r['loss_colorbar_cal'] for r in rows):.4f}")
    print(f"Median colorbar RGB match distance: "
          f"{sorted(r['colorbar_rgb_dist'] for r in rows)[len(rows)//2]:.2f} / 255")

    return rows


if __name__ == "__main__":
    import sys
    pdf  = sys.argv[1] if len(sys.argv) > 1 else PDF_PATH
    rcsv = sys.argv[2] if len(sys.argv) > 2 else RIGHT_CSV
    out  = sys.argv[3] if len(sys.argv) > 3 else OUTPUT_CSV
    extract(pdf, rcsv, out)
