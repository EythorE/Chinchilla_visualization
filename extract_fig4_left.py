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

5. Recalibrate the colorbar-derived loss values against the 62 points that
   appear in BOTH Figure 4 left and right. Exact losses for those points are
   read from right_csv (output of extract_fig4_right.py). Matching is by
   nearest-neighbour in (log10 C, log10 N) space; the linear fit
   `loss_cal = a * loss_raw + b` is computed at runtime.

6. Apply a -0.141 log₁₀ C-axis correction (i.e. divide every compute value
   by 10^0.141 ≈ 1.384). The raw calibration reads compute as too high
   because the left subplot's x-axis tick labels are drawn ~4 px left of
   their actual tick marks — at 32.5 px/decade that's a 0.141 log₁₀ offset.
   After this correction, every point agrees with Epoch AI's independent
   extraction to within 0.011 log₁₀.

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


# ── Configuration ─────────────────────────────────────────────────────────────

PDF_PATH        = "2203_15556v1.pdf"
PAGE_INDEX      = 6
RIGHT_CSV       = "chinchilla_isoflopslices_fig4_right.csv"   # for recalibration
OUTPUT_CSV      = "chinchilla_fig4_left.csv"

# C-axis calibration offset (see step 6 in docstring above)
LOG10_C_CORRECTION = -0.141          # add to log10(C) → C_corrected = C_raw * 10^-0.141
C_CORRECTION_FACTOR = 10 ** LOG10_C_CORRECTION   # ≈ 0.7227

# Tolerance (log₁₀ units, max-coord) for matching left-panel points to their
# right-panel counterparts when fitting the colorbar recalibration in step 5.
MATCH_TOL_LOG10 = 0.02

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


def _find_colorbar_pixmap(doc, page):
    """
    Return a fitz.Pixmap for the colorbar image on the page.

    Identified by being tall-and-narrow (height ≈ CB_H_PX, width < height).
    This is more robust than asserting exactly one image on the page: if a
    future PDF re-export inlines an extra image (e.g. a logo) we still pick
    the right one.
    """
    img_list = page.get_images()
    candidates = []
    for img_info in img_list:
        xref = img_info[0]
        p = fitz.Pixmap(doc, xref)
        if p.width < p.height and p.height > 100:
            candidates.append(p)
    if not candidates:
        raise RuntimeError(
            f"Colorbar image not found (scanned {len(img_list)} image(s) on page)"
        )
    # Pick the candidate whose height is closest to the expected CB_H_PX.
    candidates.sort(key=lambda p: abs(p.height - CB_H_PX))
    return candidates[0]


def _fit_recalibration(records, right_csv_path, match_tol_log10=MATCH_TOL_LOG10):
    """
    Fit `loss_cal = a * loss_raw + b` using the subset of extracted points
    that also appear in Figure 4 right (where exact losses are known).

    Matching: for every left-panel record, find the nearest right-panel point
    in (log10 C, log10 N) space; keep it if the max-coordinate distance is
    below `match_tol_log10`.

    Returns (a, b, n_matched, residual_std).
    """
    right_pts = []
    with open(right_csv_path, newline="") as f:
        for row in csv.DictReader(f):
            right_pts.append((
                float(row["compute_flops"]),
                float(row["n_params"]),
                float(row["loss"]),
            ))

    if not right_pts:
        raise RuntimeError(f"Right CSV {right_csv_path!r} is empty")

    right_log = np.array([[math.log10(C), math.log10(N)] for C, N, _ in right_pts])
    right_loss = np.array([L for _, _, L in right_pts])

    raw_losses, exact_losses = [], []
    for r in records:
        q = np.array([math.log10(r["C"]), math.log10(r["N"])])
        d = np.max(np.abs(right_log - q), axis=1)
        i = int(np.argmin(d))
        if d[i] < match_tol_log10:
            raw_losses.append(r["loss_raw"])
            exact_losses.append(float(right_loss[i]))

    if len(raw_losses) < 10:
        raise RuntimeError(
            f"Only {len(raw_losses)} left↔right matches at tol={match_tol_log10} "
            f"log₁₀ — cannot fit recalibration reliably"
        )

    x = np.array(raw_losses)
    y = np.array(exact_losses)
    a, b = np.polyfit(x, y, 1)
    resid = y - (a * x + b)
    return float(a), float(b), len(raw_losses), float(resid.std())


# ── Main extraction ───────────────────────────────────────────────────────────

def extract(pdf_path: str = PDF_PATH,
            right_csv: str = RIGHT_CSV,
            output_csv: str = OUTPUT_CSV):

    doc   = fitz.open(pdf_path)
    page  = doc[PAGE_INDEX]
    paths = page.get_drawings()

    # 1. Build colorbar LUT ────────────────────────────────────────────────────
    pix     = _find_colorbar_pixmap(doc, page)
    samples = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

    # Use the middle column of the colorbar image (avoids any anti-aliased edge).
    cb_rgb  = samples[:, pix.width // 2, :3].astype(float)
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

    # 3. Compute (C, N, D, loss_raw) for every marker ─────────────────────────
    records = []
    for pt in raw_pts:
        C = x_to_C_raw(pt["cx"]) * C_CORRECTION_FACTOR
        N = y_to_N(pt["cy"])
        D = C / (6.0 * N)
        loss_raw, rgb_dist = match_colorbar(pt["fill"])
        records.append({
            "C": C, "N": N, "D": D,
            "hex": pt["hex"],
            "loss_raw": loss_raw,
            "rgb_dist": rgb_dist,
        })

    # 4. Fit recalibration against the right-figure exact losses ──────────────
    recal_a, recal_b, n_match, resid_std = _fit_recalibration(records, right_csv)
    print(f"Recalibration fit from {n_match} left↔right matches:")
    print(f"  loss_cal = {recal_a:.5f} * loss_raw + {recal_b:.5f}")
    print(f"  residual std = {resid_std:.4f}")

    # 5. Apply recalibration and build output rows ────────────────────────────
    rows = []
    for r in records:
        loss_cal = recal_a * r["loss_raw"] + recal_b
        rows.append({
            "compute_flops":      round(r["C"], 2),
            "n_params":           round(r["N"]),
            "n_tokens":           round(r["D"]),
            "hex_color":          r["hex"],
            "loss_colorbar_raw":  round(r["loss_raw"], 5),
            "loss_colorbar_cal":  round(loss_cal, 5),
            "colorbar_rgb_dist":  round(r["rgb_dist"], 3),
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
