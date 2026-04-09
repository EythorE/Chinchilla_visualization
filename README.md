# Chinchilla Scaling Laws — Interactive 3D Explorer

An interactive 3D visualisation of the scaling law data from the landmark paper:

> **Training Compute-Optimal Large Language Models**  
> Hoffmann, Borgeaud, Mensch et al. (DeepMind, 2022) — [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)

**[Live demo →](https://eythore.github.io/Chinchilla_visualization/)**

---

## What this visualisation shows

The central question of the Chinchilla paper is: *given a fixed compute budget, how should you split it between model size (N parameters) and training data (D tokens)?*

The paper's answer — that N and D should scale equally — overturned the prevailing wisdom set by [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361), which recommended growing model size much faster than data. This single insight explains why models like GPT-3 and Gopher were dramatically undertrained relative to their compute budgets.

This explorer makes that argument tangible and rotatable. You can:

- Plot any two of **{Compute, Model Size, Tokens, Loss}** on the X and Y axes, and the third (or fourth) on Z
- Watch the isoFLOP U-curves form the loss surface
- See five competing scaling-law predictions fan out towards the Gopher/Chinchilla compute budget and observe how dramatically they diverge
- Toggle the paper's own parametric loss model and compare it against the raw experimental data

---

## The data

### What is in this repo

| File | Points | Source | Loss accuracy |
|---|---|---|---|
| `chinchilla_isoflopslices_fig4_right.csv` | 114 | Extracted from Fig 4 right (PDF vectors) | **Exact** — read from isoFLOP labels |
| `chinchilla_fig4_left_with_loss.csv` | 245 | Extracted from Fig 4 left (PDF vectors + colorbar) | ~0.021 MAE |
| `chinchilla_svg_extracted_data.csv` | 245 | [Epoch AI replication study](https://epochai.org/blog/chinchilla-replication) | ~0.013 MAE |

### Figure 4 right — isoFLOP slices (exact)

The right panel of Figure 4 shows 9 isoFLOP curves: for each of 9 fixed compute budgets (6×10¹⁸ to 3×10²¹ FLOPs), model size N is varied and the final training loss is recorded. This produces **9 U-shaped curves** — the minimum of each curve is the compute-optimal model size for that budget.

These points were extracted directly from the **PDF vector graphics content stream** using PyMuPDF (`extract_fig4_right.py`). Because the points are stored as vector paths, we can read the exact centroid and fill colour of every marker with zero rasterisation error. The 9 isoFLOP curves are distinguished by colour; the colour→FLOPs mapping is read from the legend line paths (also vector objects), so the assignment is unambiguous.

Loss values are exact because the y-axis of this subplot is a calibrated log-scale. We anchored the calibration on four horizontal gridlines at y_pdf positions corresponding to loss values 2.00, 3.00, 4.00, 5.00 — verified to < 0.001 log units.

### Figure 4 left — empirical scatter (loss from colorbar)

The left panel shows every training run as a dot coloured continuously by loss. Extracting this data is harder because loss is not directly labelled on each point.

**Pipeline (`extract_fig4_left.py`):**

1. **Vector parsing** — same technique as the right panel; extract marker centroids and fill RGB from the PDF content stream
2. **Axis calibration** — x-axis (Compute, log scale) and y-axis (Model Size, log scale) are calibrated from tick-label positions. A systematic +0.141 log₁₀ offset in the compute axis was corrected; it arose because axis tick labels sit ~4 px left of their actual tick marks, and at 32.5 px/decade this shifts inferred FLOPs by a factor of 1.38×
3. **Colorbar extraction** — the figure contains an embedded 17×342 px RGB image that is the colorbar. PyMuPDF extracts this as raw pixel data. Each of the 342 pixel rows maps to a loss value via the same log-loss calibration used for the right subplot (the colorbar spans the same y range as both plots)
4. **Colour matching** — each marker's fill RGB is matched to the nearest colorbar pixel row by Euclidean distance in RGB space (median distance: 0.46/255 — essentially perfect)
5. **Recalibration** — the raw colorbar mapping had a +0.036 loss bias. We corrected it using the 62 points present in both figures (where exact loss is known from the right panel). Final fit: `loss = 1.014 × raw − 0.073`, residual std = 0.026

### Epoch AI replication

[Epoch AI](https://epochai.org) independently extracted the same left-panel figure as part of their [Chinchilla replication study](https://epochai.org/blog/chinchilla-replication). Their CSV is included in this repo. After our C-axis correction, our coordinates match theirs within 0.011 log₁₀ for every single point. Their loss extraction (MAE 0.013) is slightly more accurate than ours (MAE 0.021).

---

## The scaling laws

The explorer includes five compute-optimal scaling law curves, all of the form **N_opt ∝ C^a**:

| Curve | Exponent *a* | Method | Prediction at Gopher budget |
|---|---|---|---|
| **Approach 1** (Chinchilla) | 0.50 | Train models at fixed sizes, 4 schedule lengths each; extract min-loss envelope | N ≈ **69B** |
| **Approach 2** (Chinchilla) | 0.49 | Fit parabola to each isoFLOP U-curve; find analytic minimum | N ≈ **64B** |
| **Approach 3** (Chinchilla) | 0.45 | Minimise parametric loss model L̂(N,D) analytically | N ≈ **40B** |
| **Kaplan et al. (2020)** | 0.73 | Training curve extrapolation | N ≈ **491B** |
| **Henighan et al. (2020)** | 0.73 | Validates Kaplan for language; different coefficient | N ≈ **154B** |

All three Chinchilla approaches agree that at Gopher's compute budget the optimal model is roughly 4–7× smaller than Gopher (280B). Approach 1 predicts N = 69.3B — almost exactly Chinchilla's actual 70B. The Kaplan prediction (491B) is 7× over-sized relative to the compute budget.

### The parametric loss model

The paper's Approach 3 fits all training runs to a closed-form decomposition (Equation 10):

```
L̂(N, D) = E + A/N^α + B/D^β
```

with fitted values: **E = 1.69, A = 406.4, B = 410.7, α = 0.34, β = 0.28**

The three terms capture:
- **E** — irreducible entropy of natural text (lower bound on loss)
- **A/N^α** — approximation error that shrinks as the model gets larger
- **B/D^β** — optimisation sub-optimality that shrinks as more data is seen

Toggling "Parametric model L̂(N,D)" in the explorer draws smooth isoFLOP curves from this formula, overlaid on the raw scatter — letting you see where the fit over- or under-predicts.

### The efficient frontier

The explorer also shows two versions of the compute-optimal ridge (the locus of optimal (N, D) pairs):

- **Raw frontier** (grey dotted) — the discrete minimum-loss point from each of the 9 isoFLOP slices. This is jagged because multiple slices share the same tested N values (the model-size grid is coarse and identical across all slices), and the 3×10²¹ slice's true optimum lies below the smallest model tested.
- **Smooth frontier** (blue) — a power-law fit through those 9 minima: N ∝ C^0.475. Close to but not identical to Approach 2 (0.49), because we fit to discrete grid minima rather than parabola-interpolated minima.

---

## Files

```
Chinchilla_visualization/
├── index.html                                  ← The self-contained interactive explorer
├── extract_fig4_right.py                       ← Extracts Fig 4 right (isoFLOP slices)
├── extract_fig4_left.py                        ← Extracts Fig 4 left (empirical scatter + colorbar)
├── chinchilla_isoflopslices_fig4_right.csv     ← 114 pts, exact loss, 9 isoFLOP curves
├── chinchilla_fig4_left_with_loss.csv          ← 245 pts, colorbar-inferred loss
├── chinchilla_svg_extracted_data.csv           ← Epoch AI independent extraction (245 pts)
├── chinchilla_guide.md                         ← Background notes on the Chinchilla paper
└── README.md
```

> **Note:** The paper PDF (`2203_15556v1.pdf`) is not included in this repo. Download it from [arXiv](https://arxiv.org/abs/2203.15556) and place it in the same directory before running the extraction scripts.

---

## Running the extraction scripts

```bash
pip install pymupdf numpy

# Extract Figure 4 right (isoFLOP slices — exact loss)
python extract_fig4_right.py 2203_15556v1.pdf chinchilla_isoflopslices_fig4_right.csv

# Extract Figure 4 left (empirical scatter — colorbar-inferred loss)
python extract_fig4_left.py 2203_15556v1.pdf chinchilla_isoflopslices_fig4_right.csv chinchilla_fig4_left_with_loss.csv
```

The left-figure script accepts the right-figure CSV as a second argument so that the exact loss values from the right panel can be used to recalibrate the colorbar mapping (removing the small bias inherent in colour→loss matching). Note that the currently committed script uses precomputed recalibration coefficients (`RECAL_A`, `RECAL_B`) rather than recomputing them from the right CSV at runtime — see the bug report for details.

---

## Publishing to GitHub Pages

The explorer is a single self-contained HTML file with all data embedded and Plotly loaded from CDN — no build step, no server required.

1. Fork or clone this repo
2. The entry point is already named `index.html`, so no renaming is required
3. Go to **Settings → Pages → Source → Deploy from branch → `main` / `(root)`**
4. Your explorer is live at `https://<username>.github.io/<reponame>/`

---

## Key takeaways from the visualisation

**Set X = Compute, Y = Model Size, Z = Loss** and rotate until you're looking at the XY plane from slightly above. You will see:

- The experimental points form a **loss surface** that descends as compute increases
- The **Approach 1/2/3 curves** (green/yellow/orange) cut diagonally through this surface, tracking the minimum-loss ridge — they all land near Chinchilla's 70B
- The **Kaplan curve** (red) angles steeply upward and lands near Gopher's 280B — 4× too large
- **Gopher** (orange diamond) sits visibly *higher on the loss surface* than **Chinchilla** (cyan diamond) despite identical compute — Gopher is undertrained

**Switch to X = Compute, Y = Tokens, Z = Loss** to see the complementary view: Gopher trained on only 300B tokens while Chinchilla trained on 1.4T, and Chinchilla's lower loss makes this tradeoff concrete.

---

## Citation

```bibtex
@article{hoffmann2022chinchilla,
  title   = {Training Compute-Optimal Large Language Models},
  author  = {Hoffmann, Jordan and Borgeaud, Sebastian and Mensch, Arthur and
             Buchatskaya, Elena and Cai, Trevor and Rutherford, Eliza and
             de Las Casas, Diego and Hendricks, Lisa Anne and Welbl, Johannes and
             Clark, Aidan and others},
  journal = {arXiv preprint arXiv:2203.15556},
  year    = {2022}
}
```

---

## Acknowledgements

- Data extracted from Hoffmann et al. (2022) using [PyMuPDF](https://pymupdf.readthedocs.io/)
- Independent replication data from [Epoch AI](https://epochai.org/blog/chinchilla-replication)
- Visualisation built with [Plotly.js](https://plotly.com/javascript/)
- Prior scaling law references: [Kaplan et al. (2020)](https://arxiv.org/abs/2001.08361), [Henighan et al. (2020)](https://arxiv.org/abs/2010.14701)
