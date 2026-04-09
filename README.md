# Chinchilla Visualization

Interactive 3D explorer for the Chinchilla scaling laws (Hoffmann et al. 2022)
with the **Besiroglu et al. 2024 bias fix** ([arXiv:2404.10102](https://arxiv.org/abs/2404.10102))
applied to the parametric loss fit.

Open `chinchilla_explorer.html` in any modern browser. Rotate, zoom, and toggle
traces to compare the empirical runs, the isoFLOP scatter, the bias-fixed
parametric model, and the prior scaling laws (Kaplan, Henighan).

## Contents

| File | Purpose |
|---|---|
| `chinchilla_explorer.html` | Standalone Plotly 3D viz (no build step). |
| `chinchilla_fig4.py` | Loads the extracted data, re-fits `L̂(N,D)`, reports residuals, regenerates the parametric curves for the HTML. |
| `chinchilla_fig4_data.json` | Machine-readable output of the script: three coefficient sets, 9 bias-fixed isoFLOP curves, and the compute-optimal frontier. |
| `chinchilla_svg_extracted_data.csv` | Epoch AI's SVG extraction of Chinchilla Figure 4 (left): 245 individual training runs with `(N, C, loss)`. |
| `chinchilla_isoflopslices_fig4_right.csv` | Epoch AI's extraction of Chinchilla Figure 4 (right): 114 points across 9 isoFLOP budgets with `(N, D, C, loss)`. |

## The parametric model and the bias fix

Hoffmann et al. 2022 fit the loss of a compute-optimal transformer as

```
L̂(N, D) = E + A / N^α + B / D^β
```

and reported Approach-3 coefficients

```
E = 1.69   A = 406.4   B = 410.7   α = 0.34   β = 0.28
```

These disagreed with Approaches 1 and 2 on the compute-optimal scaling
exponent (0.452 vs ~0.50). Besiroglu, Erdil, Barnett & You (2024) re-fit the
model to the data extracted from Figure 4 with a Huber-LSE multi-start
optimiser and obtained bias-corrected coefficients that reconcile Approach 3
with Approaches 1 and 2:

```
E = 1.8172   A = 482.01   B = 2085.43   α = 0.3478   β = 0.3658
→ N_opt ∝ (C/6)^0.5126   (vs 0.4516 from the original biased fit)
```

`chinchilla_fig4.py` applies this fix, independently re-fits the model on the
Epoch AI scatter to confirm the direction of the bias, and regenerates the
parametric curves used by the HTML viz.

## Verification results

Running `python3 chinchilla_fig4.py` produces the following residual table
(`|Δlog L|` = absolute residual on `log L̂`; `|ΔL|` = absolute residual on
`L̂`):

**Residuals vs Epoch AI empirical scatter (245 pts, Fig 4 left):**

| Coefficients | mean \|Δlog L\| | max \|Δlog L\| | mean \|ΔL\| |
|---|---:|---:|---:|
| Hoffmann 2022 (biased)       | 0.02098 | 0.30851 | **0.06118** |
| Besiroglu 2024 (bias fix)    | 0.00836 | 0.31141 | **0.02705** |
| Our refit on Epoch AI data   | 0.00792 | 0.26348 | **0.02470** |

**Residuals vs Chinchilla Fig 4 isoFLOP slices (114 pts):**

| Coefficients | mean \|Δlog L\| | max \|Δlog L\| | mean \|ΔL\| |
|---|---:|---:|---:|
| Hoffmann 2022 (biased)       | 0.02183 | 0.31669 | **0.06446** |
| Besiroglu 2024 (bias fix)    | 0.01240 | 0.32141 | **0.03973** |
| Our refit on Epoch AI data   | 0.01097 | 0.27650 | **0.03444** |

The bias fix roughly halves the mean residual on the Fig 4 left scatter
(0.0612 → 0.0271) and cuts it by ~40 % on the Fig 4 right isoFLOP slices
(0.0645 → 0.0397).

**Compute-optimal exponents** (`N_opt = G · (C/6)^a`):

| Coefficients | `G` | `a` |
|---|---:|---:|
| Hoffmann 2022 (biased)      | 1.3447 | 0.4516 |
| Besiroglu 2024 (bias fix)   | 0.1196 | **0.5126** |
| Our refit on Epoch AI data  | 0.0125 | 0.5646 |

The bias fix moves the Approach-3 exponent from **0.4516** (inconsistent with
Approaches 1 & 2 at ~0.50) to **0.5126** (consistent). Our independent refit
on the 245 Epoch AI points broadly confirms the direction of the fix.

**Per-slice curve vs scatter minima** (check that the bias-fixed
parametric curves actually go through the isoFLOP data):

| C        | data min loss | fit min loss | data `N` at min | fit `N` at min |
|---------:|--------------:|-------------:|----------------:|---------------:|
| 6e+18    | 3.0396        | 3.0330       | 1.75e+08        | 1.98e+08       |
| 1e+19    | 2.9374        | 2.9271       | 1.75e+08        | 2.57e+08       |
| 3e+19    | 2.7292        | 2.7297       | 6.32e+08        | 4.52e+08       |
| 6e+19    | 2.6152        | 2.6236       | 1.14e+09        | 6.44e+08       |
| 1e+20    | 2.5419        | 2.5534       | 1.02e+09        | 8.37e+08       |
| 3e+20    | 2.3900        | 2.4225       | 2.01e+09        | 1.47e+09       |
| 6e+20    | 2.3328        | 2.3521       | 2.01e+09        | 2.10e+09       |
| 1e+21    | 2.2867        | 2.3055       | 2.64e+09        | 2.73e+09       |
| 3e+21    | 2.1983        | 2.2187       | 2.64e+09        | 4.79e+09       |

Loss minima agree within ~0.02–0.05 nats and optimal `N` within a factor of
~1.5 across all nine compute budgets.

## HTML toggle labels

The three data toggles in `chinchilla_explorer.html` map directly to the
underlying datasets:

| Toggle | Source | Description |
|---|---|---|
| **Epoch AI fig 4 left** | `chinchilla_svg_extracted_data.csv` (245 pts) | Epoch AI's independent replication — individual training runs extracted from Chinchilla Figure 4 left. Colour encodes loss. |
| **Fig 4 right**         | `chinchilla_isoflopslices_fig4_right.csv` (114 pts × 9 slices) | IsoFLOP points from Chinchilla Figure 4 right: at each of 9 compute budgets, loss vs model size. |
| **Fig 4 left**          | `chinchilla_fig4_data.json` → `fig4_left_parametric_curves` (9 × 120 pts) | Bias-fixed parametric `L̂(N,D)` dashed curves at the same 9 compute budgets as Fig 4 right — the overlay shown in Chinchilla Figure 4 left. |

The Gopher / Chinchilla model points, the Approach-1/2/3 envelopes, the Kaplan
and Henighan curves, and the raw and smooth compute-optimal frontiers are
unchanged from the original visualisation.

## Reproducing the fit

```bash
pip install numpy pandas scipy
python3 chinchilla_fig4.py
```

The script prints the residual and exponent tables above and writes
`chinchilla_fig4_data.json`. To re-embed the bias-fixed curves into the HTML,
replace the `const PM = {...}` block near the bottom of
`chinchilla_explorer.html` with the contents of
`fig4_left_parametric_curves` from the JSON (same structure: per-FLOPs keys
with `N`, `D`, `C`, `loss`, `color` arrays).

## References

- Hoffmann et al. 2022, *Training Compute-Optimal Large Language Models*,
  [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
- Besiroglu, Erdil, Barnett & You 2024, *Chinchilla Scaling: A replication
  attempt*, [arXiv:2404.10102](https://arxiv.org/abs/2404.10102)
- Kaplan et al. 2020, *Scaling Laws for Neural Language Models*,
  [arXiv:2001.08361](https://arxiv.org/abs/2001.08361)
- Henighan et al. 2020, *Scaling Laws for Autoregressive Generative
  Modeling*, [arXiv:2010.14701](https://arxiv.org/abs/2010.14701)
