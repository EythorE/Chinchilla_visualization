"""
Chinchilla Figure 4 reproduction with the Besiroglu et al. (2024) bias fix.

Hoffmann et al. (2022) — "Training Compute-Optimal Large Language Models" —
proposed the parametric loss fit

        L̂(N, D) = E + A / N**alpha + B / D**beta

and reported Approach-3 coefficients
        E = 1.69, A = 406.4, B = 410.7, alpha = 0.34, beta = 0.28
which disagreed with Approaches 1 and 2 on the compute-optimal exponent
(0.45 vs ~0.50).  Besiroglu, Erdil, Barnett & You (2024) — "Chinchilla
Scaling: A replication attempt" (arXiv:2404.10102) — re-fit the model to
the data extracted from Chinchilla Figure 4 and obtained bias-corrected
coefficients that reconcile Approach 3 with Approaches 1 and 2:

        E ≈ 1.8172, A ≈ 482.01, B ≈ 2085.43,
        alpha ≈ 0.3478, beta ≈ 0.3658
        → N_opt ∝ C**0.513 (vs 0.452 from the original biased fit)

This script:
    (1) loads the Epoch AI empirical scatter (245 pts, Fig 4 left) and
        the isoFLOP slice points (114 pts, 9 slices, Fig 4 right);
    (2) re-fits L̂(N, D) from scratch with a Huber-loss LSE formulation
        (identical to Hoffmann et al. Eq. 5) using multiple restarts, so
        we can show the bias fix on real data, not just assume it;
    (3) reports residuals of the fit against both extracted datasets;
    (4) samples the parametric curves at the 9 isoFLOP compute budgets
        and the compute-optimal frontier N_opt(C), D_opt(C); and
    (5) writes chinchilla_fig4_data.json for embedding in the HTML.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
EPOCH_CSV   = HERE / "chinchilla_svg_extracted_data.csv"
ISOFLOP_CSV = HERE / "chinchilla_isoflopslices_fig4_right.csv"
OUT_JSON    = HERE / "chinchilla_fig4_data.json"


# ---------------------------------------------------------------------------
# Published coefficient sets
# ---------------------------------------------------------------------------

HOFFMANN_2022 = dict(E=1.69,   A=406.4,  B=410.7,  alpha=0.34,   beta=0.28)    # original (biased)
BESIROGLU_2024 = dict(E=1.8172, A=482.01, B=2085.43, alpha=0.3478, beta=0.3658)  # bias fix


def L_hat(N, D, *, E, A, B, alpha, beta):
    """Parametric loss L̂(N, D) = E + A/N^alpha + B/D^beta."""
    return E + A / np.power(N, alpha) + B / np.power(D, beta)


def compute_optimal(C, *, A, B, alpha, beta, **_):
    """Analytic minimum of L̂ subject to C = 6ND (Hoffmann et al. Sec 3.3).

    Returns (N_opt, D_opt, L_opt).
    """
    # Stationary condition: alpha*A/N^alpha = beta*B/D^beta, with D = C/(6N).
    G = ((alpha * A) / (beta * B)) ** (1.0 / (alpha + beta))
    a = beta  / (alpha + beta)
    b = alpha / (alpha + beta)
    N_opt = G       * (C / 6.0) ** a
    D_opt = (1.0/G) * (C / 6.0) ** b
    return N_opt, D_opt


# ---------------------------------------------------------------------------
# Huber-LSE fit (Hoffmann et al. Eq. 5; Besiroglu et al. Eq. 2)
# ---------------------------------------------------------------------------

def huber(x, delta=1e-3):
    ax = np.abs(x)
    return np.where(ax <= delta, 0.5 * x * x, delta * (ax - 0.5 * delta))


def fit_huber_lse(N, D, L, delta=1e-3, seed=0):
    """Fit parameters (a, b, e, alpha, beta) via Huber loss on
        log L̂ = logsumexp(a - alpha * log N, b - beta * log D, e)
    with L̂_pred = exp(log L̂), and report L̂-coefficients (E, A, B).
    Uses many restarts to escape local optima (this is precisely the
    step where Hoffmann et al.'s fit was shown to be biased)."""
    logN, logD, logL = np.log(N), np.log(D), np.log(L)

    def neg(params):
        a, b, e, alpha, beta = params
        log_Lhat = logsumexp(
            np.stack([a - alpha * logN, b - beta * logD, e * np.ones_like(logN)]),
            axis=0,
        )
        return huber(log_Lhat - logL, delta).sum()

    # Initialisation grid in the spirit of Besiroglu et al. (2024) §3.
    # We use a coarser grid than their 4500-start sweep — a handful of
    # decently-spaced starts is plenty to locate the global Huber basin
    # because the LSE log-parameterisation is well-behaved.
    a_grid     = [0.0, 10.0, 20.0]
    b_grid     = [0.0, 10.0, 20.0]
    e_grid     = [-1.0, 0.0, 1.0]
    alpha_grid = [0.0, 0.5, 2.0]
    beta_grid  = [0.0, 0.5, 2.0]

    best = None
    for a0 in a_grid:
        for b0 in b_grid:
            for e0 in e_grid:
                for al0 in alpha_grid:
                    for be0 in beta_grid:
                        x0 = np.array([a0, b0, e0, al0, be0], dtype=float)
                        try:
                            res = minimize(neg, x0, method="L-BFGS-B",
                                           options=dict(maxiter=1000, ftol=1e-10))
                        except (FloatingPointError, ValueError):
                            continue
                        if not np.isfinite(res.fun):
                            continue
                        if best is None or res.fun < best.fun:
                            best = res
    a, b, e, alpha, beta = best.x
    return dict(E=math.exp(e), A=math.exp(a), B=math.exp(b),
                alpha=alpha, beta=beta, huber_loss=float(best.fun))


# ---------------------------------------------------------------------------
# Residual reporting
# ---------------------------------------------------------------------------

def residual_stats(N, D, L_true, coeffs, label):
    L_pred = L_hat(N, D, **{k: coeffs[k] for k in ("E","A","B","alpha","beta")})
    log_resid = np.log(L_pred) - np.log(L_true)
    abs_resid = L_pred - L_true
    print(f"  {label:25s}  "
          f"mean |Δlog L| = {np.mean(np.abs(log_resid)):.5f}   "
          f"max |Δlog L| = {np.max(np.abs(log_resid)):.5f}   "
          f"mean |ΔL| = {np.mean(np.abs(abs_resid)):.5f}")
    return log_resid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- Load data ----
    epoch = pd.read_csv(EPOCH_CSV)
    epoch = epoch.rename(columns={"Model Size": "n_params",
                                  "Training FLOP": "compute"})
    # D is not provided in the Epoch AI SVG extraction — reconstruct via C = 6ND.
    epoch["n_tokens"] = epoch["compute"] / (6.0 * epoch["n_params"])

    iso = pd.read_csv(ISOFLOP_CSV)

    print(f"Loaded {len(epoch)} Epoch AI empirical points "
          f"(Chinchilla Fig 4 left extraction)")
    print(f"Loaded {len(iso)} isoFLOP slice points across "
          f"{iso['flops_label'].nunique()} compute budgets")
    print()

    # ---- Fit from scratch on the Epoch AI data (demonstrates bias fix) ----
    print("Re-fitting L̂(N, D) on the Epoch AI scatter with Huber-LSE "
          "(multi-start, delta=1e-3)…")
    fit = fit_huber_lse(epoch["n_params"].values,
                        epoch["n_tokens"].values,
                        epoch["loss"].values)
    print("  Best fit:"
          f"  E={fit['E']:.4f}  A={fit['A']:.2f}  B={fit['B']:.2f}  "
          f"alpha={fit['alpha']:.4f}  beta={fit['beta']:.4f}  "
          f"(Huber loss = {fit['huber_loss']:.6f})")
    print()

    # ---- Compare fit sources on the two datasets ----
    sources = {
        "Hoffmann 2022 (biased)":        HOFFMANN_2022,
        "Besiroglu 2024 (bias fix)":     BESIROGLU_2024,
        "Our refit on Epoch AI data":    fit,
    }

    print("Residuals vs Epoch AI empirical scatter (245 pts):")
    for label, coeffs in sources.items():
        residual_stats(epoch["n_params"].values,
                       epoch["n_tokens"].values,
                       epoch["loss"].values, coeffs, label)
    print()

    print("Residuals vs Chinchilla Fig 4 isoFLOP slices (114 pts):")
    for label, coeffs in sources.items():
        residual_stats(iso["n_params"].values,
                       iso["n_tokens"].values,
                       iso["loss"].values, coeffs, label)
    print()

    # ---- Compute-optimal scaling exponents ----
    for label, c in sources.items():
        a = c["beta"] / (c["alpha"] + c["beta"])
        G = ((c["alpha"] * c["A"]) / (c["beta"] * c["B"])) ** (1.0 / (c["alpha"] + c["beta"]))
        print(f"  {label:30s}  N_opt = {G:.4f} · (C/6)^{a:.4f}")
    print()

    # ---- Choose which fit to use for the HTML ("bias fix") ----
    chosen = BESIROGLU_2024
    chosen_name = "Besiroglu et al. 2024 (bias fix)"
    print(f"Using {chosen_name} coefficients for Fig 4 left curves.")
    print()

    # ---- Fig 4 left: parametric L̂ curves per isoFLOP budget ----
    flops_order = ["6e+18", "1e+19", "3e+19", "6e+19",
                   "1e+20", "3e+20", "6e+20", "1e+21", "3e+21"]
    flops_colors = {
        "6e+18": "#a9e1bd", "1e+19": "#60ceac", "3e+19": "#3eb4ad",
        "6e+19": "#3497a9", "1e+20": "#357ba3", "3e+20": "#395d9c",
        "6e+20": "#413f80", "1e+21": "#382a54", "3e+21": "#251729",
    }
    fig4_left_curves: dict[str, dict] = {}
    for fl in flops_order:
        C = float(f"{float(fl):.0e}")
        # N sweeps two decades around the analytic optimum.
        N_opt, _ = compute_optimal(C, **chosen)
        N_grid = np.geomspace(N_opt / 10.0, N_opt * 10.0, 120)
        D_grid = C / (6.0 * N_grid)
        loss   = L_hat(N_grid, D_grid, **{k: chosen[k] for k in ("E","A","B","alpha","beta")})
        fig4_left_curves[fl] = dict(
            N    = N_grid.tolist(),
            D    = D_grid.tolist(),
            C    = [C] * len(N_grid),
            loss = loss.tolist(),
            color = flops_colors[fl],
        )

    # ---- Fig 4 right: compute-optimal frontier (N_opt, D_opt vs C) ----
    C_frontier = np.geomspace(4e18, 3e24, 120)
    N_frontier, D_frontier = compute_optimal(C_frontier, **chosen)
    L_frontier = L_hat(N_frontier, D_frontier, **{k: chosen[k] for k in ("E","A","B","alpha","beta")})
    a_exp = chosen["beta"] / (chosen["alpha"] + chosen["beta"])
    G_coef = ((chosen["alpha"] * chosen["A"]) / (chosen["beta"] * chosen["B"])) ** (1.0 / (chosen["alpha"] + chosen["beta"]))

    fig4_right = dict(
        C    = C_frontier.tolist(),
        N    = N_frontier.tolist(),
        D    = D_frontier.tolist(),
        loss = L_frontier.tolist(),
        exponent    = a_exp,
        coefficient = G_coef,
    )

    # ---- Dump everything for the HTML ----
    out = dict(
        source_note = (
            "Generated by chinchilla_fig4.py. Parametric L̂(N,D) curves use "
            "the Besiroglu et al. (2024) bias-corrected coefficients "
            "(Chinchilla Scaling: A replication attempt, arXiv:2404.10102)."
        ),
        coefficients = dict(
            hoffmann_2022           = HOFFMANN_2022,
            besiroglu_2024_biasfix  = BESIROGLU_2024,
            our_refit_on_epoch_data = {k: float(v) for k, v in fit.items()},
        ),
        fig4_left_parametric_curves = fig4_left_curves,
        fig4_right_frontier         = fig4_right,
        flops_order = flops_order,
        flops_colors = flops_colors,
    )
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_JSON.name}")


if __name__ == "__main__":
    main()
