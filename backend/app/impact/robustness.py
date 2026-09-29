"""Ranking robustness: how stable is the candidate ranking if the (uncalibrated) Evidence Correlation Score
weights were different? Weights are resampled from a Dirichlet distribution centred on the configured weights
and every candidate is re-scored from its unchanged per-factor scores."""
from __future__ import annotations

import numpy as np


def ranking_robustness(candidates: list[dict], weights: dict, n: int = 2000, concentration: float = 40.0, seed: int = 0) -> dict:
    if not candidates:
        return {"available": False, "reason": "no candidates"}
    keys = [k for k in weights if weights[k] > 0]
    w0 = np.array([weights[k] for k in keys], float)
    w0 = w0 / w0.sum()
    F = np.array([[float(c["feature_scores"].get(k, 0.0)) for k in keys] for c in candidates])     # (C, K)
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(w0 * concentration, size=n)                                                  # (n, K)
    S = W @ F.T                                                                                     # (n, C)
    order = np.argsort(-S, axis=1)
    ranks = np.empty_like(order)
    ranks[np.arange(n)[:, None], order] = np.arange(1, len(candidates) + 1)[None, :]
    per = []
    for i, c in enumerate(candidates):
        r = ranks[:, i]
        per.append({"mmsi": c["vessel"]["mmsi"], "name": c["vessel"].get("name"), "base_rank": c["rank"],
                    "p_rank1": float((r == 1).mean()), "p_top3": float((r <= 3).mean()),
                    "rank_p05": int(np.percentile(r, 5)), "rank_p95": int(np.percentile(r, 95))})
    top = per[0]
    verdict = ("ROBUST" if top["p_rank1"] >= 0.8 else "LIKELY" if top["p_rank1"] >= 0.5 else "CONTESTED")
    return {"available": True, "n_samples": n, "concentration": concentration, "candidates": per, "verdict": verdict,
            "note": f"#1 {top['name'] or top['mmsi']} stays first in {100 * top['p_rank1']:.0f} % of {n} weight "
                    "perturbations (Dirichlet around the configured weights). "
                    + ("The ranking does not hinge on the exact weights." if verdict == "ROBUST" else
                       "The ranking is sensitive to the weights — investigate the top candidates with equal priority."
                       if verdict == "CONTESTED" else "The ranking is fairly stable but not decisive.")}
