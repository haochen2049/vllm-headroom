"""System identification of the step-time model from a profile.

Decode parameters (``b0``, ``b_dec``, ``b_ctx``) come from inter-token gaps observed while a known
number of sequences decode together and no other request is being prefilled. Prefill parameters
(``b_pf``, ``b_pfatt``) and the fixed client-visible ``overhead`` come from time-to-first-token of
single requests on an idle server. Uncertainty comes from a request-level bootstrap of both fits.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sim import StepModel


def _irls(X: np.ndarray, y: np.ndarray, iters: int = 20, nonneg: bool = True) -> np.ndarray:
    """Huber-weighted least squares with non-negative coefficients (columns dropped when negative)."""
    active = list(range(X.shape[1]))
    while True:
        Xa = X[:, active]
        w = np.ones(len(y))
        beta = np.zeros(len(active))
        for _ in range(iters):
            sw = np.sqrt(w)
            beta, *_ = np.linalg.lstsq(Xa * sw[:, None], y * sw, rcond=None)
            r = y - Xa @ beta
            s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
            k = 1.345 * s
            a = np.abs(r)
            w = np.where(a <= k, 1.0, k / np.maximum(a, 1e-18))
        if not nonneg or np.all(beta >= 0) or len(active) == 1:
            out = np.zeros(X.shape[1])
            out[active] = np.maximum(beta, 0) if nonneg else beta
            return out
        # drop the most negative non-intercept column and refit
        cand = [(b, j) for b, j in zip(beta, active) if j != 0]
        if not cand:
            out = np.zeros(X.shape[1])
            out[active] = np.maximum(beta, 0)
            return out
        b, j = min(cand)
        if b >= 0:
            out = np.zeros(X.shape[1])
            out[active] = np.maximum(beta, 0)
            return out
        active.remove(j)


@dataclass
class FitResult:
    model: StepModel
    draws: list[StepModel]
    decode_rmse: float
    prefill_rmse: float
    n_decode_obs: int
    n_prefill_obs: int


def _fit_decode(dec: np.ndarray) -> np.ndarray:
    # columns: itl, n_dec, ctx_tokens
    X = np.column_stack([np.ones(len(dec)), dec[:, 1], dec[:, 2] / 1e3])
    return _irls(X, dec[:, 0])


def _fit_prefill(pf: np.ndarray, b0: float, budget: int) -> np.ndarray:
    # columns: ttft, prompt_tokens; TTFT = overhead + n_steps * b0 + b_pf * P + b_pfatt * P^2 / 2e6
    P = pf[:, 1]
    steps = np.ceil(P / max(1, budget))
    y = pf[:, 0] - steps * b0
    X = np.column_stack([np.ones(len(pf)), P, P ** 2 / 2e6])
    return _irls(X, y)


def _fit_lag(m: StepModel, probes: np.ndarray) -> float:
    """Mean extra step boundaries before a request arriving at a busy engine is scheduled.

    Probe TTFT = overhead + (residual of the running step, mean 1/2) + lag steps + its own prefill step.
    """
    if probes is None or len(probes) == 0:
        return 0.0
    ttft, P, n, ctx = probes[:, 0], probes[:, 1], probes[:, 2], probes[:, 3]
    s_dec = m.b0 + m.b_dec * n + m.b_ctx * ctx / 1e3
    s_pf = s_dec + m.b_pf * P + m.b_pfatt * P ** 2 / 2e6
    lag = (ttft - m.overhead - s_pf) / s_dec - 0.5
    return float(np.clip(np.mean(lag), 0.0, 3.0))


def fit_profile(decode_obs, prefill_obs, decode_group, prefill_group, budget: int = 8192,
                n_boot: int = 50, seed: int = 0, probe_obs=None) -> FitResult:
    """Fit the step model.

    ``decode_obs``: rows ``(itl_s, n_decoding, decode_context_tokens)``; ``prefill_obs``: rows
    ``(ttft_s, prompt_tokens)``; ``*_group``: request id of each row, used for the bootstrap;
    ``probe_obs``: optional rows ``(ttft_s, prompt_tokens, n_decoding, decode_context_tokens)`` from
    requests sent while the engine was busy, used for ``admit_lag``.
    """
    pr = np.asarray(probe_obs, float).reshape(-1, 4) if probe_obs is not None and len(probe_obs) else None
    dec = np.asarray(decode_obs, float).reshape(-1, 3)
    pf = np.asarray(prefill_obs, float).reshape(-1, 2)
    if len(dec) < 10:
        raise ValueError("too few clean decode observations to fit (need >= 10)")
    if len(pf) < 3:
        raise ValueError("too few prefill observations to fit (need >= 3)")
    dg = np.asarray(decode_group)
    pg = np.asarray(prefill_group)

    def one(d: np.ndarray, p: np.ndarray, q: np.ndarray | None = None) -> StepModel:
        b0, bdec, bctx = _fit_decode(d)
        ov, bpf, bpfatt = _fit_prefill(p, b0, budget)
        m = StepModel(b0=b0, b_dec=bdec, b_ctx=bctx, b_pf=bpf, b_pfatt=bpfatt, overhead=ov)
        m.admit_lag = _fit_lag(m, pr if q is None else q)
        return m

    m = one(dec, pf)
    rd = dec[:, 0] - (m.b0 + m.b_dec * dec[:, 1] + m.b_ctx * dec[:, 2] / 1e3)
    P = pf[:, 1]
    rp = pf[:, 0] - (m.overhead + np.ceil(P / max(1, budget)) * m.b0 + m.b_pf * P + m.b_pfatt * P ** 2 / 2e6)

    rng = np.random.default_rng(seed)
    ud, up = np.unique(dg), np.unique(pg)
    draws = []
    for _ in range(n_boot):
        sd = rng.choice(ud, size=len(ud), replace=True)
        sp = rng.choice(up, size=len(up), replace=True)
        d = np.concatenate([dec[dg == g] for g in sd])
        p = np.concatenate([pf[pg == g] for g in sp])
        q = pr[rng.integers(len(pr), size=len(pr))] if pr is not None else None
        try:
            draws.append(one(d, p, q))
        except (np.linalg.LinAlgError, ValueError):
            continue
    return FitResult(m, draws, float(np.sqrt(np.mean(rd ** 2))), float(np.sqrt(np.mean(rp ** 2))),
                     len(dec), len(pf))
