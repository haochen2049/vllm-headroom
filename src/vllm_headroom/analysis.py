"""Latency-versus-load curves, SLO headroom and the validation backtest."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from .fit import FitResult, fit_profile
from .sim import EngineConfig, StepModel, poisson_arrivals, saturation_rate, simulate
from .workload import Workload

METRICS = ("ttft", "e2e", "tpot")


def parse_slo(items: list[str]) -> dict:
    """``["ttft_p95=2", "e2e_p99=30"]`` -> ``{("ttft", 95): 2.0, ("e2e", 99): 30.0}``."""
    out = {}
    for s in items:
        k, v = s.split("=")
        metric, q = k.strip().lower().rsplit("_p", 1)
        if metric not in METRICS:
            raise ValueError(f"unknown SLO metric {metric!r}; use one of {METRICS}")
        out[(metric, float(q))] = float(v)
    if not out:
        raise ValueError("at least one SLO is required, e.g. ttft_p95=2")
    return out


def slo_label(key) -> str:
    m, q = key
    return f"{m}_p{q:g}"


def fit_from_profile(profile: dict, n_boot: int = 40, seed: int = 0, exclude_validation: bool = True) -> FitResult:
    pf = profile["prefill"]
    return fit_profile(
        [o[:3] for o in profile["decode_obs"]],
        [(p["ttft"], p["prompt_tokens"]) for p in pf],
        profile["decode_group"], list(range(len(pf))),
        int(profile["engine"]["max_num_batched_tokens"]), n_boot=n_boot, seed=seed,
        probe_obs=[(q["ttft"], q["prompt_tokens"], q["n_dec"], q["ctx"]) for q in profile.get("probes", [])],
    )


def _quantiles(res, slo: dict, warm: int) -> dict:
    out = {}
    for (m, q) in slo:
        x = getattr(res, m)[warm:]
        x = x[np.isfinite(x)]
        out[(m, q)] = float(np.percentile(x, q)) if len(x) else math.inf
    return out


def evaluate(model: StepModel, eng: EngineConfig, wl: Workload, rate: float, slo: dict,
             n: int = 1500, seed: int = 0) -> dict:
    """Latency quantiles at a Poisson arrival rate (first 10% of requests discarded as warm-up)."""
    rng = np.random.default_rng(seed)
    arr = poisson_arrivals(rate, n, rng)
    p, o = wl.sample(n, rng)
    res = simulate(arr, p, o, model, eng, seed=seed)
    return _quantiles(res, slo, n // 10)


@dataclass
class Headroom:
    lam_sat: float
    lam_max: float
    binding: str


def max_rate(model: StepModel, eng: EngineConfig, wl: Workload, slo: dict, n: int = 1500, seed: int = 0,
             lam_sat: float | None = None, tol: float = 0.02) -> Headroom:
    """Largest Poisson rate meeting every SLO, found by bisection below the saturation rate."""
    if lam_sat is None:
        lam_sat = saturation_rate(model, eng, wl.prompt, wl.output, n=min(n, 1500), seed=seed)
    hi_cap = 0.98 * lam_sat

    def ok(rate):
        q = evaluate(model, eng, wl, rate, slo, n, seed)
        bad = [k for k, v in q.items() if v > slo[k]]
        return (not bad), (slo_label(bad[0]) if bad else "")

    lo, hi = 0.0, hi_cap
    good, why = ok(hi)
    if good:
        return Headroom(lam_sat, hi, "saturation")
    binding = why
    if not ok(1e-3 * lam_sat)[0]:
        return Headroom(lam_sat, 0.0, ok(1e-3 * lam_sat)[1])
    lo = 1e-3 * lam_sat
    while hi - lo > tol * hi:
        mid = 0.5 * (lo + hi)
        g, w = ok(mid)
        if g:
            lo = mid
        else:
            hi, binding = mid, w
    return Headroom(lam_sat, lo, binding)


def with_changes(model: StepModel, eng: EngineConfig, changes: dict) -> tuple[StepModel, EngineConfig]:
    e = {}
    for k, v in changes.items():
        if k in ("max_num_seqs", "max_num_batched_tokens", "replicas"):
            e[k] = int(v)
        elif k == "kv_capacity_tokens":
            e[k] = float(v)
        elif k == "kv_scale":
            e["kv_capacity_tokens"] = eng.kv_capacity_tokens * float(v)
        elif k == "router":
            e[k] = str(v)
        else:
            raise ValueError(f"unknown change {k!r}")
    return model, replace(eng, **e)


def headroom_analysis(fit: FitResult, eng: EngineConfig, wl: Workload, slo: dict, current_rate: float | None,
                      n: int = 1500, draws: int = 20, grid: int = 18, seed: int = 0, log=print) -> dict:
    """Point estimate and bootstrap band of the latency curve and the maximum rate meeting the SLO."""
    hr = max_rate(fit.model, eng, wl, slo, n, seed)
    log(f"saturation {hr.lam_sat:.3f} req/s; max rate meeting SLO {hr.lam_max:.3f} req/s ({hr.binding})")
    models = fit.draws[:draws]
    k = max(3, grid // 2)
    rates = np.concatenate([np.linspace(0.05, 0.8, grid - k, endpoint=False), np.linspace(0.8, 0.99, k)]) * hr.lam_sat
    curve = {slo_label(k): [] for k in slo}
    band = {slo_label(k): [] for k in slo}
    for r in rates:
        q = evaluate(fit.model, eng, wl, r, slo, n, seed)
        for k, v in q.items():
            curve[slo_label(k)].append(v)
    draw_q = {slo_label(k): np.full((len(models), grid), np.nan) for k in slo}
    lam_max_draws = []
    for di, m in enumerate(models):
        h = max_rate(m, eng, wl, slo, n, seed + 1 + di)
        lam_max_draws.append(h.lam_max)
        for gi, r in enumerate(rates):
            if r >= 0.98 * h.lam_sat:
                continue
            q = evaluate(m, eng, wl, r, slo, n, seed + 1 + di)
            for k, v in q.items():
                draw_q[slo_label(k)][di, gi] = v
    for lab, a in draw_q.items():
        with np.errstate(all="ignore"):
            band[lab] = [[float(np.nanpercentile(a[:, g], 10)) if np.any(np.isfinite(a[:, g])) else None,
                          float(np.nanpercentile(a[:, g], 90)) if np.any(np.isfinite(a[:, g])) else None]
                         for g in range(grid)]
    lm = np.asarray(lam_max_draws) if lam_max_draws else np.asarray([hr.lam_max])
    out = {
        "slo": {slo_label(k): v for k, v in slo.items()},
        "lam_sat": hr.lam_sat, "lam_max": hr.lam_max, "binding": hr.binding,
        "lam_max_band": [float(np.percentile(lm, 10)), float(np.percentile(lm, 90))],
        "rates": rates.tolist(), "curve": curve, "band": band,
        "current_rate": current_rate,
    }
    if current_rate:
        out["headroom_pct"] = 100.0 * (hr.lam_max / current_rate - 1.0)
        out["headroom_pct_band"] = [100.0 * (out["lam_max_band"][0] / current_rate - 1),
                                    100.0 * (out["lam_max_band"][1] / current_rate - 1)]
        out["at_current"] = {slo_label(k): v for k, v in
                             evaluate(fit.model, eng, wl, current_rate, slo, n, seed).items()}
    return out


def backtest(profile: dict, fit: FitResult, eng: EngineConfig, quantiles=(50, 95), reps: int = 20,
             seed: int = 0) -> list[dict]:
    """Predict each validation level's latency quantiles from the fit and compare with measurement.

    The simulator replays the measured send times and request sizes, so differences come from the
    model, not from the arrival sample. The 10-90% range covers simulation noise and the bootstrap.
    """
    rows = []
    models = [fit.model] + fit.draws[: reps - 1]
    for lv in profile.get("validation", []):
        req = [r for r in lv["requests"] if not r["err"] and r["ttft"] is not None]
        if len(req) < 5:
            continue
        arr = np.array([r["t_send"] for r in req])
        p = np.array([r["prompt_tokens"] for r in req])
        o = np.array([r["output_tokens"] for r in req])
        meas = {"ttft": np.array([r["ttft"] for r in req]), "e2e": np.array([r["e2e"] for r in req])}
        preds = {(m, q): [] for m in meas for q in quantiles}
        for mi, m in enumerate(models):
            res = simulate(arr, p, o, m, eng, seed=seed + mi)
            for (met, q) in preds:
                x = getattr(res, met)
                preds[(met, q)].append(float(np.percentile(x[np.isfinite(x)], q)))
        for (met, q), v in preds.items():
            mv = float(np.percentile(meas[met], q))
            rows.append({
                "fraction": lv["fraction"], "rate": lv["rate"], "n": len(req), "metric": f"{met}_p{q:g}",
                "measured": mv, "predicted": v[0],
                "range": [float(np.percentile(v, 10)), float(np.percentile(v, 90))],
                "rel_error": (v[0] - mv) / mv if mv > 0 else None,
            })
    return rows
