import numpy as np
import pytest

from vllm_headroom.analysis import evaluate, max_rate, parse_slo, with_changes
from vllm_headroom.fit import fit_profile
from vllm_headroom.sim import EngineConfig, StepModel
from vllm_headroom.workload import synthetic

TRUE = StepModel(b0=0.02, b_dec=0.00025, b_ctx=0.00003, b_pf=0.00006, b_pfatt=0.0, overhead=0.005)


def _synthetic_obs(seed=0):
    rng = np.random.default_rng(seed)
    dec, dg = [], []
    for g, (n, per) in enumerate([(n, p) for n in (1, 2, 4, 8, 16, 32, 64) for p in (300, 2100)]):
        for k in range(30):
            ctx = n * (per + k * 6)
            itl = TRUE.step_time(n, ctx) * rng.lognormal(0, 0.02)
            dec.append((itl, n, ctx))
            dg.append(g)
    pf, pg = [], []
    for i, P in enumerate([128, 512, 1024, 2048, 4096] * 3):
        pf.append((TRUE.overhead + TRUE.step_time(0, 0, P) * rng.lognormal(0, 0.02), P))
        pg.append(i)
    return dec, dg, pf, pg


def test_fit_recovers_parameters():
    dec, dg, pf, pg = _synthetic_obs()
    r = fit_profile(dec, pf, dg, pg, budget=8192, n_boot=10)
    assert r.model.b0 == pytest.approx(TRUE.b0, rel=0.05)
    assert r.model.b_dec == pytest.approx(TRUE.b_dec, rel=0.15)
    assert r.model.b_ctx == pytest.approx(TRUE.b_ctx, rel=0.15)
    assert r.model.b_pf == pytest.approx(TRUE.b_pf, rel=0.1)
    assert len(r.draws) == 10


def test_fit_admit_lag_from_probes():
    dec, dg, pf, pg = _synthetic_obs()
    rng = np.random.default_rng(3)
    probes = []
    for _ in range(40):
        n, ctx, P = 32, 32 * 600, 128
        s = TRUE.step_time(n, ctx)
        ttft = TRUE.overhead + rng.uniform(0, s) + 1.0 * s + TRUE.step_time(n, ctx, P)
        probes.append((ttft, P, n, ctx))
    r = fit_profile(dec, pf, dg, pg, n_boot=0, probe_obs=probes)
    assert r.model.admit_lag == pytest.approx(1.0, abs=0.25)


def test_parse_slo():
    assert parse_slo(["ttft_p95=2", "e2e_p99.9=30"]) == {("ttft", 95.0): 2.0, ("e2e", 99.9): 30.0}
    with pytest.raises(ValueError):
        parse_slo(["latency_p95=1"])


def test_headroom_monotone_in_slo_and_replicas():
    wl = synthetic(800, 200, seed=1)
    eng = EngineConfig(max_num_seqs=64, max_num_batched_tokens=4096)
    tight = max_rate(TRUE, eng, wl, parse_slo(["ttft_p95=0.3"]), n=600)
    loose = max_rate(TRUE, eng, wl, parse_slo(["ttft_p95=1.0"]), n=600)
    assert loose.lam_max >= tight.lam_max
    _, eng2 = with_changes(TRUE, eng, {"replicas": 2})
    two = max_rate(TRUE, eng2, wl, parse_slo(["ttft_p95=0.3"]), n=600)
    assert two.lam_max > 1.5 * tight.lam_max


def test_latency_increases_with_load():
    wl = synthetic(800, 200, seed=1)
    eng = EngineConfig(max_num_seqs=64)
    slo = parse_slo(["e2e_p95=100"])
    lo = evaluate(TRUE, eng, wl, 1.0, slo, n=600)[("e2e", 95.0)]
    hi = evaluate(TRUE, eng, wl, 6.0, slo, n=600)[("e2e", 95.0)]
    assert hi > lo
