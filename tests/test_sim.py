import math

import numpy as np
import pytest

from vllm_headroom.sim import EngineConfig, StepModel, saturation_rate, simulate

M = StepModel(b0=0.012, b_dec=0.0002, b_ctx=0.00002, b_pf=0.00005, overhead=0.01)


def test_single_request_closed_form():
    r = simulate([0.0], [1000], [100], M, EngineConfig())
    ttft = 0.01 + M.step_time(0, 0, 1000, 1000 * 500)
    assert r.ttft[0] == pytest.approx(ttft)
    decode = sum(M.step_time(1, 1000 + k) for k in range(1, 100))
    assert r.e2e[0] == pytest.approx(ttft + decode)


def _reference(arr, P, O, m, S=256, B=8192):
    """Plain step-by-step continuous batching (no KV limit, no lag) for cross-checking."""
    n = len(arr)
    order = sorted(range(n), key=lambda i: arr[i])
    waiting, running = [], []
    done_pf, gen = [0] * n, [0] * n
    first, last = [None] * n, [None] * n
    t, k = 0.0, 0
    while k < n or waiting or running:
        while k < n and arr[order[k]] <= t:
            waiting.append(order[k])
            k += 1
        if not waiting and not running:
            t = arr[order[k]]
            continue
        dec = [i for i in running if done_pf[i] >= P[i]]
        budget = B - len(dec)
        chunks = []
        for i in running:
            if done_pf[i] < P[i] and budget > 0:
                c = min(P[i] - done_pf[i], budget)
                chunks.append((i, c))
                budget -= c
        while waiting and budget > 0 and len(running) < S:
            i = waiting.pop(0)
            running.append(i)
            c = min(P[i], budget)
            chunks.append((i, c))
            budget -= c
        ctx = sum(P[i] + gen[i] for i in dec)
        t += m.step_time(len(dec), ctx, sum(c for _, c in chunks), sum(c * (done_pf[i] + c / 2) for i, c in chunks))
        for i in dec:
            gen[i] += 1
        for i, c in chunks:
            done_pf[i] += c
            if done_pf[i] >= P[i]:
                gen[i] += 1
                if first[i] is None:
                    first[i] = t
        keep = []
        for i in running:
            if done_pf[i] >= P[i] and gen[i] >= O[i]:
                last[i] = t
            else:
                keep.append(i)
        running = keep
    return np.array(first) - np.array(arr), np.array(last) - np.array(arr)


def test_macro_steps_match_plain_stepping():
    rng = np.random.default_rng(5)
    n = 60
    arr = list(np.round(np.cumsum(rng.exponential(0.15, n)), 6))
    P = [int(x) for x in rng.integers(20, 3000, n)]
    O = [int(x) for x in rng.integers(1, 300, n)]
    m = StepModel(M.b0, M.b_dec, M.b_ctx, M.b_pf, 2e-4)
    ref_ttft, ref_e2e = _reference(arr, P, O, m, S=16, B=2048)
    r = simulate(arr, P, O, m, EngineConfig(max_num_seqs=16, max_num_batched_tokens=2048))
    np.testing.assert_allclose(r.ttft, ref_ttft, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(r.e2e, ref_e2e, rtol=1e-9, atol=1e-9)


def test_all_requests_finish_and_order():
    rng = np.random.default_rng(1)
    n = 400
    arr = np.cumsum(rng.exponential(0.05, n))
    P = rng.integers(10, 2000, n)
    O = rng.integers(1, 400, n)
    r = simulate(arr, P, O, M, EngineConfig(max_num_seqs=32, max_num_batched_tokens=2048))
    assert np.all(np.isfinite(r.e2e))
    assert np.all(r.e2e >= r.ttft - 1e-12)
    assert np.all(r.ttft > 0)


def test_kv_pressure_preempts_but_completes():
    rng = np.random.default_rng(2)
    n = 120
    r = simulate(np.zeros(n), rng.integers(500, 1500, n), rng.integers(200, 600, n), M,
                 EngineConfig(max_num_seqs=64, kv_capacity_tokens=40_000))
    assert np.all(np.isfinite(r.e2e))
    assert r.preemptions > 0


def test_more_replicas_more_throughput():
    rng = np.random.default_rng(4)
    p, o = rng.integers(100, 1600, 500), rng.integers(20, 400, 500)
    one = saturation_rate(M, EngineConfig(replicas=1), p, o, n=600)
    two = saturation_rate(M, EngineConfig(replicas=2), p, o, n=600)
    assert two == pytest.approx(2 * one, rel=0.1)


def test_admit_lag_adds_steps_when_busy():
    arr, P, O = [0] * 8 + [0.5], [100] * 9, [500] * 8 + [2]
    lag0 = simulate(arr, P, O, StepModel(0.01, 1e-4, 1e-5, 5e-5), EngineConfig()).ttft[-1]
    lag1 = simulate(arr, P, O, StepModel(0.01, 1e-4, 1e-5, 5e-5, admit_lag=1.0), EngineConfig()).ttft[-1]
    step = 0.01 + 8e-4
    assert lag1 - lag0 == pytest.approx(step, rel=0.2)
    # an idle engine schedules immediately regardless of the lag
    idle = simulate([0.0], [100], [2], StepModel(0.01, 1e-4, 1e-5, 5e-5, admit_lag=2.0), EngineConfig()).ttft[0]
    assert idle == pytest.approx(0.01 + 100 * 5e-5)


def test_infinite_kv_roundtrip():
    e = EngineConfig()
    assert math.isinf(EngineConfig.from_dict(e.to_dict()).kv_capacity_tokens)
