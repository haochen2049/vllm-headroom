"""Discrete-event simulator of one or more vLLM-style continuous-batching replicas.

Each engine step serves every running sequence that has finished prefill (one new token each), then
continues partially prefilled sequences and admits waiting requests first-come-first-served while the
token budget (``max_num_batched_tokens``), ``max_num_seqs`` and free KV capacity allow. A step's
duration comes from :class:`StepModel`. When a sequence needs KV space that is not free, the most
recently admitted sequence is preempted and later recomputed (prompt plus generated tokens).

Long decode-only stretches are advanced analytically ("macro steps"): between two events the batch is
fixed, so the step time grows linearly with context and the end time has a closed form. An arrival
routed to a replica in a macro step cuts it at the next step boundary.
"""
from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class StepModel:
    """Engine step time in seconds.

    ``b0 + b_dec * n_decode + b_ctx * decode_context_tokens / 1e3 + b_pf * prefill_tokens
    + b_pfatt * prefill_attention / 1e6``, where ``prefill_attention`` sums, over prefill chunks,
    chunk tokens times the mean context position of the chunk. ``overhead`` is a fixed client-visible
    latency (HTTP, tokenisation, queue hand-off) added to TTFT and end-to-end latency.
    ``admit_lag`` is the mean number of extra step boundaries a request that arrives while the engine is
    busy waits before it can be scheduled (vLLM V1 schedules the next step while the current one runs).
    """

    b0: float
    b_dec: float
    b_ctx: float
    b_pf: float
    b_pfatt: float = 0.0
    overhead: float = 0.0
    admit_lag: float = 0.0

    def step_time(self, n_dec: int, ctx: float, pf: float = 0.0, pf_att: float = 0.0) -> float:
        return (self.b0 + self.b_dec * n_dec + self.b_ctx * ctx / 1e3 + self.b_pf * pf
                + self.b_pfatt * pf_att / 1e6)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StepModel":
        return cls(**{k: float(d[k]) for k in cls.__dataclass_fields__ if k in d})


@dataclass
class EngineConfig:
    max_num_seqs: int = 256
    max_num_batched_tokens: int = 8192
    kv_capacity_tokens: float = math.inf
    replicas: int = 1
    router: str = "least_loaded"  # or "round_robin"

    def to_dict(self) -> dict:
        d = asdict(self)
        if math.isinf(d["kv_capacity_tokens"]):
            d["kv_capacity_tokens"] = None
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "EngineConfig":
        d = dict(d)
        if d.get("kv_capacity_tokens") is None:
            d["kv_capacity_tokens"] = math.inf
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class SimResult:
    arrival: np.ndarray
    prompt: np.ndarray
    output: np.ndarray
    ttft: np.ndarray        # seconds, client-visible
    e2e: np.ndarray         # seconds, client-visible
    preemptions: int
    makespan: float

    @property
    def tpot(self) -> np.ndarray:
        """Mean time per output token after the first, NaN for single-token outputs."""
        with np.errstate(invalid="ignore", divide="ignore"):
            t = (self.e2e - self.ttft) / (self.output - 1)
        t[self.output <= 1] = np.nan
        return t


@dataclass
class _Replica:
    waiting: deque = field(default_factory=deque)
    running: list = field(default_factory=list)  # admission order
    kv_used: float = 0.0
    busy: bool = False
    version: int = 0
    nb: int = 0             # completed steps
    # current step
    kind: str = ""          # "single" or "macro"
    t0: float = 0.0
    t_end: float = 0.0
    dec: list = field(default_factory=list)
    chunks: list = field(default_factory=list)  # (req, tokens)
    A: float = 0.0
    c: float = 0.0
    K: int = 0


def _cum(A: float, c: float, j: int) -> float:
    return j * A + c * j * (j - 1) / 2.0


def simulate(arrival, prompt, output, model: StepModel, eng: EngineConfig, seed: int = 0) -> SimResult:
    """Simulate requests (arrival times in seconds, prompt and output token counts)."""
    arrival = np.asarray(arrival, float)
    prompt = np.asarray(prompt, np.int64)
    output = np.maximum(np.asarray(output, np.int64), 1)
    order = np.argsort(arrival, kind="stable")
    n = len(arrival)
    cap = float(eng.kv_capacity_tokens)
    B = int(eng.max_num_batched_tokens)
    S = int(eng.max_num_seqs)

    p_eff = prompt.astype(np.int64).copy()      # tokens to (re)prefill
    done_pf = np.zeros(n, np.int64)
    gen = np.zeros(n, np.int64)
    kv = np.zeros(n, float)
    t_first = np.full(n, np.nan)
    t_done = np.full(n, np.nan)
    ready = np.zeros(n, np.int64)                # first step boundary index at which a request may be scheduled
    preempt = 0
    lag_rng = np.random.default_rng(seed)
    lag_int = int(math.floor(max(0.0, model.admit_lag)))
    lag_frac = max(0.0, model.admit_lag) - lag_int

    reps = [_Replica() for _ in range(max(1, int(eng.replicas)))]
    heap: list = []
    rr = 0

    def plan(r: _Replica, ri: int, t: float) -> None:
        nonlocal preempt
        dec = [i for i in r.running if done_pf[i] >= p_eff[i]]
        # KV growth for decodes: preempt the most recently admitted sequences until it fits
        while dec and r.kv_used + len(dec) > cap:
            v = r.running.pop()
            r.kv_used -= kv[v]
            kv[v] = 0.0
            p_eff[v] = prompt[v] + gen[v]
            done_pf[v] = 0
            r.waiting.appendleft(v)
            ready[v] = 0
            preempt += 1
            dec = [i for i in r.running if done_pf[i] >= p_eff[i]]
        budget = B - len(dec)
        chunks = []
        for i in r.running:
            if done_pf[i] < p_eff[i] and budget > 0:
                k = int(min(p_eff[i] - done_pf[i], budget))
                chunks.append((i, k))
                budget -= k
        admitted = False
        if r.waiting and not dec and not chunks and ready[r.waiting[0]] > r.nb:
            ready[r.waiting[0]] = r.nb          # nothing else to run: the scheduler picks it up now
        while r.waiting and budget > 0 and len(r.running) < S:
            i = r.waiting[0]
            if ready[i] > r.nb:
                break
            need = float(p_eff[i])
            if r.kv_used + need > cap and r.running:
                break
            r.waiting.popleft()
            r.running.append(i)
            kv[i] = need
            r.kv_used += need
            k = int(min(p_eff[i], budget))
            chunks.append((i, k))
            budget -= k
            admitted = True
        if not dec and not chunks:
            r.busy = False
            return
        r.busy = True
        r.version += 1
        r.t0 = t
        r.dec = dec
        r.chunks = chunks
        ctx = float(sum(int(prompt[i]) + int(gen[i]) for i in dec))
        if chunks or admitted:
            pf = sum(k for _, k in chunks)
            att = sum(k * (done_pf[i] + k / 2.0) for i, k in chunks)
            r.kind = "single"
            r.t_end = t + model.step_time(len(dec), ctx, pf, att)
        else:
            nd = len(dec)
            K = int(min(output[i] - gen[i] for i in dec))
            if math.isfinite(cap):
                K = min(K, max(1, int((cap - r.kv_used) // nd)))
            if r.waiting and len(r.running) < S and ready[r.waiting[0]] > r.nb:
                K = min(K, int(ready[r.waiting[0]] - r.nb))      # stop when the head becomes schedulable
            K = max(1, K)
            r.kind = "macro"
            r.A = model.b0 + model.b_dec * nd + model.b_ctx * ctx / 1e3
            r.c = model.b_ctx * nd / 1e3
            r.K = K
            r.t_end = t + _cum(r.A, r.c, K)
        heapq.heappush(heap, (r.t_end, ri, r.version))

    def finish(r: _Replica, t: float) -> None:
        r.nb += r.K if r.kind == "macro" else 1
        if r.kind == "macro":
            steps = r.K
            for i in r.dec:
                gen[i] += steps
                kv[i] += steps
                r.kv_used += steps
        else:
            for i in r.dec:
                gen[i] += 1
                kv[i] += 1
                r.kv_used += 1
            for i, k in r.chunks:
                done_pf[i] += k
                if done_pf[i] >= p_eff[i]:
                    gen[i] += 1                  # the prefill step samples a token
                    kv[i] += 1
                    r.kv_used += 1
                    if math.isnan(t_first[i]):
                        t_first[i] = t
        keep = []
        for i in r.running:
            if done_pf[i] >= p_eff[i] and gen[i] >= output[i]:
                t_done[i] = t
                r.kv_used -= kv[i]
                kv[i] = 0.0
            else:
                keep.append(i)
        r.running = keep
        r.busy = False

    def interrupt(r: _Replica, ri: int, i: int, t: float) -> None:
        """Set request i's ready boundary; cut a macro step at the boundary where it becomes schedulable."""
        lag = lag_int + (1 if lag_frac > 0 and lag_rng.random() < lag_frac else 0)
        if r.kind != "macro":
            ready[i] = r.nb + 1 + lag
            return
        dt = t - r.t0
        A, c = r.A, r.c
        if c > 0:
            j = int(math.floor((-(A - c / 2) + math.sqrt((A - c / 2) ** 2 + 2 * c * dt)) / c))
        else:
            j = int(math.floor(dt / A)) if A > 0 else r.K
        j = max(0, j)
        while j > 0 and _cum(A, c, j) > dt:
            j -= 1
        while _cum(A, c, j + 1) <= dt:
            j += 1
        ready[i] = r.nb + j + 1 + lag
        new_k = min(r.K, j + 1 + lag)
        if new_k < r.K:
            r.K = new_k
            r.t_end = r.t0 + _cum(A, c, new_k)
            r.version += 1
            heapq.heappush(heap, (r.t_end, ri, r.version))

    ia = 0
    t = 0.0
    while True:
        t_arr = arrival[order[ia]] if ia < n else math.inf
        while heap and heap[0][2] != reps[heap[0][1]].version:
            heapq.heappop(heap)
        t_step = heap[0][0] if heap else math.inf
        if t_arr == math.inf and t_step == math.inf:
            break
        if t_arr <= t_step:
            i = int(order[ia])
            ia += 1
            t = t_arr
            if eng.router == "round_robin":
                ri = rr % len(reps)
                rr += 1
            else:
                ri = min(range(len(reps)), key=lambda k: (len(reps[k].running) + len(reps[k].waiting), k))
            r = reps[ri]
            r.waiting.append(i)
            if not r.busy:
                ready[i] = r.nb
                plan(r, ri, t)
            else:
                interrupt(r, ri, i, t)
        else:
            _, ri, _ = heapq.heappop(heap)
            r = reps[ri]
            t = r.t_end
            finish(r, t)
            plan(r, ri, t)

    ttft = t_first - arrival + model.overhead
    e2e = t_done - arrival + model.overhead
    return SimResult(arrival, prompt, output, ttft, e2e, preempt, float(np.nanmax(t_done)) if n else 0.0)


def poisson_arrivals(rate: float, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.cumsum(rng.exponential(1.0 / rate, size=n))


def saturation_rate(model: StepModel, eng: EngineConfig, prompt, output, n: int = 2000, seed: int = 0) -> float:
    """Maximum sustainable request rate (req/s).

    Every request is queued at time 0 and completions are counted over the middle 60% of the run,
    when the engine is full. ``n`` is raised to at least six full batches per replica.
    """
    rng = np.random.default_rng(seed)
    n = int(max(n, 6 * eng.max_num_seqs * max(1, eng.replicas)))
    idx = rng.integers(len(prompt), size=n)
    res = simulate(np.zeros(n), np.asarray(prompt)[idx], np.asarray(output)[idx], model, eng)
    done = np.sort(res.e2e - model.overhead)
    a, b = 0.2 * done[-1], 0.8 * done[-1]
    k = np.searchsorted(done, b) - np.searchsorted(done, a)
    return float(k / (b - a)) if b > a else float("inf")
