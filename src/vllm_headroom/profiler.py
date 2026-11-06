"""Short profiling run against an OpenAI-compatible vLLM endpoint.

Three phases, about three minutes in total with the defaults:

1. **Prefill sweep.** Single requests on an idle server with prompts of increasing length. Time to
   first token identifies the prefill cost and the fixed client-visible overhead.
2. **Decode waves.** ``c`` requests sent together with identical lengths, for ``c`` = 1, 2, 4, ...
   They prefill together and then decode in lockstep, so inter-token gaps give the step time for a
   known batch size and context. Gaps that overlap another request's prefill are discarded.
3. **Validation.** Open-loop Poisson load at a few fractions of the estimated saturation rate, with
   the target workload. These requests are never used for fitting; ``analyze`` uses them to check
   the model's predictions (a backtest).

Prompts are sent as random token ids, so prefix caching cannot shorten prefill and prompt lengths
are exact. ``ignore_eos`` makes output lengths exact. Run this against a replica that serves no
other traffic.
"""
from __future__ import annotations

import asyncio
import json
import platform
import re
import time
from dataclasses import asdict, dataclass, field

import httpx
import numpy as np

from . import __version__
from .fit import fit_profile
from .sim import EngineConfig, saturation_rate
from .workload import Workload, synthetic


@dataclass
class ProfileConfig:
    url: str = "http://localhost:8000"
    model: str | None = None
    api_key: str | None = None
    max_concurrency: int = 256
    decode_prompts: tuple = (256, 2048)
    decode_tokens: int = 192
    prefill_prompts: tuple = (128, 512, 1024, 2048, 4096, 8192)
    prefill_reps: int = 3
    validation_fracs: tuple = (0.3, 0.5, 0.7)
    validation_seconds: float = 25.0
    max_num_seqs: int | None = None
    max_num_batched_tokens: int | None = None
    kv_tokens: int | None = None
    token_ids: bool = True
    request_timeout: float = 600.0
    seed: int = 0
    workload: Workload | None = field(default=None, repr=False)


def _now() -> float:
    return time.perf_counter()


def _parse_metrics(text: str) -> dict:
    out: dict = {}
    m = re.search(r'[{,]num_gpu_blocks="(\d+)"', text)
    b = re.search(r'[{,]block_size="(\d+)"', text)
    s = re.search(r'[{,]kv_cache_size_tokens="(\d+)"', text)
    if m and b:
        out["num_gpu_blocks"] = int(m.group(1))
        out["block_size"] = int(b.group(1))
        out["kv_tokens"] = int(m.group(1)) * int(b.group(1))
    if s:
        out["kv_tokens"] = int(s.group(1))
    for name in ("num_requests_running", "num_requests_waiting"):
        g = re.search(rf"^vllm:{name}(?:{{[^}}]*}})?\s+([0-9.eE+-]+)", text, re.M)
        if g:
            out[name] = float(g.group(1))
    return out


class Profiler:
    def __init__(self, cfg: ProfileConfig, log=print):
        self.cfg = cfg
        self.log = log
        self.rng = np.random.default_rng(cfg.seed)
        self.base = cfg.url.rstrip("/")
        if self.base.endswith("/v1"):
            self.base = self.base[:-3]
        headers = {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
        self.client = httpx.AsyncClient(headers=headers, timeout=httpx.Timeout(cfg.request_timeout, connect=30.0),
                                        limits=limits)
        self.model = cfg.model
        self.max_model_len: int | None = None
        self.metrics_samples: list = []

    async def _server_info(self) -> dict:
        r = await self.client.get(self.base + "/v1/models")
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            raise RuntimeError("/v1/models returned no models")
        entry = next((d for d in data if d.get("id") == self.model), data[0]) if self.model else data[0]
        self.model = entry["id"]
        self.max_model_len = entry.get("max_model_len")
        info = {"model": self.model, "max_model_len": self.max_model_len}
        try:
            mr = await self.client.get(self.base + "/metrics", timeout=10)
            if mr.status_code == 200:
                info["metrics"] = _parse_metrics(mr.text)
        except httpx.HTTPError:
            info["metrics"] = None
        return info

    def _prompt(self, n: int):
        if self.cfg.token_ids:
            return self.rng.integers(1000, 20000, size=int(n)).tolist()
        return " ".join(["hello"] * int(n))

    async def request(self, prompt_tokens: int, max_tokens: int, on_first: asyncio.Event | None = None) -> dict:
        body = {
            "model": self.model, "prompt": self._prompt(prompt_tokens), "max_tokens": int(max_tokens),
            "stream": True, "ignore_eos": True, "temperature": 0.0,
            "stream_options": {"include_usage": True},
        }
        t_send = _now()
        times: list[float] = []
        usage = None
        err = None
        try:
            async with self.client.stream("POST", self.base + "/v1/completions", json=body) as r:
                if r.status_code != 200:
                    err = f"HTTP {r.status_code}: {(await r.aread())[:300]!r}"
                else:
                    async for line in r.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        t = _now()
                        d = json.loads(payload)
                        if d.get("usage"):
                            usage = d["usage"]
                        if d.get("choices"):
                            times.append(t)
                            if on_first is not None and len(times) == 1:
                                on_first.set()
        except httpx.HTTPError as e:
            err = repr(e)
        return {
            "t_send": t_send, "times": times, "err": err,
            "prompt_tokens": int(usage["prompt_tokens"]) if usage else int(prompt_tokens),
            "output_tokens": int(usage["completion_tokens"]) if usage else len(times),
        }

    async def _poll_metrics(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                mr = await self.client.get(self.base + "/metrics", timeout=5)
                if mr.status_code == 200:
                    m = _parse_metrics(mr.text)
                    m["t"] = _now()
                    self.metrics_samples.append(m)
            except httpx.HTTPError:
                pass
            try:
                await asyncio.wait_for(stop.wait(), 0.5)
            except asyncio.TimeoutError:
                pass

    def _clip_prompt(self, p: int, out: int) -> int:
        if self.max_model_len:
            return int(max(1, min(p, self.max_model_len - out - 8)))
        return int(p)

    async def run(self) -> dict:
        cfg = self.cfg
        info = await self._server_info()
        self.log(f"server: model={self.model} max_model_len={self.max_model_len}")
        await asyncio.gather(*[self.request(32, 8) for _ in range(4)])    # warm-up

        # 1. prefill sweep
        prefill = []
        prompts = sorted({self._clip_prompt(p, 2) for p in cfg.prefill_prompts})
        for p in prompts:
            for _ in range(cfg.prefill_reps):
                r = await self.request(p, 2)
                if r["err"] or not r["times"]:
                    raise RuntimeError(f"prefill request failed: {r['err']}")
                prefill.append({"prompt_tokens": r["prompt_tokens"], "ttft": r["times"][0] - r["t_send"]})
        self.log(f"prefill sweep: {len(prefill)} requests, prompts {prompts}")

        # 2. decode waves
        levels = []
        c = 1
        while c <= cfg.max_concurrency:
            levels.append(c)
            c *= 2
        if levels[-1] != cfg.max_concurrency:
            levels.append(cfg.max_concurrency)
        stop = asyncio.Event()
        poller = asyncio.create_task(self._poll_metrics(stop))
        waves = []
        kv_known = (info.get("metrics") or {}).get("kv_tokens") or cfg.kv_tokens
        for c in levels:
            for dp in sorted({self._clip_prompt(p, cfg.decode_tokens) for p in cfg.decode_prompts}):
                if kv_known and c * (dp + cfg.decode_tokens) > 0.9 * kv_known and dp != min(cfg.decode_prompts):
                    continue        # a long-prompt wave this wide would not fit in KV; skip it
                t0 = _now()
                rs = await asyncio.gather(*[self.request(dp, cfg.decode_tokens) for _ in range(c)])
                bad = [r for r in rs if r["err"]]
                if bad:
                    raise RuntimeError(f"decode wave c={c}: {len(bad)} failed, first: {bad[0]['err']}")
                waves.append({"concurrency": c, "prompt": dp, "requests": rs})
                self.log(f"decode wave c={c} prompt={dp}: {_now() - t0:.1f} s")
        stop.set()
        await poller

        # 2b. admission probes: short requests sent while a wave decodes measure how long a new request
        # waits before the engine schedules it
        probes = []
        _, _, seen = extract_decode(waves)
        seats = cfg.max_num_seqs or seen or cfg.max_concurrency
        for c in sorted({max(1, min(16, seats // 4)), max(1, min(64, seats // 2))}):   # leave free seats
            _, rows = await self._probe_wave(c, self._clip_prompt(min(cfg.decode_prompts), 512), 512)
            probes.extend(rows)         # probe waves are not used for decode fitting (probes prefill inside)
            self.log(f"admission probes during a c={c} wave: {len(rows)} usable")

        decode_obs, decode_group, observed_max = extract_decode(waves)
        self.log(f"decode: {len(decode_obs)} clean observations, max concurrent decoders seen {observed_max}")

        # engine configuration
        eng, eng_src = self._engine_config(info, levels, observed_max)

        # 3. validation (open loop) at fractions of the predicted saturation rate
        wl = cfg.workload or synthetic(512, 128, 1.0, max_len=self.max_model_len, seed=cfg.seed)
        fitres = fit_profile([o[:3] for o in decode_obs], [(p["ttft"], p["prompt_tokens"]) for p in prefill],
                             decode_group, list(range(len(prefill))), eng.max_num_batched_tokens, n_boot=0)
        lam_sat = saturation_rate(fitres.model, eng, wl.prompt, wl.output, n=1000)
        self.log(f"preliminary saturation estimate: {lam_sat:.2f} req/s for the validation workload")
        validation = []
        for frac in cfg.validation_fracs:
            rate = frac * lam_sat
            n = max(5, int(rate * cfg.validation_seconds))
            gaps = self.rng.exponential(1.0 / rate, size=n)
            offs = np.cumsum(gaps) - gaps[0]
            p, o = wl.sample(n, self.rng)
            t0 = _now()

            async def fire(delay, pt, ot):
                await asyncio.sleep(max(0.0, t0 + delay - _now()))
                return await self.request(self._clip_prompt(int(pt), int(ot)), int(ot))

            rs = await asyncio.gather(*[fire(d, pt, ot) for d, pt, ot in zip(offs, p, o)])
            rows = [{"t_send": r["t_send"] - t0, "ttft": (r["times"][0] - r["t_send"]) if r["times"] else None,
                     "e2e": (r["times"][-1] - r["t_send"]) if r["times"] else None,
                     "prompt_tokens": r["prompt_tokens"], "output_tokens": r["output_tokens"], "err": r["err"]}
                    for r in rs]
            validation.append({"fraction": frac, "rate": rate, "duration": float(offs[-1]) if n else 0.0,
                               "requests": rows})
            self.log(f"validation {frac:.0%} of saturation ({rate:.2f} req/s): {len(rows)} requests, "
                     f"{sum(1 for r in rows if r['err'])} errors")

        await self.client.aclose()
        return {
            "tool": "vllm-headroom", "version": __version__, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "server": info, "platform": platform.platform(),
            "config": {k: v for k, v in asdict(cfg).items() if k not in ("workload", "api_key")},
            "engine": eng.to_dict(), "engine_source": eng_src,
            "prefill": prefill,
            "decode_obs": [list(o) for o in decode_obs], "decode_group": list(map(int, decode_group)),
            "probes": probes,
            "observed_max_decoders": int(observed_max),
            "metrics_samples": self.metrics_samples[-200:],
            "validation_workload": wl.to_dict(),
            "validation": validation,
        }

    async def _probe_wave(self, c: int, prompt: int, out: int, n_probe: int = 15, probe_prompt: int = 128):
        started = asyncio.Event()
        wave_task = asyncio.ensure_future(asyncio.gather(*[
            self.request(prompt, out, on_first=started if k == 0 else None) for k in range(c)]))
        await started.wait()
        await asyncio.sleep(0.05)
        raw = []
        for _ in range(n_probe):
            if wave_task.done():
                break
            r = await self.request(probe_prompt, 2)
            if not r["err"] and r["times"]:
                raw.append(r)
            await asyncio.sleep(float(self.rng.uniform(0.0, 0.05)))
        rs = await wave_task
        T = [np.asarray(r["times"]) for r in rs if r["times"]]
        first = np.array([t[0] for t in T])
        last = np.array([t[-1] for t in T])
        prompts = np.array([r["prompt_tokens"] for r in rs if r["times"]])
        rows = []
        for r in raw:
            a, b = r["t_send"], r["times"][0]
            if np.any((first > a) & (first <= b)):
                continue                       # a wave request was prefilled meanwhile
            alive = (first <= a) & (last >= b)
            n_dec = int(alive.sum())
            if n_dec == 0:
                continue
            ctx = float(sum(prompts[k] + np.searchsorted(T[k], a) for k in np.nonzero(alive)[0]))
            rows.append({"ttft": b - a, "prompt_tokens": r["prompt_tokens"], "n_dec": n_dec, "ctx": ctx})
        return {"concurrency": c, "prompt": prompt, "requests": rs}, rows

    def _engine_config(self, info: dict, levels: list, observed_max: int) -> tuple[EngineConfig, dict]:
        cfg = self.cfg
        src = {}
        met = info.get("metrics") or {}
        if cfg.kv_tokens:
            kv, src["kv_capacity_tokens"] = float(cfg.kv_tokens), "user"
        elif met.get("kv_tokens"):
            kv, src["kv_capacity_tokens"] = float(met["kv_tokens"]), "server /metrics (num_gpu_blocks x block_size)"
        else:
            kv, src["kv_capacity_tokens"] = float("inf"), "unknown: treated as unlimited"
        if cfg.max_num_seqs:
            mns, src["max_num_seqs"] = int(cfg.max_num_seqs), "user"
        elif observed_max < max(levels):
            mns, src["max_num_seqs"] = int(observed_max), "inferred from the largest decode wave (or KV-limited)"
        else:
            mns, src["max_num_seqs"] = 256, "assumed vLLM default (pass --max-num-seqs)"
        if cfg.max_num_batched_tokens:
            mbt, src["max_num_batched_tokens"] = int(cfg.max_num_batched_tokens), "user"
        else:
            mbt, src["max_num_batched_tokens"] = 8192, "assumed (pass --max-num-batched-tokens)"
        return EngineConfig(max_num_seqs=mns, max_num_batched_tokens=mbt, kv_capacity_tokens=kv), src


def extract_decode(waves: list, block: int = 16) -> tuple[list, list, int]:
    """Clean decode observations from lockstep waves.

    For each request, consecutive inter-chunk gaps are grouped into blocks of ``block`` gaps during
    which the number of decoding sequences is constant and no request in the wave receives its first
    token (no prefill in the step). Each block gives one row: (mean gap, decoders, mean context).
    """
    obs, group = [], []
    observed_max = 0
    for wi, w in enumerate(waves):
        rs = [r for r in w["requests"] if r["times"] and len(r["times"]) == r["output_tokens"]]
        if not rs:
            continue
        T = [np.asarray(r["times"]) for r in rs]
        first = np.array([t[0] for t in T])
        last = np.array([t[-1] for t in T])
        prompt = np.array([r["prompt_tokens"] for r in rs])
        grid = np.sort(np.concatenate(T))
        if len(grid):
            live = np.array([np.sum((first <= g) & (last >= g)) for g in grid[:: max(1, len(grid) // 400)]])
            observed_max = max(observed_max, int(live.max()))
        for i, t in enumerate(T):
            if len(t) < 3:
                continue
            seg_start = None
            seg_n = None
            for j in range(1, len(t)):
                a, b = t[j - 1], t[j]
                prefill_in = np.any((first > a) & (first <= b))
                n_dec = int(np.sum((first <= a) & (last >= b)))
                if prefill_in or n_dec == 0:
                    seg_start = None
                    continue
                if seg_start is None or n_dec != seg_n:
                    seg_start, seg_n = j - 1, n_dec
                if j - seg_start == block:
                    ta, tb = t[seg_start], t[j]
                    mid = 0.5 * (ta + tb)
                    alive = (first <= mid) & (last >= mid)
                    ctx = float(sum(prompt[k] + np.searchsorted(T[k], mid) for k in np.nonzero(alive)[0]))
                    obs.append(((tb - ta) / block, seg_n, ctx))
                    group.append(wi)
                    seg_start = j
    return obs, group, observed_max


def run_profile(cfg: ProfileConfig, log=print) -> dict:
    async def go():
        return await Profiler(cfg, log).run()
    return asyncio.run(go())
