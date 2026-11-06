"""Command line: ``vllm-headroom profile | analyze | whatif | demo | mock-server``."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from . import __version__
from .analysis import (backtest, fit_from_profile, headroom_analysis, max_rate, parse_slo, slo_label,
                       with_changes)
from .profiler import ProfileConfig, Profiler, run_profile
from .report import render
from .sim import EngineConfig
from .workload import Workload, load_jsonl, synthetic


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _changes(items: list[str]) -> dict:
    out = {}
    for s in items or []:
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _workload(args, profile: dict) -> Workload:
    if getattr(args, "workload", None):
        return load_jsonl(args.workload)
    if getattr(args, "prompt_mean", None) and getattr(args, "output_mean", None):
        return synthetic(args.prompt_mean, args.output_mean, args.cv,
                         max_len=profile.get("server", {}).get("max_model_len"))
    return Workload.from_dict(profile["validation_workload"])


def _profile_cfg(args) -> ProfileConfig:
    wl = load_jsonl(args.workload) if args.workload else None
    return ProfileConfig(url=args.url, model=args.model, api_key=args.api_key, max_concurrency=args.max_concurrency,
                         max_num_seqs=args.max_num_seqs, max_num_batched_tokens=args.max_num_batched_tokens,
                         kv_tokens=args.kv_tokens, token_ids=not args.text_prompts,
                         validation_seconds=args.validation_seconds, seed=args.seed, workload=wl)


def cmd_profile(args) -> None:
    prof = run_profile(_profile_cfg(args), _log)
    Path(args.output).write_text(json.dumps(prof), encoding="utf-8")
    _log(f"wrote {args.output}")


def analyze(profile: dict, args) -> tuple[str, dict]:
    slo = parse_slo(args.slo)
    eng = EngineConfig.from_dict(profile["engine"])
    base = _changes(args.base)
    if base:
        _, eng = with_changes(None, eng, base)
        profile = dict(profile, engine=eng.to_dict())
    wl = _workload(args, profile)
    _log("fitting step-time model")
    fit = fit_from_profile(profile, n_boot=max(args.draws, 1), seed=args.seed)
    _log(f"fit: {fit.model}")
    _log("latency curve and headroom")
    ana = headroom_analysis(fit, eng, wl, slo, args.current_rate, n=args.requests, draws=args.draws,
                            seed=args.seed, log=_log)
    _log("backtest on validation load")
    bt = backtest(profile, fit, eng, reps=min(20, args.draws + 1), seed=args.seed)
    whatif = []
    for spec in args.whatif or []:
        ch = _changes(spec.split(","))
        _, e2 = with_changes(fit.model, eng, ch)
        hr = max_rate(fit.model, e2, wl, slo, n=args.requests, seed=args.seed)
        whatif.append({"label": spec, "lam_sat": hr.lam_sat, "lam_max": hr.lam_max, "binding": hr.binding})
        _log(f"what-if {spec}: max {hr.lam_max:.3f} req/s")
    page = render(profile, fit, ana, bt, whatif, wl.summary())
    result = {"version": __version__, "fit": fit.model.to_dict(), "engine": eng.to_dict(), "workload": wl.summary(),
              "headroom": ana, "backtest": bt, "whatif": whatif}
    return page, result


def cmd_analyze(args) -> None:
    profile = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    page, result = analyze(profile, args)
    out = Path(args.output)
    out.write_text(page, encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    _print_summary(result)
    _log(f"wrote {out} and {out.with_suffix('.json')}")


def _print_summary(r: dict) -> None:
    h = r["headroom"]
    print(f"saturation:             {h['lam_sat']:.3f} req/s")
    if h["lam_max"] <= 0:
        print(f"the SLO is not met even at very low load ({h['binding']}): a single request already exceeds it")
    print(f"max load meeting SLO:   {h['lam_max']:.3f} req/s  (10-90%: {h['lam_max_band'][0]:.3f}-"
          f"{h['lam_max_band'][1]:.3f}; binding: {h['binding']})")
    if h.get("current_rate"):
        print(f"headroom vs current:    {h['headroom_pct']:+.0f}%  (10-90%: {h['headroom_pct_band'][0]:+.0f}% to "
              f"{h['headroom_pct_band'][1]:+.0f}%)")
    for w in r.get("whatif", []):
        print(f"what-if {w['label']}: max {w['lam_max']:.3f} req/s ({w['binding']})")


def cmd_whatif(args) -> None:
    profile = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    slo = parse_slo(args.slo)
    eng = EngineConfig.from_dict(profile["engine"])
    wl = _workload(args, profile)
    fit = fit_from_profile(profile, n_boot=0, seed=args.seed)
    rows = [("baseline", eng)] + [(spec, with_changes(fit.model, eng, _changes(spec.split(",")))[1])
                                  for spec in args.change]
    base = None
    print(f"{'change':40s} {'saturation':>12s} {'max@SLO':>10s} {'vs base':>8s}  binding")
    for label, e in rows:
        hr = max_rate(fit.model, e, wl, slo, n=args.requests, seed=args.seed)
        base = hr.lam_max if base is None else base
        d = f"{100 * (hr.lam_max / base - 1):+.0f}%" if base else "-"
        print(f"{label:40s} {hr.lam_sat:10.3f}/s {hr.lam_max:8.3f}/s {d:>8s}  {hr.binding}")


def cmd_demo(args) -> None:
    from . import mock_server

    async def go():
        server, port, task = await mock_server.start(max_num_seqs=64, budget=4096, kv_tokens=300_000)
        _log(f"started a simulated server on port {port} (no GPU, not a real model)")
        cfg = ProfileConfig(url=f"http://127.0.0.1:{port}", max_concurrency=64, decode_tokens=96,
                            decode_prompts=(256, 1024), prefill_prompts=(128, 512, 1024, 2048, 4096),
                            prefill_reps=2, validation_fracs=(0.4, 0.7), validation_seconds=8,
                            max_num_seqs=64, max_num_batched_tokens=4096)
        prof = await Profiler(cfg, _log).run()
        server.close()
        task.cancel()
        return prof

    prof = asyncio.run(go())
    prof["server"]["model"] = "mock-model (simulated)"
    Path(args.output).with_suffix(".profile.json").write_text(json.dumps(prof), encoding="utf-8")
    ns = argparse.Namespace(slo=args.slo, base=[], workload=None, prompt_mean=None, output_mean=None, cv=1.0,
                            current_rate=args.current_rate, requests=1200, draws=12, seed=0,
                            whatif=["max_num_seqs=128", "replicas=2"])
    page, result = analyze(prof, ns)
    Path(args.output).write_text(page, encoding="utf-8")
    _print_summary(result)
    _log(f"wrote {args.output}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="vllm-headroom", description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("profile", help="profile a running vLLM server (about 3-4 minutes)")
    p.add_argument("--url", default="http://localhost:8000", help="server base URL")
    p.add_argument("--model", help="model id (default: first model from /v1/models)")
    p.add_argument("--api-key")
    p.add_argument("--max-concurrency", type=int, default=256, help="largest decode wave")
    p.add_argument("--max-num-seqs", type=int, help="the server's --max-num-seqs")
    p.add_argument("--max-num-batched-tokens", type=int, help="the server's --max-num-batched-tokens")
    p.add_argument("--kv-tokens", type=int, help="KV cache capacity in tokens (default: read from /metrics)")
    p.add_argument("--workload", help="JSONL of {prompt_tokens, output_tokens} for the validation phase")
    p.add_argument("--validation-seconds", type=float, default=25.0)
    p.add_argument("--text-prompts", action="store_true", help="send text prompts instead of token ids")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-o", "--output", default="profile.json")
    p.set_defaults(func=cmd_profile)

    def common(q):
        q.add_argument("profile", help="profile.json from `vllm-headroom profile`")
        q.add_argument("--slo", action="append", default=None,
                       help="latency SLO, repeatable: ttft_p95=2, e2e_p99=30, tpot_p95=0.1 (seconds)")
        q.add_argument("--workload", help="JSONL of {prompt_tokens, output_tokens} (default: validation workload)")
        q.add_argument("--prompt-mean", type=float, help="synthetic workload: mean prompt tokens")
        q.add_argument("--output-mean", type=float, help="synthetic workload: mean output tokens")
        q.add_argument("--cv", type=float, default=1.0, help="synthetic workload: coefficient of variation")
        q.add_argument("--requests", type=int, default=1500, help="simulated requests per load point")
        q.add_argument("--seed", type=int, default=0)

    a = sub.add_parser("analyze", help="fit, compute headroom, backtest and write an HTML report")
    common(a)
    a.add_argument("--current-rate", type=float, help="current arrival rate (req/s) to report headroom against")
    a.add_argument("--base", action="append", default=[],
                   help="override the profiled configuration, e.g. replicas=4 or max_num_seqs=128")
    a.add_argument("--whatif", action="append", help="comma-separated changes, e.g. max_num_seqs=512,replicas=2")
    a.add_argument("--draws", type=int, default=20, help="bootstrap refits for the uncertainty range")
    a.add_argument("-o", "--output", default="headroom.html")
    a.set_defaults(func=cmd_analyze)

    w = sub.add_parser("whatif", help="compare configurations quickly (point estimates)")
    common(w)
    w.add_argument("--change", action="append", required=True,
                   help="comma-separated changes: max_num_seqs, max_num_batched_tokens, replicas, kv_scale, router")
    w.set_defaults(func=cmd_whatif)

    d = sub.add_parser("demo", help="profile a simulated server and write a sample report (about 1 minute)")
    d.add_argument("--slo", action="append", default=None)
    d.add_argument("--current-rate", type=float, default=5.0)
    d.add_argument("-o", "--output", default="demo_report.html")
    d.set_defaults(func=cmd_demo)

    m = sub.add_parser("mock-server", help="run the simulated OpenAI-compatible server")
    m.add_argument("--host", default="127.0.0.1")
    m.add_argument("--port", type=int, default=8001)
    m.add_argument("--max-num-seqs", type=int, default=64)
    m.set_defaults(func=lambda a: __import__("vllm_headroom.mock_server", fromlist=["x"]).serve_forever(
        a.host, a.port, max_num_seqs=a.max_num_seqs))
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    if getattr(args, "slo", "absent") is None:
        args.slo = ["ttft_p95=2", "tpot_p95=0.1"]
    args.func(args)


if __name__ == "__main__":
    main()
