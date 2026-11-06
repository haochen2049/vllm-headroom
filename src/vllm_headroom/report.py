"""Self-contained HTML report (inline SVG charts, no external assets)."""
from __future__ import annotations

import html
import math

import numpy as np

CSS = """
:root{--bg:#fff;--fg:#1d2433;--mut:#5b6475;--line:#d9dde5;--card:#f6f7f9;--a:#2563eb;--b:#d97706;--ok:#15803d;--bad:#b91c1c}
@media (prefers-color-scheme:dark){:root{--bg:#12151b;--fg:#e6e9ef;--mut:#9aa3b2;--line:#2c323d;--card:#1a1e26;--a:#60a5fa;--b:#fbbf24;--ok:#4ade80;--bad:#f87171}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:980px;margin:0 auto;padding:28px 16px 60px}h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:34px 0 10px}
.mut{color:var(--mut)}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:18px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}.card .v{font-size:24px;font-weight:650}
.card .l{font-size:13px;color:var(--mut)}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left}th{color:var(--mut);font-weight:600}
svg{width:100%;height:auto;display:block}.ax{stroke:var(--line)}.tk{fill:var(--mut);font-size:11px}
.note{background:var(--card);border-left:3px solid var(--b);padding:10px 14px;border-radius:6px;font-size:14px}
code{font-size:13px}.ok{color:var(--ok)}.bad{color:var(--bad)}
"""


def _nice(lo: float, hi: float, n: int = 5) -> list[float]:
    if not (math.isfinite(lo) and math.isfinite(hi)) or hi <= lo:
        hi = lo + 1
    step = 10 ** math.floor(math.log10((hi - lo) / n))
    for m in (1, 2, 5, 10):
        if (hi - lo) / (step * m) <= n:
            step *= m
            break
    start = math.floor(lo / step) * step
    return [start + i * step for i in range(int((hi - start) / step) + 2) if start + i * step <= hi * 1.0001]


def _fmt(v: float) -> str:
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 100:
        return f"{v:.0f}"
    if a >= 1:
        return f"{v:.3g}"
    return f"{v:.2g}"


class Chart:
    def __init__(self, xlabel: str, ylabel: str, xr, yr, w=900, h=320):
        self.w, self.h, self.ml, self.mr, self.mt, self.mb = w, h, 64, 16, 14, 44
        self.xr, self.yr = xr, yr
        self.parts: list[str] = []
        self.xl, self.yl = xlabel, ylabel
        self.hys: list[float] = []  # y of horizontal reference lines, kept clear of vline labels

    def X(self, x):
        return self.ml + (x - self.xr[0]) / (self.xr[1] - self.xr[0]) * (self.w - self.ml - self.mr)

    def Y(self, y):
        y = min(max(y, self.yr[0]), self.yr[1] * 1.2)
        return self.h - self.mb - (y - self.yr[0]) / (self.yr[1] - self.yr[0]) * (self.h - self.mt - self.mb)

    def line(self, xs, ys, color, width=2, dash=None):
        pts = [(self.X(x), self.Y(y)) for x, y in zip(xs, ys) if y is not None and math.isfinite(y)]
        if len(pts) < 2:
            return
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        da = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="{width}"{da} points="{d}"/>')

    def band(self, xs, lo, hi, color):
        pts = [(x, a, b) for x, a, b in zip(xs, lo, hi) if a is not None and b is not None]
        if len(pts) < 2:
            return
        up = " ".join(f"{self.X(x):.1f},{self.Y(b):.1f}" for x, a, b in pts)
        dn = " ".join(f"{self.X(x):.1f},{self.Y(a):.1f}" for x, a, b in reversed(pts))
        self.parts.append(f'<polygon fill="{color}" fill-opacity="0.18" stroke="none" points="{up} {dn}"/>')

    def hline(self, y, color, label=""):
        yy = self.Y(y)
        self.hys.append(yy)
        self.parts.append(f'<line x1="{self.ml}" x2="{self.w - self.mr}" y1="{yy:.1f}" y2="{yy:.1f}" stroke="{color}" '
                          f'stroke-dasharray="5 4"/>')
        if label:
            self.parts.append(f'<text class="tk" x="{self.ml + 6}" y="{yy - 4:.1f}">{html.escape(label)}</text>')

    def vline(self, x, color, label="", row=0):
        xx = self.X(x)
        self.parts.append(f'<line y1="{self.mt}" y2="{self.h - self.mb}" x1="{xx:.1f}" x2="{xx:.1f}" stroke="{color}" '
                          f'stroke-dasharray="3 3"/>')
        if label:
            right = xx > self.w - 220
            tx, anchor = (xx - 4, "end") if right else (xx + 4, "start")
            ty = self.mt + 12 + 14 * row
            while any(ty - 12 <= hy <= ty + 4 for hy in self.hys):
                ty += 14
            self.parts.append(f'<text class="tk" x="{tx:.1f}" y="{ty}" text-anchor="{anchor}">'
                              f'{html.escape(label)}</text>')

    def dots(self, xs, ys, color, r=3):
        for x, y in zip(xs, ys):
            if y is not None and math.isfinite(y):
                self.parts.append(f'<circle cx="{self.X(x):.1f}" cy="{self.Y(y):.1f}" r="{r}" fill="{color}"/>')

    def errbar(self, x, lo, hi, color):
        self.parts.append(f'<line x1="{self.X(x):.1f}" x2="{self.X(x):.1f}" y1="{self.Y(lo):.1f}" y2="{self.Y(hi):.1f}" '
                          f'stroke="{color}" stroke-width="2"/>')

    def svg(self) -> str:
        ax = []
        for t in _nice(*self.xr):
            if self.xr[0] <= t <= self.xr[1]:
                ax.append(f'<line class="ax" x1="{self.X(t):.1f}" x2="{self.X(t):.1f}" y1="{self.mt}" y2="{self.h - self.mb}"/>'
                          f'<text class="tk" x="{self.X(t):.1f}" y="{self.h - self.mb + 16}" text-anchor="middle">{_fmt(t)}</text>')
        for t in _nice(*self.yr):
            if self.yr[0] <= t <= self.yr[1]:
                ax.append(f'<line class="ax" x1="{self.ml}" x2="{self.w - self.mr}" y1="{self.Y(t):.1f}" y2="{self.Y(t):.1f}"/>'
                          f'<text class="tk" x="{self.ml - 6}" y="{self.Y(t) + 4:.1f}" text-anchor="end">{_fmt(t)}</text>')
        ax.append(f'<text class="tk" x="{(self.ml + self.w - self.mr) / 2:.0f}" y="{self.h - 8}" text-anchor="middle">'
                  f'{html.escape(self.xl)}</text>')
        ax.append(f'<text class="tk" transform="translate(14,{(self.mt + self.h - self.mb) / 2:.0f}) rotate(-90)" '
                  f'text-anchor="middle">{html.escape(self.yl)}</text>')
        return (f'<svg viewBox="0 0 {self.w} {self.h}" role="img">{"".join(ax)}{"".join(self.parts)}</svg>')


def _card(v: str, l: str) -> str:
    return f'<div class="card"><div class="v">{v}</div><div class="l">{l}</div></div>'


def render(profile: dict, fit, analysis: dict, backtest_rows: list, whatif: list | None, wl_summary: dict) -> str:
    srv = profile.get("server", {})
    eng = profile["engine"]
    src = profile.get("engine_source", {})
    a = analysis
    h = [f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
         f"<title>vLLM headroom report</title><style>{CSS}</style></head><body><main>"]
    h.append(f"<h1>vLLM headroom report</h1><div class=mut>model <code>{html.escape(str(srv.get('model')))}</code> · "
             f"profiled {html.escape(profile.get('created', ''))} · vllm-headroom {html.escape(profile.get('version', ''))}</div>")

    slo_txt = ", ".join(f"{k.replace('_p', ' p')} ≤ {v:g} s" for k, v in a["slo"].items())
    cards = [_card(f"{a['lam_max']:.2f} req/s", f"max load meeting SLO ({slo_txt})<br>10–90% range "
                   f"{a['lam_max_band'][0]:.2f}–{a['lam_max_band'][1]:.2f}"),
             _card(f"{a['lam_sat']:.2f} req/s", "saturation (maximum sustainable throughput)"),
             _card(html.escape(a["binding"] or "-"), "SLO that binds first")]
    if a.get("current_rate"):
        cls = "ok" if a["headroom_pct"] >= 0 else "bad"
        cards.append(_card(f"<span class={cls}>{a['headroom_pct']:+.0f}%</span>",
                           f"headroom over current {a['current_rate']:g} req/s<br>10–90% range "
                           f"{a['headroom_pct_band'][0]:+.0f}% to {a['headroom_pct_band'][1]:+.0f}%"))
    h.append(f"<div class=cards>{''.join(cards)}</div>")
    if a["lam_max"] <= 0:
        h.append(f"<div class=note>The SLO ({html.escape(a['binding'])}) is not met even at very low load: with this "
                 "workload a request on an idle server already exceeds it. Loosen the SLO, shorten outputs or use "
                 "faster hardware or a smaller model.</div>")
    elif a["binding"] == "saturation":
        h.append("<div class=note>Every SLO holds right up to saturation, so throughput, not latency, is the limit. "
                 "Running this close to saturation leaves no room for bursts.</div>")
    h.append(f"<p class=mut>Workload: {html.escape(wl_summary['source'])}; prompt mean {wl_summary['prompt_mean']:.0f} "
             f"(p95 {wl_summary['prompt_p95']:.0f}) tokens, output mean {wl_summary['output_mean']:.0f} "
             f"(p95 {wl_summary['output_p95']:.0f}) tokens. Poisson arrivals. Replicas: {eng.get('replicas', 1)}.</p>")

    # latency vs load
    rates = a["rates"]
    for lab, ys in a["curve"].items():
        lim = a["slo"][lab]
        finite = [y for y in ys if y is not None and math.isfinite(y)] + [lim]
        ymax = min(max(finite) * 1.1, lim * 3)
        c = Chart("arrival rate (req/s)", f"{lab.replace('_p', ' p')} (s)", (0, max(rates) * 1.02), (0, ymax))
        lo = [b[0] for b in a["band"][lab]]
        hi = [b[1] for b in a["band"][lab]]
        c.band(rates, lo, hi, "var(--a)")
        c.line(rates, ys, "var(--a)")
        c.hline(lim, "var(--bad)", f"SLO {lim:g} s")
        c.vline(a["lam_max"], "var(--ok)", "max load meeting SLO", row=1)
        if a.get("current_rate"):
            c.vline(a["current_rate"], "var(--b)", "current")
        h.append(f"<h2>{lab.replace('_p', ' p')} versus load</h2>{c.svg()}"
                 f"<p class=mut>Line: fitted model. Band: 10–90% across bootstrap refits of the profile.</p>")

    if whatif:
        h.append("<h2>What-if</h2><table><tr><th>change</th><th>saturation</th><th>max load meeting SLO</th>"
                 "<th>binding</th><th>vs. baseline</th></tr>")
        base = a["lam_max"]
        for w in whatif:
            d = 100 * (w["lam_max"] / base - 1) if base > 0 else float("nan")
            h.append(f"<tr><td><code>{html.escape(w['label'])}</code></td><td>{w['lam_sat']:.2f} req/s</td>"
                     f"<td>{w['lam_max']:.2f} req/s</td><td>{html.escape(w['binding'])}</td><td>{d:+.0f}%</td></tr>")
        h.append("</table><p class=mut>Point estimates from the fitted model. Changes far outside the profiled range "
                 "(for example batch sizes larger than the largest decode wave) are extrapolations.</p>")

    # backtest
    if backtest_rows:
        h.append("<h2>Backtest on held-out validation load</h2><p class=mut>The validation requests were sent "
                 "open-loop after the fitting phases and were not used to fit. The simulator replays their send "
                 "times and sizes. Range: 10–90% across bootstrap refits and simulation seeds; it is shown in "
                 "red when the measurement falls outside it.</p>"
                 "<table><tr><th>load</th><th>metric</th><th>measured</th><th>predicted</th><th>range</th>"
                 "<th>error</th></tr>")
        for r in backtest_rows:
            inside = r["range"][0] <= r["measured"] <= r["range"][1]
            err = f"{100 * r['rel_error']:+.0f}%" if r["rel_error"] is not None else "-"
            h.append(f"<tr><td>{r['fraction']:.0%} ({r['rate']:.2f} req/s, n={r['n']})</td><td>{r['metric']}</td>"
                     f"<td>{r['measured']:.3f} s</td><td>{r['predicted']:.3f} s</td>"
                     f"<td class={'ok' if inside else 'bad'}>{r['range'][0]:.3f}–{r['range'][1]:.3f}</td><td>{err}</td></tr>")
        h.append("</table>")

    # fit diagnostics
    dec = np.asarray(profile["decode_obs"], float).reshape(-1, 3)
    if len(dec):
        m = fit.model
        c = Chart("sequences decoding together", "time per step (ms)", (0, dec[:, 1].max() * 1.05),
                  (0, float(np.percentile(dec[:, 0], 99)) * 1e3 * 1.15))
        show = dec[np.random.default_rng(0).permutation(len(dec))[:1200]]
        c.dots(show[:, 1], show[:, 0] * 1e3, "var(--mut)", 2)
        ns = np.linspace(1, dec[:, 1].max(), 40)
        for q, col in ((25, "var(--a)"), (75, "var(--b)")):
            per_seq = float(np.percentile(dec[:, 2] / dec[:, 1], q))
            c.line(ns, [(m.b0 + m.b_dec * n + m.b_ctx * n * per_seq / 1e3) * 1e3 for n in ns], col)
        h.append("<h2>Decode step time</h2>" + c.svg() +
                 "<p class=mut>Dots: measured mean gap per 16 tokens. Lines: fitted model at the 25th and 75th "
                 "percentile of context per sequence.</p>")
    pf = profile.get("prefill", [])
    if pf:
        P = np.array([p["prompt_tokens"] for p in pf])
        T = np.array([p["ttft"] for p in pf])
        c = Chart("prompt tokens", "time to first token, idle server (ms)", (0, P.max() * 1.05), (0, T.max() * 1e3 * 1.15))
        c.dots(P, T * 1e3, "var(--mut)", 3)
        m = fit.model
        B = int(eng["max_num_batched_tokens"])
        xs = np.linspace(1, P.max(), 60)
        c.line(xs, [(m.overhead + math.ceil(x / B) * m.b0 + m.b_pf * x + m.b_pfatt * x * x / 2e6) * 1e3 for x in xs], "var(--a)")
        h.append("<h2>Prefill</h2>" + c.svg())

    rows = [("b0 (fixed per step)", f"{fit.model.b0 * 1e3:.3f} ms"),
            ("b_dec (per decoding sequence)", f"{fit.model.b_dec * 1e3:.4f} ms"),
            ("b_ctx (per 1k context tokens)", f"{fit.model.b_ctx * 1e3:.4f} ms"),
            ("b_pf (per prefill token)", f"{fit.model.b_pf * 1e6:.3f} µs"),
            ("b_pfatt (prefill attention)", f"{fit.model.b_pfatt * 1e3:.4f} ms per 1M token²"),
            ("overhead (client-visible)", f"{fit.model.overhead * 1e3:.2f} ms"),
            ("admit_lag (extra steps before a new request is scheduled)", f"{fit.model.admit_lag:.2f}"),
            ("decode fit RMSE", f"{fit.decode_rmse * 1e3:.3f} ms ({fit.n_decode_obs} obs)"),
            ("prefill fit RMSE", f"{fit.prefill_rmse * 1e3:.2f} ms ({fit.n_prefill_obs} obs)")]
    h.append("<h2>Fitted model</h2><table>" + "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows) + "</table>")
    h.append("<h2>Engine configuration used</h2><table><tr><th>setting</th><th>value</th><th>source</th></tr>")
    for k in ("max_num_seqs", "max_num_batched_tokens", "kv_capacity_tokens", "replicas", "router"):
        v = eng.get(k)
        if isinstance(v, float) and v.is_integer():
            v = int(v)
        h.append(f"<tr><td>{k}</td><td>{'unlimited' if v is None else v}</td><td class=mut>{html.escape(src.get(k, 'default'))}</td></tr>")
    h.append("</table>")
    h.append("<h2>Assumptions and limits</h2><div class=note><ul>"
             "<li>One replica was profiled; several replicas are modelled as identical copies behind the router shown above.</li>"
             "<li>Arrivals are Poisson at a constant rate. Bursty traffic needs more headroom than this report shows.</li>"
             "<li>Prefix caching, speculative decoding, LoRA switching and multi-model serving are not modelled.</li>"
             "<li>Results outside the profiled batch sizes and prompt lengths are extrapolated.</li>"
             "<li>The 10–90% ranges reflect profile noise only; they are not a guarantee of production behaviour.</li>"
             "</ul></div></main></body></html>")
    return "".join(h)
