"""Extra out-of-sample check near saturation, using a frozen fit."""
import asyncio, json, sys
import numpy as np
from vllm_headroom.profiler import Profiler, ProfileConfig, _now
from vllm_headroom.analysis import fit_from_profile, backtest
from vllm_headroom.sim import EngineConfig, saturation_rate
from vllm_headroom.workload import Workload

prof = json.load(open(sys.argv[1]))
url, fracs, secs = sys.argv[2], [float(x) for x in sys.argv[3].split(",")], float(sys.argv[4])
fit = fit_from_profile(prof, n_boot=20)
eng = EngineConfig.from_dict(prof["engine"])
wl = Workload.from_dict(prof["validation_workload"])
lam = saturation_rate(fit.model, eng, wl.prompt, wl.output)
print("frozen fit saturation", lam)

async def go():
    p = Profiler(ProfileConfig(url=url, seed=7))
    await p._server_info()
    levels = []
    for f in fracs:
        rate = f * lam
        n = int(rate * secs)
        gaps = p.rng.exponential(1 / rate, size=n); offs = np.cumsum(gaps) - gaps[0]
        P, O = wl.sample(n, p.rng)
        t0 = _now()
        async def fire(d, pt, ot):
            await asyncio.sleep(max(0, t0 + d - _now()))
            return await p.request(int(pt), int(ot))
        rs = await asyncio.gather(*[fire(d, a, b) for d, a, b in zip(offs, P, O)])
        rows = [{"t_send": r["t_send"] - t0, "ttft": r["times"][0] - r["t_send"] if r["times"] else None,
                 "e2e": r["times"][-1] - r["t_send"] if r["times"] else None, "prompt_tokens": r["prompt_tokens"],
                 "output_tokens": r["output_tokens"], "err": r["err"]} for r in rs]
        levels.append({"fraction": f, "rate": rate, "requests": rows})
        print("level", f, rate, len(rows), sum(1 for r in rows if r["err"]), flush=True)
    await p.client.aclose()
    return levels

levels = asyncio.run(go())
bt = backtest({"validation": levels}, fit, eng, reps=20)
for b in bt:
    print(b["fraction"], round(b["rate"], 2), b["n"], b["metric"], round(b["measured"], 4), round(b["predicted"], 4),
          [round(x, 4) for x in b["range"]], round(100 * b["rel_error"], 1))
json.dump({"levels": levels, "backtest": bt}, open(sys.argv[5], "w"))
