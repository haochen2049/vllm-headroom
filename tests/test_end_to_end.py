"""Profile the bundled simulated server over HTTP, fit, analyse and render a report."""
import asyncio
import json

import pytest

from vllm_headroom import mock_server
from vllm_headroom.analysis import backtest, fit_from_profile, headroom_analysis, parse_slo
from vllm_headroom.profiler import ProfileConfig, Profiler, _parse_metrics
from vllm_headroom.report import render
from vllm_headroom.sim import EngineConfig
from vllm_headroom.workload import Workload


def test_parse_vllm_metrics():
    txt = ('vllm:cache_config_info{_block_size_resolved="True",block_size="16",effective_attention_block_size="16",'
           'kv_cache_size_tokens="1626752",num_gpu_blocks="101672"} 1.0\n'
           'vllm:num_requests_running{engine="0",model_name="m"} 3.0\n')
    m = _parse_metrics(txt)
    assert m["kv_tokens"] == 1626752 and m["block_size"] == 16 and m["num_requests_running"] == 3.0


def test_profile_mock_server_end_to_end(tmp_path):
    async def go():
        server, port, task = await mock_server.start(max_num_seqs=32, budget=4096, kv_tokens=300_000)
        cfg = ProfileConfig(url=f"http://127.0.0.1:{port}", max_concurrency=32, decode_tokens=64,
                            decode_prompts=(256, 1024), prefill_prompts=(128, 1024, 4096), prefill_reps=2,
                            validation_fracs=(0.5,), validation_seconds=4, max_num_seqs=32,
                            max_num_batched_tokens=4096)
        prof = await Profiler(cfg, log=lambda *_: None).run()
        server.close()
        task.cancel()
        return prof

    prof = asyncio.run(go())
    json.dumps(prof)  # serialisable
    assert prof["engine"]["kv_capacity_tokens"] == 300_000
    fit = fit_from_profile(prof, n_boot=4)
    true = mock_server.DEFAULT_MODEL
    assert fit.model.b0 == pytest.approx(true.b0, rel=0.25)
    assert fit.model.b_pf == pytest.approx(true.b_pf, rel=0.35)
    eng = EngineConfig.from_dict(prof["engine"])
    wl = Workload.from_dict(prof["validation_workload"])
    ana = headroom_analysis(fit, eng, wl, parse_slo(["ttft_p95=1", "tpot_p95=0.1"]), 2.0, n=400, draws=3,
                            grid=5, log=lambda *_: None)
    assert 0 < ana["lam_max"] <= ana["lam_sat"]
    bt = backtest(prof, fit, eng, reps=3)
    assert bt and all(r["measured"] > 0 for r in bt)
    e2e = [r for r in bt if r["metric"] == "e2e_p50"][0]
    assert abs(e2e["rel_error"]) < 0.3
    page = render(prof, fit, ana, bt, [], wl.summary())
    assert page.startswith("<!doctype html>") and "<svg" in page
    (tmp_path / "r.html").write_text(page, encoding="utf-8")
