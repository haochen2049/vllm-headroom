# Example: Qwen2.5-7B-Instruct on vLLM 0.31.0

Server (one NVIDIA RTX PRO 6000 Blackwell, 96 GB):

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --max-num-seqs 64 --max-num-batched-tokens 4096 \
    --max-model-len 16384 --gpu-memory-utilization 0.5
```

Files:

- `profile.json`: output of `vllm-headroom profile --url http://localhost:8124 --max-num-batched-tokens 4096 --max-concurrency 128`
- `report.html`, `report.json`: output of
  `vllm-headroom analyze profile.json --slo ttft_p95=1 --slo tpot_p95=0.05 --current-rate 10 --whatif max_num_seqs=128 --whatif replicas=2`
- `near_saturation_check.py`, `near_saturation_check.json`: an extra open-loop run at 85% and 95% of
  the predicted saturation rate, compared with the frozen fit
  (`python near_saturation_check.py profile.json http://localhost:8124 0.85,0.95 45 out.json`).

Re-run the analysis without a GPU:

```bash
vllm-headroom analyze examples/qwen2.5-7b-vllm0.31/profile.json --slo ttft_p95=1 --slo tpot_p95=0.05 --current-rate 10
```
