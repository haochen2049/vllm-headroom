# vllm-headroom

[English](README.md) | 中文

**你的 vLLM 服务器在违反延迟 SLO 之前，还能再承受多少流量？**

`vllm-headroom` 对一台正在运行的 vLLM 服务器做约三分钟的剖析，拟合每个引擎 step 耗时的模型，再模拟连续批处理，
预测任意负载下的 p95/p99 延迟。它报告仍能满足 SLO 的最高请求速率并附上不确定度区间，还会用实测过、但没有参与拟合的
负载检验自己的预测。

```text
$ vllm-headroom analyze profile.json --slo ttft_p95=1 --slo tpot_p95=0.05 --current-rate 10 \
      --whatif max_num_seqs=128 --whatif replicas=2
saturation:             21.589 req/s
max load meeting SLO:   20.827 req/s  (10-90%: 19.948-21.538; binding: ttft_p95)
headroom vs current:    +108%  (10-90%: +99% to +115%)
what-if max_num_seqs=128: max 27.745 req/s (saturation)
what-if replicas=2: max 42.475 req/s (saturation)
```

（各行依次为：饱和速率；满足 SLO 的最大负载，括号内是 10–90% 区间和最先触线的 SLO；相对当前负载的余量；两项假设分析的结果。）

它会生成一个独立的 HTML 报告，包含延迟-负载曲线、回测结果和拟合出的模型。所有数据都留在你的机器上。

> 状态：alpha 版（0.1）。只在 vLLM 0.31 和一种 GPU 上测试过，见[验证](#验证)。
> 与 vLLM 项目没有隶属关系。

## 为什么需要它

仪表盘展示已经发生的事，预测工具把过去外推出去。两者都回答不了这些问题：如果你改了点什么，比如流量变大、
`--max-num-seqs` 换个值、少一个副本、提示变长，会发生什么？离延迟突然飙升的那个临界点还有多远？把每种方案都压测
一遍很慢。一个针对*你自己的*服务器拟合出来的模型，几秒钟就能回答。

## 快速开始

```bash
pip install git+https://github.com/haochen2049/vllm-headroom

# 没有 GPU？剖析自带的模拟服务器并生成示例报告（约 2-3 分钟）
vllm-headroom demo -o demo_report.html
```

针对真实服务器时，最好用一个空闲副本或预发布（staging）环境：

```bash
# 1. 剖析（约 3-4 分钟）；如果知道服务器自己的配置，请一并传入
vllm-headroom profile --url http://localhost:8000 \
    --max-num-seqs 256 --max-num-batched-tokens 8192 -o profile.json

# 2. 按你的 SLO 和当前流量分析
vllm-headroom analyze profile.json --slo ttft_p95=2 --slo tpot_p95=0.1 \
    --current-rate 12 --workload my_requests.jsonl -o headroom.html

# 3. 快速比较不同配置
vllm-headroom whatif profile.json --slo ttft_p95=2 \
    --change max_num_seqs=512 --change replicas=3 --change kv_scale=0.5
```

`--workload` 是一个 JSONL 文件，每行一个请求，例如 `{"prompt_tokens": 812, "output_tokens": 230}`
（直接用 OpenAI 的 `usage` 对象也可以）。不提供时，可以用 `--prompt-mean/--output-mean` 生成对数正态分布的请求组合；
两者都不给，就沿用剖析时的请求组合。SLO 指标有三个：`ttft`（首 token 时间）、`tpot`（首 token 之后每个输出 token
的时间）和 `e2e`（端到端时间），每个都可以配任意分位数，例如 `ttft_p95=2`、`e2e_p99=30`、`tpot_p50=0.05`。

## 工作原理

**1. 剖析。** 对 OpenAI 兼容接口分四个阶段发送请求：

| 阶段 | 发送什么 | 辨识什么 |
|---|---|---|
| 预填充扫描 | 服务器空闲时逐个发送单请求，提示长度从 128 到 8192 token | 每个 token 的预填充成本和固定开销 |
| 解码波次 | 一次同时发出 `c` 个相同请求，`c` = 1, 2, 4 ... 256，分短提示和长提示两组 | step 时间与批大小、上下文长度的关系 |
| 准入探测 | 在一个波次解码期间插入短请求 | 新请求要等几个 step 才会被调度 |
| 验证 | 以预测饱和速率的 30%、50%、70% 施加开环泊松负载 | 不用于辨识，留作回测 |

提示用的是随机 token id，所以前缀缓存无法缩短预填充，长度也是精确的；`ignore_eos` 固定了输出长度。流式返回的分块
时间戳给出 token 间隔；只有其间没有其他请求做预填充的间隔才用于解码拟合。KV 容量从 `/metrics` 读取；如果不传
`max_num_seqs`，会从波次数据中推断。

**2. 拟合。** 引擎 step 时间建模为

```
step = b0 + b_dec·(解码中的序列数) + b_ctx·(这些序列的上下文 token 数)
          + b_pf·(预填充 token 数) + b_pfatt·(预填充注意力项)
```

用稳健最小二乘拟合。另有两个参数：客户端可见的固定开销 `overhead`，以及准入滞后 `admit_lag`（vLLM V1 在当前
step 运行时就规划下一个 step，所以新请求大约要多等一到两个 step）。参数不确定度来自对剖析请求的 bootstrap 重采样。

**3. 模拟。** 一个 vLLM 式连续批处理的离散事件模拟器，覆盖：受 `max_num_batched_tokens` 约束的分块预填充、
`max_num_seqs`、KV 容量及容量不足时的重算式抢占，以及位于最少负载或轮询路由之后的多个副本。纯解码区间用闭式解直接
推进，所以模拟 1,500 个请求远不到一秒。

**4. 报告。** 满足所有 SLO 的最大泊松速率（二分搜索求得）、饱和速率、带 10–90% 区间（来自 bootstrap 重拟合）的
延迟曲线、假设分析对比，以及针对留出验证负载的回测。

## 验证

目前只跑过一组：vLLM 0.31.0 上的 Qwen2.5-7B-Instruct，一张 NVIDIA RTX PRO 6000 Blackwell（96 GB），启动参数
`--max-num-seqs 64 --max-num-batched-tokens 4096 --gpu-memory-utilization 0.5`。剖析数据、报告和检验脚本都在
[`examples/qwen2.5-7b-vllm0.31`](examples/qwen2.5-7b-vllm0.31)。工作负载是默认的对数正态组合（提示平均 505 token，
输出平均 130 token）。没有传入 `max_num_seqs`，工具正确推断出了 64。

**回测**（同一次剖析中的留出验证负载，从未用于拟合）：

| 负载（占预测饱和速率） | 请求数 | TTFT p50 | TTFT p95 | E2E p50 | E2E p95 |
|---|---|---|---|---|---|
| 30%（6.5 req/s） | 161 | +8% | +7% | +2% | +2% |
| 50%（10.8 req/s） | 268 | +9% | +12% | +2% | +2% |
| 70%（15.1 req/s） | 376 | +8% | +14% | +3% | +4% |

**接近饱和**（之后对同一台服务器单独跑的一次测试，拟合参数冻结不变，脚本为 `near_saturation_check.py`）：

| 负载 | 请求数 | TTFT p50 | TTFT p95 | E2E p50 | E2E p95 |
|---|---|---|---|---|---|
| 85%（18.5 req/s） | 833 | +7% | +5% | +3% | +5% |
| 95%（20.7 req/s） | 931 | 0% | −19% | −3% | 0% |

表中数值是预测相对实测的误差（正值表示预测比实测慢）。端到端延迟在所有负载下误差都在 5% 以内。85% 负载以内，TTFT
的预测略偏悲观。95% 负载时 TTFT 尾部被低估 19%，因为接近饱和时，速率上的微小误差会让尾部大幅移动。饱和前的最后几个
百分点应视为不可靠，要留出余量。这只是一个模型、一张 GPU、一种工作负载的结果，换到其他环境可能不同，非常欢迎提交
其他硬件上的回测结果。

单元测试还检查了：模拟器与一个朴素的逐 step 参考实现结果完全一致；能从合成数据中还原已知参数；能通过 HTTP 对自带的
模拟服务器跑通"剖析 → 拟合 → 报告"全流程。

## 局限

- 只剖析一个副本；多副本按完全相同的副本建模。
- 到达过程假设为恒定速率的泊松过程。突发流量需要比报告更多的余量。
- 未建模：前缀缓存命中、投机解码、LoRA 适配器切换、多模型服务、预填充/解码分离部署、流水线并行气泡、客户端与路由
  之间的网络。
- 超出剖析范围的批大小和提示长度属于外推。
- 不确定度区间只反映剖析噪声，不能覆盖生产环境可能出现的所有差异。
- 剖析会发送真实负载，请在不服务用户的副本上运行。

## 输出文件

`profile` 写出原始计时数据（不含任何提示或输出文本）。`analyze` 写出 `headroom.html` 和 `headroom.json`，内容包括
拟合参数、曲线、余量、回测和假设分析结果。

## 开发

```bash
pip install -e ".[dev]"
pytest -q
vllm-headroom mock-server --port 8001   # 用于实验的模拟 OpenAI 兼容服务器
```

## 许可证

Apache-2.0
