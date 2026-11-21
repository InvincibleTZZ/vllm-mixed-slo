# 混合负载压测指南

本工具从一个有固定到达时刻的请求文件回放短、长两类生成请求。每条请求存有完整的输入 token ID 和目标生成长度，方便不同调度器重放同一负载。压测端只需要 Python 标准库。服务端接口以 [vLLM v0.10.2 流式 completions 实现](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/entrypoints/openai/serving_completion.py) 为准。

## 跑一次可对照的实验

在仓库根目录运行以下命令。先启动项目所用的 vLLM 服务，URL 可以是服务根路径或以 `/v1` 结尾的路径。`--model` 填服务暴露的模型名。下面的 SLO 是演示阈值，正式实验应在观察测试集结果前选定两类请求的目标值。

```bash
python -m bench generate --out runs/trace.jsonl --mix mixed --requests 100 --rate 3 --seed 42 --short-input 128 --short-output 32 --long-input 2048 --long-output 64 --long-fraction 0.25
python -m bench run --trace runs/trace.jsonl --url http://127.0.0.1:8000 --model MODEL --out runs/baseline.raw.jsonl --label baseline --timeout 120 --max-inflight 128
python -m bench summarize --results runs/baseline.raw.jsonl --short-ttft-ms 1000 --long-ttft-ms 3000 --tpot-ms 100 --out runs/baseline.summary.json
```

切换服务配置并用独立 trace 预热后，用同一个测试文件执行第二次回放。为复现实验，将模型 revision、vLLM 版本、驱动/GPU、服务启动参数、种子、日期和同机负载随结果保存。命令使用独占创建，每次复测请换输出名。

```bash
python -m bench run --trace runs/trace.jsonl --url http://127.0.0.1:8000 --model MODEL --out runs/candidate.raw.jsonl --label candidate --timeout 120 --max-inflight 128
python -m bench summarize --results runs/candidate.raw.jsonl --short-ttft-ms 1000 --long-ttft-ms 3000 --tpot-ms 100 --out runs/candidate.summary.json
python -m bench compare --left runs/baseline.raw.jsonl --right runs/candidate.raw.jsonl --out runs/paired.json
```

`compare` 要求 trace 文件 SHA-256 和模型名一致，按请求 ID 比对成功请求的每一个输出 token，另列失败及缺少成功配对数。批次变化可能引入数值差异，一旦发现序列差异，需要查看对应请求及模型/服务配置。此处输出一致性只覆盖该工作负载。

## 负载设计与控制变量

| 参数 | 实验含义 |
| --- | --- |
| `--mix short/long` | 单一长度类别，用于识别瓶颈和做负对照。 |
| `--mix mixed` | 以 `--rate` 个请求/秒作为泊松到达的期望强度，首个请求在零点到达。类别顺序洗牌，长请求总数取 `round(请求数 × long_fraction)`。 |
| `--mix burst --burst-size B` | 每组最多 B 条同时到达，相邻组间隔 B/rate 秒；最后一组可少于 B 条。 |
| `--token-min/--token-max` | 闭区间内抽取实际词表可用的非特殊 ID。默认 1024–4095，首次更换模型务必核对词表。保证输入长度 + 目标输出长度满足服务的上下文上限。 |
| `--timeout` | 从原定到达时刻开始的客户端总截止时间，单位秒。超时关闭连接。 |
| `--max-inflight` | 客户端并行请求上限。到达时槽已满就记录 `client_overload`，仍计入请求总数。 |
| `--short-tpot-ms/--long-tpot-ms` | 覆盖某一类别的 TPOT 目标；默认使用共同的 `--tpot-ms`。 |

每种配置先跑独立热身流量。正式比较可交替运行 baseline 和 candidate，使用同一组独立种子配对，固定模型权重、量化、张量并行度、上下文上限、KV 配置、prefix caching、输出长度和服务端后台任务。先扫短流量、长流量、混合流量，再在相近的过载水平测试 burst。将选参数用的 trace 和最终报告用的 trace 分开。合成 ID 适合控制计算量；真实会话中的文本分布、前缀复用和提前停止需要独立实验。

## 计时与原始证据

客户端在单调时钟上调度独立到达，创建每次请求自己的 HTTP 连接，以 SSE 接收 `/v1/completions`。固定 `temperature=0`、`ignore_eos=true`、`add_special_tokens=false`，要求返回增量 `token_ids` 和最终 `usage`。首个空 delta、usage-only 事件不计作 token。收到 `[DONE]` 时同时核对最终原因 `length`、输入 token 数、输出 ID 个数和请求目标。任何一项不合格都保留错误结果。建连、排队及流式传输时间包含在客户端体验中。

每条到达对应一条原始 JSONL，按完成时间写出，关联键为 `id`。主要字段如下。

| 字段 | 定义 |
| --- | --- |
| `run_id, trace_sha256, expected_requests, label, model, backend` | 运行标识、输入文件内容哈希、预期请求数及配置来源。`backend` 以启动参数申明，本地 mock 还会用响应头自标识。 |
| `scheduled_s, sent_s, end_s` | 以本次回放起点为零的相对秒。依次为预约到达、客户端工作线程开始建连和请求完成/失败。客户端拒绝时 `sent_s=null`。 |
| `input_tokens, target_tokens, output_ids` | 输入与目标长度及实际顺序输出的完整 token ID。 |
| `events` | 每个含 token 的 SSE 数据事件保存 `[客户端收到时刻, 此事件含有的 token 数]`。 |
| `status, http_status, finish_reason, usage, error` | 正常、超时、HTTP 返回错误、协议/连接异常和客户端容量不足；不适用的字段为空或省略。 |

汇总会核对同一 run、SHA-256、请求数、模型名及 ID 唯一性，拒绝对半途退出的残缺回放出报告。客户端失联前已收到的 token 可以从原始结果查看。

## 如何读汇总

| 指标 | 计算方式 |
| --- | --- |
| `ttft_ms` | 首个含 token 的 SSE 事件到达时刻减去原定到达时刻，含客户端调度滞后。首事件含多个 token 时，测到的是客户端第一次看到输出的时间。 |
| `tpot_ms` | 若首事件只有一个 token 且总输出 N>1，取 `(最后 token 事件时刻 − 首 token 事件时刻)/(N−1)`。它是每请求均值。首事件合并多个 token 时无法分离起始 token，该请求的 TPOT 记为缺测，严格 SLO 视为未验证达标。 |
| `token_itl_ms` | 仅在该请求所有含 token 事件均为单 token 时，取相邻可见时间差后跨请求汇总。长输出贡献更多间隔。 |
| `sse_token_chunk_gap_ms` | 所有成功请求相邻含 token 的 SSE 事件到达间隔；另报合并事件的请求数和首事件合并的请求数。 |
| `e2e_ms, dispatch_lag_ms` | 分别为处理至终止的总耗时和工作线程实际开始时刻减去预约到达时刻。耗时显著受建连、网络及客户端开销影响。 |
| `attainment` | 成功且同时符合自身类别 TTFT、TPOT 阈值的请求数除以该类全部到达数；单 token 请求只考察 TTFT。 |
| `goodput_rps` | 该类双目标达标数除以从回放零点至最后一条请求结束的全局墙钟秒数，包含 drain。短、长两类 goodput 之和等于总体 goodput。 |
| `offered_rps, completed_rps, successful_output_tokens_per_s` | 各类到达数、成功数、成功输出 token 数分别除以同一全局窗口。`--rate` 是配置的流量强度，有限窗口和排空会改变实际观测值。 |

TTFT、TPOT、ITL 和 E2E 的百分位只含完整成功请求；每项都给出样本数、线性插值 p50/p95/p99 和最大值。失败和超时保留在各类失败数、attainment 和 goodput 中。报告性能时同时展示短/长类样本量、p95/p99、失败数、达标吞吐及输出 token/s。检查 `dispatch_lag_ms` 和 `client_overload`，若负载发生器已饱和，应调整客户端容量并重新测量。客户端 TTFT 包含多个阶段，定位服务端等待时间需结合引擎指标。小样本的 p99 主要取决于一两个请求，给出单个 smoke run 的数字时应保留这个限制。

## CPU 验证

用两个终端从仓库根目录启动 mock 和客户端。mock 实际通过本机 TCP 返回 HTTP chunked SSE，输出由输入 token 派生，仅供协议测试。

```bash
# 终端一
python -m bench.mock_server --port 8765
# 终端二
python -m bench generate --out runs/mock.trace.jsonl --requests 8 --mix burst --rate 20 --short-input 8 --long-input 16 --short-output 3 --long-output 4
python -m bench run --trace runs/mock.trace.jsonl --url http://127.0.0.1:8765 --model single --backend mock --label protocol-smoke --out runs/mock.raw.jsonl
python -m bench summarize --results runs/mock.raw.jsonl --short-ttft-ms 5000 --long-ttft-ms 5000 --tpot-ms 5000
python -m unittest discover -s tests -p "test_bench*.py" -v
```

测试覆盖重复生成相同 trace、不同类别阈值、单 token 请求、同一事件合并多个 token、HTTP 429、错误 usage、缺少 `[DONE]`、截止时间、客户端并发上限和残缺结果。mock 的吞吐和延迟不代表 vLLM/GPU 的性能。
