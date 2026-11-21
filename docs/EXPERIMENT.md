# 单 GPU 配对实验

这个实验回答三个问题。在相同到达时刻和输入下，自适应 prefill 是否改变短请求与长请求的 SLO 达标吞吐？变化来自调度决策、采样开销，还是单纯降低了固定 token 预算？独立数据上的耗时表，能覆盖多少真实 mixed step？

## 启动

在项目根目录预览实验。此命令只打印种子、负载、运行顺序和预算。

```bash
python -m scripts.experiment --model /models/pinned-snapshot --gpu 0 \
  --out runs/quick-01 --quick --dry-run
```

实跑需要 Linux、独占空闲 NVIDIA GPU、固定的本地 Hugging Face 模型和 vLLM 0.10.2 环境。选用模型时核对生成器默认的输入 ID 范围 1024–4095 与模型词表，正式模式的最长请求为 2048 个输入和 64 个输出 token。先在专用环境执行 `python -m engine.install apply`。矩阵会比对解释器当前安装的 scheduler 与仓库生成的补丁字节。每次启动服务前，单轮脚本重新检查物理 GPU 序号和占用。

```bash
# 8 次服务启动，检查数据采集、回退和输出一致性
python -m scripts.experiment --model /models/pinned-snapshot --gpu 0 \
  --out runs/quick-01 --quick

# 83 次服务启动，每个测试负载都有三个独立种子
python -m scripts.experiment --model /models/pinned-snapshot --gpu 0 \
  --out runs/full-01 --rate-scale 1 --target-ms 40 \
  --short-ttft-ms 1000 --long-ttft-ms 3000 --tpot-ms 100
```

通过 `--rate-scale` 对所有到达率乘同一系数。用独立预实验选择能让 prefill 和 decode 实际竞争的负载范围，再固定速率、阈值和模型快照。默认的 40 ms 是调度器每步墙钟目标，另外三个毫秒数定义客户端 SLO。默认设置仅是可复现的起点。确认模型路径引用不可变的 revision，保存模型权重来源与 revision。运行清单校验模型配置与 scheduler 的 SHA-256，没有对整个权重文件重新散列。

初次运行指定新的输出目录。中断后用原参数加 `--resume`。程序核对实验计划、代码指纹、输入 trace 和已完成试验的清单，在新的 attempt 目录重做失败轮次，保留旧日志。每次运行按顺序复用同一张卡，同一输出目录同时只允许一个实验进程。变更代码、目标或参数时使用新目录。

## 实验设计

| 阶段 | Quick | Full |
| --- | --- | --- |
| 校准 | 两条独立 trace，原生调度加采样，全局预算 128、512 | 预算 256、512、1024 分别运行一次混合泊松和一次突发，共 6 条 trace |
| 独立验证 | 另一条突发 trace，预算 512 | 混合和突发各 1 条独立 trace，预算 1024 |
| 测试 | 一条突发 trace，所有 5 个实验臂共用 | 5 类场景 × 3 个种子 × 5 个实验臂，按种子与场景轮换顺序 |
| 请求规模 | 每条 trace 16 个请求，短 64→8、长 512→16 | 每条 trace 96 个请求，短 128→32、长 2048→64 |
| 分箱样本数 | 至少 2 个 step | 至少 20 个 step |

Full 场景包括纯短 12 req/s、纯长 12 req/s、混合泊松 6/18 req/s 和混合突发 24 req/s。混合流量的长请求占 25%，突发每 8 条同刻到达。校准、验证和测试使用互不重叠的种子。每个实验臂重新启动服务、重放相同热身请求，正式请求共享同一个 trace 文件和 SHA-256。耗时表只拟合 `measured-steps.jsonl` 中的正式步。验证后冻结 profile；测试阶段各臂不会触发重拟合。

| 实验臂 | 读法 |
| --- | --- |
| `off` | 补丁处于关闭状态的原生调度，预算取表中最大档，用作配对参照。 |
| `observe` | 相同预算和调度，增加状态采样、结果计时与写盘。与 off 比较可估计观测开销。 |
| `adaptive` | 相同原生预算上限，启用采样和 prefill 控制。与 observe 比较调度效果，与 off 比较整套方案效果。 |
| `static-small`、`static-mid` | 原生调度，关闭采样，分别选择较小/中间的全局预算。全局预算约束所有调度 token；自适应约束实际 prompt 分配。两者的机制与计算量均应从实测解释。 |

## 审阅结果

1. 先打开 `aggregate.json` 的 `checks`。程序核对每条正式到达都成功、所有配对请求的完整输出 token 一致、独立验证覆盖了 mixed step、adaptive 至少有一次使用 profile 的 mixed step。设 `--min-mixed-coverage 0.8` 可以把事先选定的覆盖率阈值加进验收。任一检查为假时仍写完整报告，进程返回 2。异常中止时查看对应目录的 `manifest.json` 与 `server.log`。
2. 打开 `validation.json`，同时检查 `groups.mixed.rows`、`covered_rows`、超出预测上界比例、毫秒误差和 `target_check.false_safe_fraction`。低支持度或样本过少时，把结果写成在当前负载下无法验证相关收益。Quick 每个 case 只有一个种子，仅用于验证运行链路。
3. `aggregate.json → evaluations → <trace> → arms` 并列给出各类请求数、成功数、TTFT/TPOT/ITL p95、达标比例和 goodput。每臂保留 `paired_vs_off` 的相同/不同输出数及异常 ID。即使所有请求等长结束，也可能在个别 token 上分歧。优先取出对应 ID 在两份 `requests.jsonl` 中的输出逐项定位。
4. `paired_goodput_deltas` 展示每个场景与速率下，各臂相对 off 的 goodput 绝对差，单位 req/s，跨种子给出最小、中位、最大值。它们是三个独立重复的描述统计；保留每个种子结果以及各场景短/长请求尾延迟。权衡可能表现为短请求 ITL 降低、长请求 TTFT 升高或静态小预算同样有效。
5. `step_evidence` 给出独立测试过程中 profile 在已记录 step 上的覆盖率、mixed 覆盖率、每类决策原因和最后可见的累计排除数。查看 `mixed_step_reasons.profile` 与 `uncovered`，区分有证据支持的正预算与回退。正预算出现本身不能证明它改变了分配量。累计排除数由本次服务启动起累计，可能包含热身，并可能漏掉最后一条已记录 step 之后的排除项。
6. 复查 `trials/attempt-*/<trial>/summary.json` 中 `dispatch_lag_ms`、`client_overload`、独立的短/长样本量和 p99。若客户端饱和、GPU 型号或模型修订变化，记录受影响的比较。包含失败请求的 SLO 分母与 TTFT/ITL 等百分位采用不同口径，详见仓库中的 bench 文档。

客户端延迟来自本机流式回放。固定随机 token 的合成工作负载可以控制输入长度、输出长度和到达顺序；对真实会话中的前缀复用、提前停止和不同模型，需要收集额外证据。适合写入简历的数值应同时注明 GPU、模型 revision、到达率、对照预算、SLO、运行种子、输出一致性检查和原始结果位置。

## CPU 测试

```bash
python -m unittest discover -s tests -p 'test_experiment.py' -v
```

测试覆盖独立种子、五臂共享 trace、预算与 profile 参数传递、dry-run 无磁盘副作用，以及合成数据的端到端汇总。测试特意改变一个配对请求的一个输出 token，验证请求全部成功时，矩阵仍能标出不同序列并返回 2。
