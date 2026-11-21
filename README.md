# vLLM Mixed-SLO

**在 vLLM V1 的一次推理迭代里，已经有请求正在逐 token 输出时，还能放进多少个 prompt token？** 本项目在 vLLM 0.10.2 的单卡同步调度器中实现了按步调整的 chunked prefill 预算，并提供从引擎采样、离线校准、在线决策到客户端配对压测的完整实验。

长 prompt 分块越大，完成 prefill 所需的迭代通常越少，但它可能拉长同批 decode 用户下一次看到输出的等待时间。把分块一律调小也有代价，长请求可能更晚看到首 token。本项目分别跟踪短、长两类请求的首 token 延迟（TTFT）、后续 token 平均时延（TPOT）、可观察的相邻 token 间隔（ITL），以及同时达到 TTFT 和 TPOT 目标的每秒请求数（SLO goodput）。

**当前验证记录。** 2026-11-21 在本地 CPU 上载入经 SHA-256 校验的 vLLM 0.10.2 上游调度源码，27 项回归测试全部通过。其中一项在上游循环中重放运行中与等待中的请求，验证两个入口共享 6 个 prefill token，另一个 ready decode 获得 1 个 token。校准样例的耗时为人工构造值。GPU 吞吐和尾延迟应从对应模型的配对运行结果中读取。[上游循环断言](tests/test_engine_install.py) · [实验矩阵](docs/EXPERIMENT.md)

## 从一次调度到用户指标

| 环节 | 实现 | 可检查的数据 |
| --- | --- | --- |
| 观察原生策略 | 在调度前记录就绪 decode 数及其已计算上下文 token 总数，提交后统计实际分配的 prompt token。计时覆盖从调度开始到收到这一批模型执行结果的主机墙钟区间。 | 每个有效 step 的形状、`schedule_ms`、`step_ms`、排除原因累计数。[引擎接入](engine/README.md) |
| 拟合并留出验证 | 依据上述三个特征分桶，拟合有足够样本的条件 p95。同一 decode 状态下，对有数据的 prefill 桶取顺序累计最大值。用另一批完整运行检查覆盖率、误差及被误判为达标的比例。 | `profile.json`、`validation.json` 和输入文件 SHA-256。[校准方法](calibration/README.md) |
| 在线选择预算 | 为已就绪的 decode 预留 token，在有数据支持的候选中选择符合本步目标的最大 prefill 额度。两个原生分配入口共享该额度，KV 分配成功才扣账。未知状态回退到小块，连续零进展后周期性释放有限额度。 | 每步的预算、实际 prompt 数与决策原因。[控制器源码](engine/policy.py) |
| 配对评测 | 独立的开环客户端按同一文件重放到达时刻、输入 ID 和目标输出长度。各实验臂逐请求检查完整输出 ID，并把超时、错误和客户端过载保留在全部到达数中。 | 原始流式事件、按长短分类的指标、各臂相对原生调度的 goodput 差值。[压测与指标](bench/README.md) |

控制器的 `target_ms` 约束的是引擎单步的预测耗时。客户端的 TTFT 包含排队、完整 prompt 计算和传输，TPOT 跨越多步。项目保留两个时钟及其对应原始记录，便于查明“混合步缩短，长请求首 token 变慢”一类权衡。

## CPU 快速验证

在仓库根目录使用 Python 3.10 或以上版本。CPU 路径只使用标准库，可在没有模型权重和 CUDA 的机器上运行。

```bash
python -m unittest discover -s tests -v
python -m calibration fit --traces calibration/fixtures/SYNTHETIC_fit.jsonl --metadata calibration/fixtures/SYNTHETIC_environment.json --min-samples 2 --out runs/cpu/profile.json
python -m calibration evaluate --profile runs/cpu/profile.json --traces calibration/fixtures/SYNTHETIC_holdout.jsonl --target-ms 11 --out runs/cpu/validation.json
```

这组合成样例拟合出 3 个受支持的桶。留出的 5 步中有 3 步得到估计，模型认为会在 11 ms 内完成的 3 步里有 1 步实际超标。这个反例解释了为什么除了误差还要报告覆盖率，以及“预测达标”集合的违约数和分母。仓库还包含本机 HTTP 分块流式回放测试，覆盖请求超时、过载和单个事件合并多个 token 的情况。[合成数据](calibration/fixtures/) · [流式协议测试](tests/test_bench.py)

常规 CPU 测试在没有上游 fixture 时跳过两项版本集成测试。可用下面的命令在 Linux 重跑全部测试。测试会核对上游文件哈希，然后执行固定版本的真实调度循环，使用 CPU 替身处理 KV 分配等外部依赖。[上游回归测试](tests/test_engine_install.py) · [CI 工作流](.github/workflows/ci.yml)

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm/v0.10.2/vllm/v1/core/sched/scheduler.py -o /tmp/vllm-v0102-scheduler.py
MIXED_SLO_UPSTREAM_FIXTURE=/tmp/vllm-v0102-scheduler.py python -m unittest discover -s tests -v
```

## GPU 快速实验

共享服务器上已有固定版本时，可从只读环境建立项目私有副本，使用该副本的 Python 执行实验。脚本会核对补丁前后的公共文件哈希和私有解释器的实际导入位置。[共享环境使用手册](docs/SHARED_ENV.md)

准备 Linux、兼容的 NVIDIA 驱动、单张空闲 GPU 和固定版本的本地 Hugging Face 生成模型。模型须支持至少 3072 的上下文窗口，并检查输入 token ID 1024–4095 在其词表内。以下示例在仓库根目录和专用虚拟环境执行，`MODEL` 指向已有模型快照，`GPU` 为物理编号。安装器会修改这个环境中经 SHA-256 校验的 vLLM 调度器文件，正在使用它的服务应先停止。

```bash
python -m pip install -e '.[gpu]'
python -m engine.install apply
MODEL=/absolute/path/to/pinned-model
GPU=0
python -m scripts.experiment --model "$MODEL" --gpu "$GPU" --out runs/quick-01 --quick --dry-run
python -m scripts.experiment --model "$MODEL" --gpu "$GPU" --out runs/quick-01 --quick
```

Quick 依次启动 8 次服务。先做 2 个校准运行、1 个独立留出运行，再用同一测试 trace 比较 5 个实验臂：原生关闭采样、原生开启采样、自适应调度、小固定预算和中固定预算。全流程核对 trace、运行配置、成功请求数及输出序列。若出现序列差异、留出集无 mixed step 支持，或策略没有在 mixed step 命中 profile，程序仍写出报告并返回非零状态。

去掉 `--quick` 并更换 `--out` 可运行完整矩阵，包含纯短、纯长、两档混合速率和突发负载，每个测试场景使用 3 个独立种子。中断后可用相同参数加 `--resume` 复用已完成的轮次。实验结束并停止服务后，执行 `python -m engine.install revert` 可恢复原始调度器。[矩阵参数、对照设计与恢复方法](docs/EXPERIMENT.md)

## 如何核查性能结论

项目记录了实际部署中遇到的包名冲突、公共环境版本检查与 GPU 占用变化。模型快照的权重、配置和分词文件 SHA-256 存入 `plan.json → implementation → model_snapshot`。实验完成或中断后均可导出可复算的证据包。[验证与失败复盘](docs/RESULTS.md) · [结果导出指南](docs/PUBLISH_EVIDENCE.md)

| 想回答的问题 | 打开的文件及应查看的字段 |
| --- | --- |
| 两臂真的处理了同一批请求吗？ | `aggregate.json → evaluations → <trace> → arms → paired_vs_off` 给出成功配对数、输出不同数以及样例请求 ID。各轮 `manifest.json` 给出硬件、模型配置及 trace 哈希、命令和状态。 |
| 目标负载上有多少步真的受策略支持？ | 校准读取各轮去除热身的 `measured-steps.jsonl`。用 `validation.json → groups.mixed` 看独立 mixed 步覆盖率，再用 `aggregate.json` 的 `step_evidence` 查策略实际命中、回退和进展释放的次数。 |
| 哪类用户承担了代价？ | 各臂的 `summary.json → groups.short/long` 提供请求数、失败、TTFT/TPOT/ITL 的样本数与分位数、达标比例和 goodput。原始 `requests.jsonl` 可以追到具体请求。吞吐差异应与同一模型、同一 trace 下的静态预算和采样开销一起阅读。 |

本项目的适用范围是同步单 GPU、FCFS、启用 chunked prefill 的文本生成。默认矩阵固定 eager 模式并关闭 prefix caching。按形状分桶会丢失同一总长度下的请求长度分布，抢占、重算或实际 decode 与事前状态不符的步会从拟合样本中排除，客户端仍统计这些时间段的全部到达。整体 prefill 进展保护无法为每个排队请求给出截止时间保证。固定长度的随机 token 负载适于隔离调度影响，真实业务中的前缀复用和自然停止需要独立验证。

可重放的失败案例、对应测试、修复取舍及面试讲解见[设计复盘与面试说明](docs/INTERVIEW.md)。项目使用 [Apache License 2.0](LICENSE)。
