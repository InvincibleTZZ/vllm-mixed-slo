# vLLM V1 调度器接入说明

本项目在 vLLM 0.10.2 的 `Scheduler.schedule` 内给每个混合步设置共享的 prompt token 上限。已有 decode 请求沿用 vLLM 的调度逻辑。研究范围是 Linux 单卡、单引擎、解码器文本生成、FCFS、chunked prefill、同步调度和本机 KV cache。张量、流水线、数据和 decode context 并行度均为 1。启用控制器时会检查这些配置，并排除投机解码、LoRA、远程 KV 和多模态配置。

## 安装和撤销

在项目目录和专用 vLLM 0.10.2 虚拟环境里运行以下命令。应用或撤销补丁前先停止该环境的服务，重启后生效。安装器读取源码字节，只有与官方标签 SHA-256 匹配的版本才能应用。补丁将 Python 策略嵌入单个调度器文件，保持原安装路径及权限。通过同目录临时文件和原子替换发布，撤销时从带校验的插入区恢复原始字节。重复执行同一操作可得到相同结果。发现第三方修改时会输出校验错误。

```bash
python -m engine.install status
python -m engine.install apply
python -m engine.install status
# 实验结束并关闭服务后
python -m engine.install revert
```

若目标是另一个源码副本，可在任一命令后加 `--vllm-root /path/to/vllm/vllm`，路径下应有 `v1/core/sched/scheduler.py`。官方源码 SHA-256 为 `2bd28fc32cea71370298bb8da0d9a0bbd163ab315b2c6428540131e3999a255f`。[上游 Scheduler](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/v1/core/sched/scheduler.py) · [模型执行和取回结果的 EngineCore.step](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/v1/engine/core.py)

## 启动方式

安装补丁后沿用 `vllm serve`。下面的变量在执行服务命令的同一 shell 设置。GPU 编号、模型路径、原生预算和输出目录需由当前实验指定。每次重启分配一个唯一 run ID 和新 trace 文件，避免把热身、拟合与测试步混入同一次运行。

在 v0.10.2 默认单卡同步执行器中，最多只有一个批次在执行，EngineCore 依次调用调度、模型执行、状态回写。默认 `async_scheduling=False`，启用异步调度会在初始化时触发检查。使用 `--enforce-eager` 可固定 eager 执行，选择图模式时须重新校准。Docker 运行需将 profile、trace 所在目录映射到容器，并为 PyTorch 多进程设置足够的共享内存；可通过 `--shm-size` 指定容量，在容器内用 `df -h /dev/shm` 核实。[vLLM 0.10.2 执行器](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/v1/executor/abstract.py) · [官方 Docker 文档](https://docs.vllm.ai/en/v0.10.2/deployment/docker.html)

```bash
mkdir -p runs/native-01
export VLLM_USE_V1=1
export MIXED_SLO_MODE=observe
export MIXED_SLO_RUN_ID=native-01
export MIXED_SLO_TRACE="$PWD/runs/native-01/steps.jsonl"
vllm serve "$MODEL" --enable-chunked-prefill --max-num-batched-tokens 1024 --max-num-seqs 32
```

用训练 trace 得到 profile 后，另起一个服务运行 `adaptive`。其余 vLLM、设备和模型参数应与相应实验对照一致。

```bash
mkdir -p runs/adaptive-01
export MIXED_SLO_MODE=adaptive
export MIXED_SLO_RUN_ID=adaptive-01
export MIXED_SLO_TRACE="$PWD/runs/adaptive-01/steps.jsonl"
export MIXED_SLO_PROFILE="$PWD/runs/profile.json"
export MIXED_SLO_TARGET_MS=25
export MIXED_SLO_FALLBACK_TOKENS=64
export MIXED_SLO_MAX_DEFER_STEPS=8
vllm serve "$MODEL" --enable-chunked-prefill --max-num-batched-tokens 1024 --max-num-seqs 32
```

| 环境变量 | 含义 |
| --- | --- |
| `MIXED_SLO_MODE` | 默认 `off`，执行原生分配。`observe` 执行原生分配并采集 trace，无需 profile。`adaptive` 读取已拟合的 profile 并限制混合步。 |
| `MIXED_SLO_PROFILE` | `adaptive` 必填的 v1 JSON。profile 包含分桶、样本数门槛、已有桶的 p95、时钟及校准元数据。运行前核对 GPU、模型修订、dtype、KV 配置和引擎参数。 |
| `MIXED_SLO_TRACE` | 每步 JSONL。父目录须存在，采用追加和行缓冲。`observe` 必填，`adaptive` 可选。做性能对照时各组应使用相同采集方式。 |
| `MIXED_SLO_RUN_ID` | 写 trace 时必填，每个独立运行使用一个唯一 ID。 |
| `MIXED_SLO_TARGET_MS` | 默认 50，要求有限正数。控制器用于挑选预算的整步耗时目标，单位毫秒。用户可感知的 TTFT、ITL、TPOT、goodput 通过客户端测量。 |
| `MIXED_SLO_FALLBACK_TOKENS` | 默认 64，正整数。混合步缺乏校准数据时的配额，也是等待保护触发时的最高 prompt 配额。仍受原生剩余额度限制。 |
| `MIXED_SLO_MAX_DEFER_STEPS` | 默认 8，非负整数。在连续指定数量的步中有合格 prompt 尝试因配额被截断且实际 prompt 进展为零，则下一个有剩余额度的零预算混合步允许一次进展。设为 0 时立即允许。 |

## 一步如何工作

1. 在任何 KV 分配之前读取 running 列表。把已经计算完 prompt 且仍有 token 可计算的请求作为当前 ready decode，得到数量 `d` 和其已计算上下文长度之和 `c`。首次 prompt 完成时生成第一个输出 token，对本步计费的是实际执行的 prompt token。下一个解码步才出现 ready decode。
2. 纯 prefill 步沿用原生上限。混合步最多可分配 `max(0, native_token_budget − d)` 个 prompt token，给每个 ready decode 留出一个 token 的空间。保留原生 FCFS、KV 内存分配和最大驻留序列数规则。
3. 将 `d`、`c` 及预算候选映射至向上取整的桶，按候选升序读取有足够样本的 p95。低于或等于目标的最大可行候选成为预算。首个已测超时的候选终止搜索。仅 decode 的已测成本已超过目标时返回零。没有可用的正候选测量时回退至配置的固定配额。profile 对一个桶的估计利用桶内样本，迁移到其它模型、硬件或工作负载时要再测。
4. 原生调度循环先处理已驻留请求，再接纳等待请求。两个循环读取同一份本步配额，仅在成功分配 KV 并提交执行量后扣账。prompt 成本为 `min(actual_grant, max(0, prompt_length − computed_length))`。等待队列使用 vLLM 真实缓存命中后的 computed length。一个 prompt 被切短后不能跨过未计算的数据直接执行其后续 token。前面某个 running prompt 暂时没有预算时继续扫描 running 列表，保留下游 decode 的机会。
5. 根据是否确实拦住了 prompt 并造成零进展更新推迟计数，发生 prompt 进展时清零。该保护针对整体 prompt 工作。容量满、KV 分配失败、优先处理更早的请求都可能继续推迟某个请求。实验必须在客户端检查长 prompt 的等待时间和尾部 TTFT。

例：原生上限 16，running 中有 3 个 prompt token 尚未计算，其后还有 1 个 ready decode。等待请求的 30 个 prompt token 已有 16 个前缀命中。本步 profile 配额为 6 时，依次分配 3 个已有 prompt、1 个 decode 和 3 个新请求 prompt。下步该新请求的已计算长度为 19。配额为零时单独执行 decode，waiting 请求仍在队列。

## 测量和可解释边界

- `schedule_ms` 记录读取事前特征至 `schedule` 即将返回的主机墙钟耗时。`step_ms` 使用同一开始时间，结束点是 `update_from_output` 收到模型执行结果之初，覆盖调度与执行器等待。测量结束后写 trace。时钟标识为 `scheduler_result_wall`。
- 每行包含 `run_id`、`decode_count`、`context_tokens`、`prefill_tokens`、`step_ms`、`schedule_ms`、`cap_tokens` 和 `reason`。其中 `prefill_tokens` 为实际新执行量；`cap_tokens=null` 表示该步使用原生预算。`reason` 可为 `native`、`profile`、`uncovered`、`decode_over_target`、`progress`。
- 空步、抢占步、混有输出 token 多步重算的步，以及实际 decode 数量或缓存上下文总和与事前特征不一致的步会被排除，避免用错自变量拟合耗时。`excluded_cumulative` 在下一条有效记录中给出各原因累计次数。运行尾部最后一条有效记录后的排除数不会再写入文件；客户端全量指标仍应覆盖对应时间段。
- 三个特征无法完全区分短 prompt 批量拼接、长上下文单请求、前缀命中以及内核形状差异。持出运行分别检查混合步支持率和“预测达标但实际超时”的比例。稳定的 p95 拟合也不能单独证明端到端有提升。
- 如果 KV 压力导致抢占后重算，历史输出 token 仍然按 vLLM 原生逻辑执行。配额只扣本轮的 prompt 部分，重算可能使这一轮超过 step 目标。若 trace 中抢占计数大，应同时核查 GPU cache 使用量、端到端尾延迟和被排除步数，再决定是否降低并发或调整 KV 配置。比较前后保持请求集合相同。

## 可重复的调度回归

常规单测只需 Python 标准库。要执行实际 v0.10.2 调度循环回归，可把未修改的官方源码提供给测试。加载前固定 SHA-256。测试直接运行上游 `schedule` 与 `_update_after_schedule`，使用 CPU 测试替身提供 KV 分配、请求队列容器和输出传输依赖。测试运行不依赖 CUDA。

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm/v0.10.2/vllm/v1/core/sched/scheduler.py -o /tmp/vllm-v0102-scheduler.py
MIXED_SLO_UPSTREAM_FIXTURE=/tmp/vllm-v0102-scheduler.py python -m unittest discover -s tests -p 'test_engine*.py' -v
```

用例验证上游、关闭模式与观察模式在同一组请求上的分配一致；激活策略后两个入口共享配额；零配额保留 decode；prefix 命中按新计算量记账；超时、缺桶、进展保护和回放过滤按预期工作；重复安装、回滚和未知修改检测符合约定。端到端 GPU 服务验证与客户端 workload 对照需要另行执行。
