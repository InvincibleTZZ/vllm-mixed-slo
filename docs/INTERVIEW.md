# 如何在 AI Infra 面试中讲清这个项目

## 60 秒介绍

长 prompt 加入正在逐 token 输出的批次时，单次迭代要同时处理 prefill 和 decode。批次做得大可以减少分块次数，但已经开始输出的用户可能等更久才看到下一个 token。这个项目针对 vLLM 0.10.2 V1 的同步单卡调度，研究一个具体问题：给定当前 decode 负载，这一轮还能处理多少 prompt token？

离线阶段采集引擎从调度开始到收到模型执行结果的耗时，按就绪 decode 数、它们的上下文 token 总量和实际加入的 prefill token 数做分桶，估计条件 p95。在线阶段按目标毫秒数选共享 prefill 预算，覆盖运行中和新入队的两个分配入口。评测端回放固定到达时间、固定输入输出长度的短长混合请求，分别看两类请求的 TTFT、TPOT、失败率和 SLO goodput。可以先画出这三个阶段，再给面试官看与兴趣最相关的代码。[调度](../engine/policy.py) · [校准](../calibration/table.py) · [压测](../bench/README.md)

## 面试官深挖时怎么回答

### 1. vLLM 已经支持 chunked prefill，为什么还要加策略？

chunked prefill 让一条长 prompt 可分多个引擎迭代执行。`max_num_batched_tokens` 为整轮处理的 token 数设固定上限。在相同上限下，1 个短上下文 decode 和 16 个长上下文 decode 会消耗不同的资源。这里保留上游总预算和 FCFS，在有 decode 的轮次进一步决定本轮共享的 prompt 份额。纯 prefill 轮直接使用原生调度。代价是长 prompt 可能经历更多轮，TTFT 可能变差，因此需要同时观察两个类别。

### 2. 为什么用三个特征？prefill 和 decode 的计算强度如何解释？

prefill 可让较多 query token 共用模型权重读取，常能获得较高矩阵运算利用率。decode 每请求每步通常生成一个 token，要读取权重和历史 KV，在低 batch 或很长上下文下带宽压力会很明显。decode 并行序列数刻画并行度，它们已计算的上下文长度之和粗略刻画 KV 读取工作，新增 prompt token 数刻画这一轮加入的 prefill 工作。相同总上下文长度仍可能由不同长度分布构成，后端、精度、prefix 命中及批次形状也会改变实际耗时。回答时把这些当作需要测量的因素。要定位内核瓶颈，可补充 CUDA events、Nsight Systems 时间线或 Nsight Compute 的 DRAM 吞吐与算力利用率，再与主机端时钟对齐。

KV cache 减少后续 decode 中反复计算历史 token 的 K/V。对单张保存全部 KV 头的 GPU，普通不量化 KV 的有效缓存字节约为 `2 × 层数 × 有效上下文 token 总数 × KV 头数 × head_dim × 每元素字节数`；2 对应 K 和 V。GQA 用 KV 头数计算。PagedAttention 的 block table 把逻辑块映射到物理 KV 块，实际分配还受 block 对齐、共享前缀和预分配影响。本项目调的是提交给原生 KV 分配流程的 token 预算，保留上游块映射与注意力算子。

### 3. 把算法写成约束问题，会怎么写？

设一轮最多处理 B 个 token，当前有 d 个就绪 decode，累计上下文长度为 c。先给 decode 留出空间，候选 prefill 量 p 满足 `0 ≤ p ≤ max(B−d, 0)`。目标是在有观测支持的候选上尽量增大 p，同时拟合的 `Q95(step_ms | d, c, p)` 小于等于预设的步时延目标。实际选取的 p 还受剩余 prompt 长度、cache 和上游分配约束限制。这里的 Q95 是历史样本的经验分位数，没有未来违约概率的分布无关保证。

实现用各轴包含上界的离散分桶，稀疏格和越界状态没有预测值。固定 d、c 时，对样本数足够的 p 桶按 p 从小到大取累计最大值，避免有限样本偶然让更大的 p 显得更便宜。decode-only 已超过目标时先把 prefill 压到零；可选区域完全无覆盖时退回固定小块。相关性会令同一 run 的相邻步骤表现相似，验证按完整 run 留出。除了报告 coverage，还要报告预测达标的已覆盖步中实际超标的比例，并列出其分母。[离线校准与验证](../calibration/README.md)

### 4. 这个参数到底怎么进入真实 vLLM 调度循环？

阅读固定版本 `Scheduler.schedule()`，找到处理 running 请求和 waiting 请求的两个入口。两处都在原生计算 proposed tokens 后调用同一轮的 `limit`。本轮共享一个已花费计数器，原生 KV 分配成功并真的接纳了对应工作后，才 `charge` 提交。已经命中的前缀不应二次计作本次新增 prefill。被恢复的请求还可能同时包含剩余 prompt 与后续重算，必须尊重 prompt 的前置依赖。新增逻辑对普通 decode token 不做 prefill 扣费。安装器限定上游源码的 SHA-256，检查每个补丁锚点唯一、编译生成文件、原子替换，还可恢复原始字节。[适配器](../engine/install.py) · [固定版本调度源码](https://github.com/vllm-project/vllm/blob/v0.10.2/vllm/v1/core/sched/scheduler.py)

可以画一个上游循环测试中的批次：一个 running 请求剩 3 个 prompt token，一个请求 decode 1 个 token，一个 waiting 请求命中 16 个前缀 token。若本轮 prefill cap 为 6，就允许 running prompt 用 3、decode 用 1、waiting 再用 3。两个 prompt 合计 6；waiting 计数可以推进到 19。这里 CPU 测试实际执行上游调度循环，KV 分配和 GPU 执行用测试替身。[执行原始循环的测试](../tests/test_engine_install.py)

### 5. 限制 prefill 后，会不会让长请求饿死？

有这个风险。实现记录连续因限额拒绝 prefill 且完全没有 prefill 进展的轮数。达到上限且有原生可分配空间时，放行一个有界 fallback 块，然后重新计数。这是对全局 prompt 进展的缓解。它不能保证每个请求在指定毫秒数内被服务；队首阻塞、KV 可用量和负载仍然有影响。应检查长类 p95/p99 TTFT、超时和 goodput，观察恢复进展的轮次是否多到说明目标设得不现实。证明单请求公平性还需逐请求等待时间模型或单独的期限/老化规则。[策略和进展测试](../tests/test_engine_policy.py)

### 6. 如何避免测到的只是客户端更快，或失败请求从统计里消失？

使用开环重放，客户端根据同一时钟上的预约到达时间发起请求。闭环中上一请求越慢，下一请求越晚到达，会自动减小背压。保存 raw JSONL，每个到达都占一行；拒绝、超时和异常进入 SLO 分母。两种策略使用一致的输入 ID、输出目标、模型、引擎配置和热身方式。检查客户端的实际发起滞后、并发上限，以及两边共同成功请求的实际生成 token ID。固定 token 长度的合成负载提高了重复性，真实前缀复用、自然停止和不同服务实例的泛化需另做实验。

三个容易混的数字可以当场推导。TTFT 是首个可见含 token 事件减原定到达时间。每请求 TPOT 是首尾 token 事件之间的时间除以后续 token 数；首事件含多个 token 时该工具单独报告观测缺失。ITL 是相邻可见 token 间隔，流式合并事件会丢失包内时间。双目标 goodput 是在各自类别 TTFT、TPOT 上都达标的成功请求数除以整次公共时间窗，包含收尾排空。它的单位是请求/秒；仅时延达标率的单位是百分比。[原始记录与指标字典](../bench/README.md)

### 7. 如果面试官说“降了每步 p95 就等于改善用户体验”？

单步目标是内环控制量。用户 TPOT 横跨多步，还有输出传输开销。TTFT 额外包含客户端调度、请求排队和完整 prompt 处理。一种策略可能缩短 mixed 步时长，却拉长长 prompt 处理所需轮数。回答可以给一个具体可证伪的预测：在中等混合负载下，decode 的长间隔有机会减少；在几乎全是 prefill 的负载或 fallback 频繁出现时，收益可能接近零。用多种静态预算的原生基线、短/长/混合/突发负载、独立 trace 的配对重复实验检验。小样本只报告观察结果和分母，避免把上限外推到其他吞吐区间。

## 可复现的失败案例

下表是仓库能重放的反例或边界，适合回答「你怎么发现指标会骗人」和「这个设计什么时候退化」。其中耗时、序列和失败响应由测试控制。

| 问题与触发条件 | 如果忽略会发生什么 | 项目处理与证据 |
| --- | --- | --- |
| 一个 SSE 事件带两个 token | 包间隔混进 token ITL；首包含两个 token 时尚无独立的首 token 时间 | 以 `token_ids` 个数辨识每个事件。4 个首包合并的请求，strict TPOT 样本数 0，SLO 通过 0，同时保留 chunk gap。`test_multitoken_chunks_are_not_single_token_itls`。[测试](../tests/test_bench.py) |
| 4 个请求同时到达，客户端容量 1，服务停顿 400 ms，总截止 80 ms | 只统计完成项会丢掉所有失败；静默排队还会改变预设到达强度 | 结果为 3 个 `client_overload`、1 个 `timeout`、4 个 offered，达标率 0。`test_deadline_and_client_overload_remain_offered`。[测试](../tests/test_bench.py) |
| 小样本表看起来「都能满足目标」 | 同一工作负载拟合和评估会高估覆盖率，已覆盖桶也会错判 | 合成留出集共有 5 步，其中 3 步有覆盖，3 次预测达标中 1 次实际超标；重复使用训练 run ID 会拒绝验证。`test_cli_reproducible_fit_holdout_report_and_invalid_input`。[测试](../tests/test_calibration.py) |
| running 和 waiting 各自能批一份 cap | 分配路径各自正确，合并后的预算却超出目标 | 单轮共享预算，分配成功后扣账；用上游循环检查两入口合计 6 个 prefill token 和一个 decode。`test_both_admission_paths_share_one_budget_and_cached_tokens`。[测试](../tests/test_engine_install.py) |
| decode 持续超过目标，把 prefill 连续压到 0 | 保护输出间隔的同时，排队 prompt 无进展 | 在可用空间内放行有界块；测试中连续阻止两轮后放行 2 个 token。需要在真实负载中同时看长类 TTFT 的成本。`test_progress_pulse_and_pure_prefill`。[测试](../tests/test_engine_policy.py) |
| 只计调度函数返回时间 | 会用 CPU 调度开销预测包含设备执行的完整迭代，校准目标不同 | 测试控制时钟，让 schedule 为 1 ms、调度到结果为 21 ms，训练字段记录后者；抢占、重算和 decode 特征失配步另外计数。`test_observe_wall_includes_runner_and_trace_agrees_with_fit`。[测试](../tests/test_engine_policy.py) |

## 简历可用表述与证据

**vLLM V1 混合负载调度与 SLO 评测｜Python、vLLM、流式服务、性能建模**

- 基于 vLLM V1 同步单卡调度器实现按迭代调整的 chunked prefill 预算，按就绪 decode 数、已计算上下文总量和待分配 prefill token 数估计条件 p95，支持覆盖不足回退与有界进展机制。
- 实现可复放的长短混合及突发流量压测，按类别输出 TTFT、TPOT、ITL 与 SLO goodput，记录每次到达的超时、失败和输出 token，用固定版本上游循环和真实 HTTP 流式测试覆盖预算与统计口径。

这两条描述的是仓库可检查的功能。CPU 测试覆盖控制逻辑、固定版本调度循环和客户端协议；synthetic fixture 的毫秒数只用于检验计算；GPU 性能陈述需要来自同一设备、模型及重放 trace 的实测原始记录。只有亲自复现并能解释对应代码后，再按自己的实际工作范围使用「实现」或「复现并扩展」等动词。若要增加结果句，保留硬件、模型精度、固定基线与所选预算、负载强度、样本量、双类阈值及短/长两类得失。没有成对数据时，上述工程描述就可以独立使用。
