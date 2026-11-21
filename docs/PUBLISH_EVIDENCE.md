# 公开可复算的实验结果

从仓库根目录执行。输入目录是 `scripts/experiment.py` 产生的私有矩阵；目标须是矩阵以外的全新目录。可选 `--model-name` 填可公开的模型名称，省略时使用 `local-model`。

```bash
python scripts/publish_evidence.py export --matrix runs/matrix --out evidence/pilot --model-name org/public-model
python scripts/publish_evidence.py verify --bundle evidence/pilot
```

`index.json` 给出原始计划的公共实验条件、按类别生成输入所需的种子/形状、每个工作负载的 SHA-256 以及所有已识别试验尝试。未执行的试验对应空数组，启动中和失败尝试保留状态，失败原因只输出固定分类。输入是确定生成的合成 token，verify 会按原生成逻辑逐行重建输入，核对全部 trace 哈希；无需公开重复的 prompt、权重和服务原始日志。每条实际完成的请求保留客户端所用的后端标签，使用 mock 时仍显示 mock。

`trials/<trial>/attempt-<n>/requests.jsonl.gz` 保存输入请求 ID、长度、预约/派发/完成时间、数值 token ID、流事件、生成结束原因及状态。`steps.jsonl.gz` 仅从已隔离热身的实测步文件导出，含分桶特征、墙钟时延、预算及原因。导出器只保留列明的数值和状态，对错误文本归类。设备只报告可识别的型号系列、显存容量和数字版本。若私有计划记录了权重和分词器文件的快照指纹，`implementation.model_snapshot_sha256` 会保留该指纹；另保留引擎补丁与实验代码的 SHA-256。

`report.json` 列出完整尝试的分组指标、失败分母和相同工作负载下与 off 基线逐请求的输出比较。报告同时记录已完成数与计划数。残缺尝试仅有已记录的状态计数。收齐训练和留出阶段时，从公开步记录重拟合经验表并计算留出覆盖与超标比例。若私有矩阵已有冻结 profile 或单臂汇总，会比对校准单元格及原始指标。verify 检查 `sha256.json` 以及全部报告数值。单次 quick 矩阵的复算通过，说明该次有限样本的数值与公开记录一致。

单独考察某个 arm 时，可把对应的 gzip 解压为 JSONL，在仓库根目录执行 `python -m bench summarize`，传入 `index.json` 中的同一组 SLO。
