# 在共享 vLLM 环境上建立私有调度器副本

服务器已有与 CUDA 驱动匹配的 vLLM 0.10.2、PyTorch 和原生扩展时，可以在用户可写目录中创建一个私有 Python 环境。程序复制安装好的 vLLM 包和对应的 wheel 元数据，在私有副本上应用本仓库的调度器补丁。PyTorch、NVIDIA 库及其他 Python 依赖继续从公共环境读取。这样可以在保持已有二进制兼容性的同时，使用私有调度器运行实验。

## 创建与运行

在 Linux 上进入仓库根目录。选择实际环境内的绝对解释器路径，包括它是符号链接的情形。真实 Python venv 的 `bin/python` 常常链接到基础解释器，对它执行 `readlink -f` 后当作参数会选中另一个环境。

```bash
PUBLIC_PY=/absolute/path/to/public-env/bin/python
PRIVATE="$HOME/.local/share/vllm-envs/mixed-slo-0102"
python -B -m scripts.clone_runtime --python "$PUBLIC_PY" --out "$PRIVATE"

"$PRIVATE/bin/python" -m engine.install status
"$PRIVATE/bin/python" -m scripts.experiment --model /absolute/path/to/pinned-model \
  --gpu 0 --out runs/private-quick --quick --dry-run
"$PRIVATE/bin/python" -m scripts.experiment --model /absolute/path/to/pinned-model \
  --gpu 0 --out runs/private-quick --quick
```

`--out` 必须是尚不存在的新路径，与公共环境前缀、基础解释器前缀及依赖目录互不嵌套。程序以独占方式创建私有目录，普通异常发生时清理本次新建的目录。若上一进程被强制中止并留下中间文件，可选用另一个新路径重试。虚拟环境中的激活脚本和缓存配置包含绝对路径，移动后需要重新创建。构建和 dry-run 均无需 GPU。

## 具体检查了哪些事

1. 以指定解释器运行隔离的只读探针，读取其实际 `sys.prefix`、`sys.base_prefix`、有效 site 路径、vLLM 导入位置和安装元数据。探针不导入 vLLM。源安装需包含标准 wheel 的 `.dist-info/METADATA` 和 `RECORD`，版本为 0.10.2，可带本地构建后缀。公共 scheduler 必须命中项目锁定的官方源码 SHA-256。安装包的其他 Python 和二进制文件以该环境中的实际版本为准。
2. 由源解释器运行 `venv --without-pip`，在目标位置建立同一 Python ABI 的环境。程序查询新解释器自己的 `purelib`，在其中创建实际文件副本，跳过源 `__pycache__`，并使用原子替换给新调度器打补丁。全程不调用 pip、不创建指向公共包文件的硬链接。对安装目录的符号链接子目录会报错，避免意外复制安装范围外的目录树。
3. 在新解释器中先搜索私有 site，再搜索源解释器报告的依赖路径。程序使用 `find_spec("vllm")` 验证导入路径和元数据都属于私有前缀，重新计算公共 scheduler 的 SHA-256，同时校验私有 scheduler 的完整字节与预期补丁一致。
4. 私有 Python 启动时关闭 bytecode 写入，并为其子进程设置相同选项。常见缓存（vLLM、Torch/TorchInductor、Triton、CUDA、Hugging Face、pip 缓存）以及临时文件定向到 `$PRIVATE/cache` 的独立子目录。vLLM 进程应由私有解释器启动；直接使用公共环境的 `vllm` 入口脚本会根据其 shebang 启动公共解释器。实验脚本使用当前解释器启动服务。
5. `$PRIVATE/clone-manifest.json` 记录源/目标解释器与包路径、运行时读取的公共目录、公共 scheduler 操作前后 SHA-256、私有补丁 SHA-256 和缓存路径。可以复查其中的路径与版本，再记录实际模型、驱动、GPU 及实验参数。[解释器验证逻辑](../scripts/clone_runtime.py) · [运行矩阵](EXPERIMENT.md)

## 生命周期与适用范围

私有副本持续依赖公共解释器、标准库、依赖 wheel 和 CUDA 运行库。公共环境被删除、更新或换用不兼容的驱动后，需要重新建立副本并验证原生导入。需要公共环境激活脚本设置的动态链接器路径时，保留其 `LD_LIBRARY_PATH`，再显式运行私有 `bin/python`。若服务依赖额外的可执行 entry point、可编辑安装的导入钩子或自定义启动逻辑，需要独立核对。脚本验证包发现与源码兼容性，实际 CUDA 内核加载和服务返回 token 由 GPU 试验验证。

此模式给已有可信依赖提供只读复用。避免在该私有视图上用 pip 卸载或升级公共依赖；另行安装一套独立环境更适合更换 PyTorch/vLLM 版本。对共享机器上的文件权限有强制隔离要求时，应由系统对公共目录提供只读权限。缓存环境变量约束对应工具的常规写入行为，自定义第三方代码仍受实际文件系统权限约束。

## 无 GPU 的验证

```bash
curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm/v0.10.2/vllm/v1/core/sched/scheduler.py -o /tmp/vllm-v0102-scheduler.py
MIXED_SLO_UPSTREAM_FIXTURE=/tmp/vllm-v0102-scheduler.py python -m unittest discover -s tests -p 'test_clone_runtime.py' -v
```

Linux 用例在临时目录内建立一个真实 venv，放入固定上游 scheduler、最小 wheel 元数据和供读取的模拟依赖。验证源 venv 与基础解释器具有不同前缀、嵌套或重复目标被拒绝、私有 vLLM 与公共依赖各自解析到预期位置，以及加载公共依赖时没有向其目录写入字节码。[测试用例](../tests/test_clone_runtime.py)
