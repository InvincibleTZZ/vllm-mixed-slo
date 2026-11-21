# 在旧 Linux 节点上定位 CUDA 初始化失败

**在一台 NVIDIA A800 上，宿主 glibc 2.17 无法启动实验需要的 Python 3.12。项目使用私有解释器和 glibc 2.35 后，显式初始化原生驱动的进程成功分配显存，PyTorch 的 GPU 矩阵乘法返回 `2.0`。** 在同一环境里，直接用 PyTorch 首次分配 GPU 张量则遇到了驱动初始化错误。下面记录观察到差别的具体步骤，供复现和面试复盘。

## 两个版本号和两层依赖

| 环境项 | 本次配置 | 对复现的影响 |
| --- | --- | --- |
| Linux 宿主 | kernel 3.10，系统 glibc 2.17 | 原有 Python 3.12 ELF 需要至少 glibc 2.28。建立普通 venv 的第一步就需要执行这个解释器。 |
| 项目进程 | 专用解释器副本，匹配的用户态动态加载器与 glibc 2.35 | 该进程使用私有运行库，仍在宿主 3.10 内核上执行。子进程及其加载的原生扩展同样受动态链接和系统调用兼容性约束。 |
| GPU 与驱动 | NVIDIA A800，SM 8.0，驱动 R550 | 实际矩阵乘法加载宿主原生 `libcuda`，其 Driver API 报告 12.4。另行检查了官方 compat 库，二者的观察结果应标明各自来源。 |
| PyTorch | PyTorch 2.8，CUDA 12.8 构建（cu128） | `torch.version.cuda` 记录 PyTorch 构建所用的 CUDA 版本。驱动可支持的 API 版本由 `cuDriverGetVersion` 查询。 |

值得实测的原因是 CUDA 12.x 提供有限的次版本兼容性。R550 可以执行部分针对 SM 8.0 生成的机器码。若实际后端要求较新的驱动功能，或把较新工具链生成的 PTX 交给旧驱动即时编译，执行仍可能失败。每个算子和服务进程实际使用的代码路径决定最终结果。[NVIDIA 次版本兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html) · [CUDA Driver API 版本查询](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/group__CUDA__VERSION.html)

## 改变第一条 CUDA 调用

每种顺序分别使用新 Python 进程。同一进程一旦初始化了驱动，就无法再用它重现首次调用时的状态。

| 单独一次运行的调用顺序 | 实际观察 |
| --- | --- |
| `import torch`，由首次 GPU 张量分配触发 CUDA 初始化 | PyTorch 报 `CUDA driver initialization failed`。该次日志仅保留了错误文本，数值错误码未知。 |
| `import torch`，在首次 CUDA Runtime 调用前显式执行 `cuInit(0)`，再调用 Runtime API | `cuInit` 成功，`cudaGetDeviceCount`、`cudaSetDevice`、`cudaMalloc` 的返回码依次为 `0, 0, 0`。随后 GPU 矩阵乘法返回 `2.0`。 |
| 先对原生 `libcuda` 执行 `cuInit(0)`，再 `import torch` 并执行 GPU 算子 | `cuInit` 成功，经 PyTorch 在 GPU 上计算并读回矩阵乘法结果 `2.0`。 |

这组对照把故障缩小到了旧节点上的首次 CUDA 初始化路径。显式初始化放在 `import torch` 前后都能成功，公共条件是它先于首次 Runtime 操作发生。这里观测到的是调用顺序与结果的关系。要进一步定位失败发生在哪次库调用，还需相应的底层错误和加载记录。[CUDA 驱动初始化接口](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-driver-api/group__CUDA__INITIALIZE.html)

## 最小复核

下面给出重新检查三种顺序的 Bash 示例。在仅有一张已分配 GPU 对当前进程可见、私有解释器可启动、库路径已选定的前提下，将 `PY` 设为该解释器。使用没有自动执行 `cuInit` 的启动配置，并为三次运行固定相同的库路径。这里的示例矩阵形状只用于快速核对数值。每个分支都会启动一个新进程，任一分支失败时保留其退出状态和异常文本。

```bash
PY=/absolute/path/to/project-python
for order in runtime_first after_torch before_torch; do
  echo "--- $order ---"
  if "$PY" -B - "$order" <<'PYTHON'
import ctypes as C
import sys

order = sys.argv[1]

def initialize_driver():
    driver = C.CDLL("libcuda.so.1")
    driver.cuInit.argtypes = [C.c_uint]
    driver.cuInit.restype = C.c_int
    rc = driver.cuInit(0)
    print("cuInit", rc, flush=True)
    if rc != 0:
        raise RuntimeError(f"cuInit returned {rc}")
    return driver

if order == "before_torch":
    driver = initialize_driver()
import torch
if order == "after_torch":
    driver = initialize_driver()

x = torch.ones((1, 2), device="cuda")
y = torch.ones((2, 1), device="cuda")
result = (x @ y).item()
print("torch", torch.__version__, "build_cuda", torch.version.cuda, "result", result)
assert result == 2.0
PYTHON
  then echo "exit 0"; else echo "exit $?"; fi
done
```

要定位到首次显存分配，可在 `after_torch` 的 `cuInit(0)` 后单独检查同一进程加载的 CUDA Runtime。按表中的顺序记录 `cudaGetDeviceCount` 的返回码及设备数、`cudaSetDevice(0)` 的返回码、`cudaMalloc` 的返回码，最后释放成功分配的内存。参数 `0` 指可见设备列表中的逻辑序号。将运行时库来源一并记录，有助于识别原生驱动库与 compat 库是否混用。

## 应用到服务进程

标准部署可在主机直接运行兼容的解释器，按仓库说明安装固定版本 vLLM 和调度补丁。已有共享环境的克隆工具也以“源 Python 能启动”为前提。它复制 vLLM 包并复用源环境的 PyTorch 等依赖。此次额外引入的私有 glibc、动态加载器和解释器属于旧节点的环境适配，复用这条路径还需保证源依赖在运行期间可读，并验证新拉起的 worker 选择了相同运行库。[标准 GPU 实验](EXPERIMENT.md) · [共享环境的依赖关系](SHARED_ENV.md)

单进程探针与 vLLM V1 的进程模型有实际差异。服务启动后，模型加载、attention 后端和调度运行在相应的引擎或 worker 进程中。父进程先初始化 CUDA 再 `fork` 会影响子进程的初始化，移植时应记录 worker 的启动方式，并在实际执行 CUDA 的进程验证。模型热身、流式生成和同一 trace 的成组压测分别有各自的验收记录。[PyTorch 2.8 多进程说明](https://docs.pytorch.org/docs/2.8/notes/multiprocessing.html#poison-fork-in-multiprocessing) · [项目实验报告](RESULTS.md)

**公开证据的表述可写成**「在 A800 / R550、私有 glibc 2.35、PyTorch 2.8 cu128 环境中，对照首次 CUDA 调用顺序；在原生驱动上验证显存分配成功和矩阵乘法结果 `2.0`」。发布该记录时保留宿主及进程内 glibc、驱动与 CUDA 构建版本、原生或 compat 库选择、调用顺序、返回码和 GPU 计算读回值。模型 revision、服务 warmup、配对输出、TTFT/ITL、吞吐和 goodput 则从完成相应运行后的实验产物中引用。[公开证据说明](PUBLISH_EVIDENCE.md)
