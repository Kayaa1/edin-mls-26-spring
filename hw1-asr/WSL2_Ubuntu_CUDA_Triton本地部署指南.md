# WSL2 + Ubuntu + CUDA + Triton 本地部署指南

## 1. 文档目的

这份文档说明如何在当前这台 Windows 笔记本上，搭建一套适合本仓库 `edin-mls-26-spring` 的本地开发环境：

- Windows 宿主机
- WSL2
- Ubuntu
- NVIDIA GPU / CUDA
- PyTorch
- Triton

目标不是“泛泛介绍 WSL”，而是直接给出一条能服务当前作业的落地路线。

---

## 2. 当前机器状态

基于本地检查，截至 `2026-03-16`，当前机器的相关状态如下：

- Windows 版本：`10.0.26200.8037`
- 宿主机可见 NVIDIA GPU：`NVIDIA GeForce RTX 5070 Laptop GPU`
- 当前 `wsl -l -q` 没有列出已安装发行版
- 当前 Windows 默认 `python` 环境里还没有 `torch`

这意味着：

1. 你的机器硬件层面具备本地跑 GPU 的前提
2. 当前主要缺的是 WSL Linux 环境和 Python/ML 软件栈
3. 现在最值得做的是把本地开发环境搭起来，而不是继续等集群

---

## 3. 这个方案适不适合当前作业

适合，而且是当前最推荐的路线。

原因如下：

- 本仓库的脚本明显按 Linux/Bash 工作流设计
- `utils/setup-triton.sh` 直接面向 Linux 环境
- `hw1-asr` 的 Triton 轨道需要真实 CUDA/Triton 环境验证
- Windows 原生 Python 可以勉强凑环境，但可维护性通常不如 WSL2
- WSL2 能提供接近 Linux 的开发体验，同时继续使用 Windows 主系统

对这份作业来说，`WSL2 + Ubuntu + 本地 NVIDIA GPU` 基本上是“最像教学集群但更容易控制”的方案。

---

## 4. 推荐架构

建议采用两阶段策略，而不是一上来就把所有东西都装满。

### 4.1 阶段 A：最小可用方案

先装：

- Windows NVIDIA 驱动
- WSL2
- Ubuntu
- Python 环境
- PyTorch
- Triton

然后直接跑本仓库的：

```bash
source utils/setup-triton.sh
```

这是最快能让仓库开始工作的方式。

### 4.2 阶段 B：完整 CUDA 工具链方案

如果后续你需要更完整的 Linux CUDA 开发工具，再在 WSL 里补装：

- WSL 专用 CUDA Toolkit

但注意，这一步不是最开始的硬性前提。

对本仓库来说，建议先跑通阶段 A，再决定是否需要阶段 B。

---

## 5. 关键原则

在正式步骤前，先把几个关键原则讲清楚。

### 5.1 不要在 WSL 里安装 Linux NVIDIA 驱动

这是官方文档最强调的一点。

NVIDIA 在 CUDA on WSL 文档中明确指出：

- Windows 上安装的 NVIDIA 驱动会映射到 WSL2 里
- WSL2 内部不要再安装 Linux display driver
- 否则容易把映射进来的 WSL 驱动环境覆盖掉

因此：

- 驱动装在 Windows 宿主机
- WSL 里不要装 Linux GPU driver

### 5.2 如果要在 WSL 里装 CUDA Toolkit，只装 WSL/Toolkit 包，不装驱动元包

官方文档明确提醒：

- 不要在 WSL2 下安装 `cuda`
- 不要安装 `cuda-12-x`
- 不要安装 `cuda-drivers`
- 如果走 apt 元包路线，只安装 `cuda-toolkit-12-x`

这条规则非常重要。

### 5.3 代码仓库尽量放在 Linux 文件系统里

微软官方 WSL 最佳实践明确建议：

- 用 Linux 工具开发的项目，放在 WSL 的 Linux 文件系统里更快
- 不建议把主要开发目录长期放在 `/mnt/c/...`

所以这个仓库建议放在类似：

```bash
/home/<your-user>/src/edin-mls-26-spring
```

而不是：

```bash
/mnt/c/Users/<your-user>/...
```

---

## 6. 官方依据

以下结论来自官方文档：

- Microsoft 官方 WSL 安装文档说明可以用 `wsl --install` 一步安装，并且新安装默认就是 WSL2
- NVIDIA 官方 CUDA on WSL 文档说明：Windows 驱动装好后，CUDA 即可在 WSL2 中可用；不要在 WSL 中安装 Linux 驱动
- Triton 官方安装文档说明：稳定版可直接 `pip install triton`
- PyTorch 官方安装文档说明：Linux 上使用 pip 安装，并按 CUDA 平台选择合适安装方式

文末给出官方链接。

---

## 7. 推荐安装路线总览

完整流程建议如下：

1. 在 Windows 上更新 NVIDIA 驱动
2. 在管理员 PowerShell 中安装/更新 WSL2
3. 安装 Ubuntu 发行版
4. 首次进入 Ubuntu，创建 Linux 用户
5. 在 Ubuntu 中安装基础工具
6. 把仓库克隆到 WSL 的 Linux 文件系统
7. 用仓库自带脚本安装 Triton 环境
8. 验证 `nvidia-smi`、`torch.cuda.is_available()`、`triton`
9. 运行 Triton tutorial 环境检查
10. 运行 `hw1-asr` 的 Triton baseline

---

## 8. 详细步骤

## 8.1 Windows 侧：确认和更新 NVIDIA 驱动

### 目标

确保 Windows 宿主机已经安装可支持 WSL2 CUDA 的 NVIDIA 驱动。

### 你现在的状态

宿主机已经能看到：

```text
NVIDIA GeForce RTX 5070 Laptop GPU
```

说明驱动至少不是完全缺失状态。

### 建议动作

1. 打开 NVIDIA 官方驱动下载页面
2. 按你的 GPU 型号下载最新稳定版 Windows 驱动
3. 安装完成后重启 Windows

### 验证

在 Windows PowerShell 中运行：

```powershell
nvidia-smi
```

如果能看到 GPU、驱动版本、显存信息，说明宿主机驱动层正常。

---

## 8.2 Windows 侧：安装 WSL2

### 目标

安装 WSL 及 Ubuntu 发行版，并确保默认版本为 WSL2。

### 官方推荐命令

以管理员身份打开 PowerShell，执行：

```powershell
wsl --install
```

如果你想先看可选发行版：

```powershell
wsl --list --online
```

如果你要显式安装某个发行版：

```powershell
wsl --install -d Ubuntu
```

如果安装过程卡在 `0.0%`，微软官方文档建议尝试：

```powershell
wsl --install --web-download -d Ubuntu
```

### 建议再执行

```powershell
wsl --update
wsl --set-default-version 2
```

### 验证

```powershell
wsl -l -v
```

你希望看到类似：

```text
NAME      STATE    VERSION
Ubuntu    Stopped  2
```

如果显示的是 `1`，则改成：

```powershell
wsl --set-version Ubuntu 2
```

---

## 8.3 发行版选择建议

### 推荐结论

优先选 Ubuntu LTS。

### 建议顺序

1. `Ubuntu-22.04` 或 `Ubuntu 22.04 LTS`
2. `Ubuntu-24.04` 或 `Ubuntu 24.04 LTS`
3. 如果商店 / 在线列表里只有 `Ubuntu`，也可以先装默认项

### 为什么这样选

这不是官方强制要求，而是工程判断：

- Ubuntu 在 WSL 生态中最常见
- CUDA / PyTorch / Triton / HuggingFace 在 Ubuntu 上资料最多
- 课程仓库脚本默认按 Ubuntu/Bash 习惯写

如果你主要目标是尽快跑通作业，而不是折腾发行版差异，Ubuntu 是最稳的。

---

## 8.4 首次进入 Ubuntu

首次启动 Ubuntu 时，系统会要求你设置：

- Linux 用户名
- Linux 密码

这是 WSL 内部的 Linux 用户，不是你的 Windows 用户密码。

完成后，先更新系统包：

```bash
sudo apt update
sudo apt upgrade -y
```

再装基础工具：

```bash
sudo apt install -y \
  build-essential \
  git \
  curl \
  wget \
  ca-certificates \
  pkg-config \
  python3 \
  python3-pip \
  python3-venv
```

建议再做一个命令别名适配：

```bash
python3 --version
pip3 --version
```

如果你想让 `python` 和 `pip` 直接可用，可以自己做软链接，但这不是必须。

---

## 8.5 把仓库放到 WSL Linux 文件系统

### 推荐目录

```bash
mkdir -p ~/src
cd ~/src
git clone <repo-url> edin-mls-26-spring
cd edin-mls-26-spring
```

如果仓库已经在 Windows 盘里，也建议重新在 WSL 里克隆一份，而不是长期直接用 `/mnt/f/...` 开发。

### 原因

WSL 官方建议：

- Linux 工具处理 Linux 文件系统会更快
- 跨文件系统访问通常会慢不少

这对 Triton 编译缓存、Python 依赖、benchmark 文件访问都更友好。

---

## 8.6 用仓库自带脚本安装 Triton 环境

对于当前仓库，最推荐的方式不是自己重新设计环境，而是先用仓库提供的：

- `utils/setup-triton.sh`

这个脚本会：

- 处理 conda / miniconda
- 建立名为 `mls` 的环境
- 安装 `torch`
- 安装 `numpy`
- 安装 `triton`
- 安装 `cupy-cuda12x`
- 安装 `datasets`

### 执行方式

从仓库根目录：

```bash
source utils/setup-triton.sh
```

### 注意

这个脚本是 Bash 脚本，适合在 Ubuntu / WSL Bash 中运行。  
这也是为什么 WSL 路线比 Windows 原生 PowerShell 路线更自然。

---

## 8.7 最小验证：WSL 中是否真正看到了 GPU

完成上一步后，在 WSL Ubuntu 中执行：

```bash
nvidia-smi
```

在 WSL 里，`nvidia-smi` 的功能集可能比原生 Linux 少一些，这是 NVIDIA 官方文档说明过的正常现象。  
重点不是“功能是否完全一致”，而是：

- GPU 是否可见
- 驱动是否可见

### 再验证 PyTorch

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no gpu')"
```

你希望看到：

- `torch.cuda.is_available()` 为 `True`
- 设备名接近 `RTX 5070 Laptop GPU`

### 再验证 Triton

```bash
python -c "import triton; print(triton.__version__)"
```

---

## 8.8 运行仓库自带的环境检查

在仓库根目录执行：

```bash
python triton-tutorial/0-environment/check.py
```

如果这一步失败，不要急着跑 `hw1-asr`，先把这里的问题解决掉。

因为如果最基础的 Triton 检查都过不了，后面 benchmark 通常也跑不稳。

---

## 8.9 运行 Triton baseline

进入作业目录：

```bash
cd hw1-asr
bash benchmark.sh glm_asr_triton_example
```

### 这一步会发生什么

首次运行时通常会有几个“慢”的地方：

- HuggingFace 权重下载
- Triton JIT 编译
- Python 依赖首次导入

这都正常。

### 验收目标

你希望最终看到：

```text
Transcription: Concord returned to its place amidst the tents.
Accuracy: 100.0%
Status: PASS
```

如果 baseline 都过不了，不建议开始改 template。

---

## 9. 关于 CUDA Toolkit：要不要在 WSL 里额外安装

## 9.1 结论

对当前仓库，建议分两种情况：

- 想尽快跑作业：先不急着单独安装完整 Linux CUDA Toolkit
- 想做更完整的 CUDA/调试/编译环境：再安装 WSL 专用 CUDA Toolkit

## 9.2 为什么可以先不装

原因有两点：

1. 本仓库自己的 `setup-triton.sh` 走的是 Python 包路线，不要求你先手装完整 CUDA Toolkit
2. PyTorch 官方文档推荐优先使用预编译二进制包；Triton 官方文档也支持直接 `pip install triton`

因此，对“先把作业跑起来”来说，最小路径通常足够。

## 9.3 什么时候建议再装 WSL 版 CUDA Toolkit

如果你遇到下面这些需求，再补装更合理：

- 需要更完整的 CUDA 编译工具链
- 需要额外 CUDA 工具和头文件
- 需要运行某些依赖系统 CUDA Toolkit 的库
- 需要更系统地调试 CUDA/WSL 环境

## 9.4 如果要装，怎么装才安全

只走 NVIDIA 官方文档推荐路线：

- 使用 `WSL-Ubuntu` 对应的 CUDA Toolkit 安装入口
- 或只装 `cuda-toolkit-12-x`

不要装：

- `cuda`
- `cuda-12-x`
- `cuda-drivers`

### 额外说明

这里我特意写成 `12-x`，是因为本仓库当前的 `setup-triton.sh` 安装的是：

```text
cupy-cuda12x
```

因此从工程一致性看，当前仓库更偏向 CUDA 12 代的软件栈。  
而 NVIDIA 官方站点上的 CUDA Toolkit 页面已经出现了更新版本信息，例如 `CUDA 13.2`。这说明官方 CUDA 版本会继续前进，但对当前仓库而言，不必盲目追最新大版本，优先和仓库依赖保持一致更稳。

这是一个工程判断，不是 NVIDIA 的硬性要求。

---

## 10. 推荐的最小可用命令清单

下面这组命令适合你直接照做。

### 10.1 Windows 管理员 PowerShell

```powershell
wsl --install
wsl --update
wsl --set-default-version 2
wsl --list --online
```

如果需要显式安装 Ubuntu：

```powershell
wsl --install -d Ubuntu
```

安装后重启机器。

### 10.2 Ubuntu / WSL 内

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y build-essential git curl wget ca-certificates pkg-config python3 python3-pip python3-venv
mkdir -p ~/src
cd ~/src
git clone <repo-url> edin-mls-26-spring
cd edin-mls-26-spring
source utils/setup-triton.sh
python triton-tutorial/0-environment/check.py
cd hw1-asr
bash benchmark.sh glm_asr_triton_example
```

---

## 11. VS Code 建议

如果你平时用 VS Code，建议安装：

- `Remote - WSL`

然后在 WSL Ubuntu 里进入仓库目录后执行：

```bash
code .
```

这样你可以：

- 在 Windows 里开 VS Code
- 但实际编辑和运行都发生在 WSL Linux 环境里

这通常是 Windows + WSL 开发里体验最好的一种方式。

---

## 12. 常见坑

## 12.1 在 WSL 里误装 Linux NVIDIA 驱动

这是最危险的坑之一。

后果通常是：

- `nvidia-smi` 失效
- CUDA 环境紊乱
- WSL GPU 访问异常

规避方式：

- 驱动只装在 Windows
- WSL 里不装 Linux display driver

## 12.2 在 WSL 里安装了错误的 CUDA 元包

尤其是这些包：

- `cuda`
- `cuda-12-x`
- `cuda-drivers`

官方文档已经明确提醒这类包会试图把 Linux 驱动装进 WSL。

## 12.3 把项目放在 `/mnt/c` 或 `/mnt/f`

能跑，不代表适合长期开发。

对于：

- Python 包
- Triton 编译缓存
- benchmark
- 大量小文件 IO

Linux 文件系统路径通常更合适。

## 12.4 直接在 Windows PowerShell 里跑仓库 Bash 脚本

仓库里的关键环境脚本是：

- `utils/setup-triton.sh`

它们是按 Bash/Linux 设计的。  
虽然 Windows 上可能有 `bash.exe`，但长期来看还是在 WSL Ubuntu 中执行最稳。

## 12.5 第一次 benchmark 很慢

这是正常现象，原因通常包括：

- 权重下载
- Triton JIT 编译
- Python 缓存未建立

不要拿第一次的时间直接作为性能结论。

---

## 13. 如果安装后还是有问题，怎么排查

建议按下面顺序检查。

### 第 1 层：Windows 宿主机

```powershell
nvidia-smi
```

如果这里都失败，先不要进 WSL 排查 Triton。

### 第 2 层：WSL 是否正确安装

```powershell
wsl -l -v
```

确认：

- 发行版存在
- 版本是 `2`

### 第 3 层：WSL 里是否能看到 GPU

```bash
nvidia-smi
```

### 第 4 层：PyTorch 是否看到 CUDA

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 第 5 层：Triton 是否能导入

```bash
python -c "import triton; print(triton.__version__)"
```

### 第 6 层：仓库环境检查

```bash
python triton-tutorial/0-environment/check.py
```

### 第 7 层：baseline

```bash
cd hw1-asr
bash benchmark.sh glm_asr_triton_example
```

---

## 14. 对当前作业的最终建议

如果你的目标是尽快开始做 `Triton` 作业，我建议：

1. 先完成 `WSL2 + Ubuntu + 仓库 setup-triton.sh` 的最小路线
2. 先把 `triton_example` baseline 跑通
3. 再开始改 `glm_asr_triton_template`
4. 只有在确实需要时，再补装 WSL 专用 CUDA Toolkit

这条路线最符合当前仓库，也最节省排障时间。

---

## 15. 官方参考链接

以下是本指南使用的主要官方来源：

- Microsoft WSL 安装文档  
  https://learn.microsoft.com/en-us/windows/wsl/install

- Microsoft WSL 开发环境最佳实践  
  https://learn.microsoft.com/en-us/windows/wsl/setup/environment

- Microsoft WSL 文件系统与性能建议  
  https://learn.microsoft.com/en-us/windows/wsl/filesystems

- NVIDIA CUDA on WSL 用户指南  
  https://docs.nvidia.com/cuda/archive/12.4.0/wsl-user-guide/index.html

- NVIDIA CUDA on WSL 入口页  
  https://developer.nvidia.com/cuda/wsl

- NVIDIA CUDA Toolkit 页面  
  https://developer.nvidia.com/cuda/toolkit

- Triton 官方安装文档  
  https://triton-lang.org/main/getting-started/installation.html

- PyTorch 官方安装文档  
  https://docs.pytorch.org/get-started/locally/

---

## 16. 一句话总结

对这份仓库来说，最稳的本地方案不是继续折腾 Windows 原生 Python，而是尽快把 `WSL2 + Ubuntu + 宿主机 NVIDIA 驱动 + 仓库 setup-triton.sh` 这条链路搭起来；先跑通 baseline，再开始做 Triton kernel 实现和优化。
