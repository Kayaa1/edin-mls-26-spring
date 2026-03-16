# Triton任务计划与项目框架总说明

## 1. 文档目的

这份文档用于统一替代仓库根目录下原有的中文总结/说明文件，目标是把下面几件事一次说清楚：

- 这门作业到底要完成什么
- 为什么当前应选择 `Triton` 轨道
- 仓库的整体结构是什么
- `hw1-asr` 目录里每个部分是干什么的
- `benchmark`、`benchmark_detailed`、`demo`、`show_tunnel` 等脚本分别做什么
- 具体应该改哪些文件
- 一个可以直接执行到最终交付的完整工作计划是什么
- 每个阶段如何验证是否完成

这份文档的定位不是“简短总结”，而是“执行总手册”。

---

## 2. 任务结论

当前建议明确采用 `Triton` 轨道，不再同时推进 `cuTile`。

原因很直接：

- 仓库和课程文档本身更推荐 `Triton`
- `Triton` 对 GPU 兼容性更友好
- 仓库里已经提供了完整的 `Triton example` 作为直接参考
- 你的作业要求是完成一条轨道，不是同时完成两条

因此，当前这份作业的主线是：

1. 只做 `hw1-asr/glm_asr_triton_template/`
2. 完成模板中的 Triton kernels
3. 跑通端到端语音识别
4. 达到正确性验收
5. 补齐性能优化要求
6. 整理 benchmark 与 profiling 结果作为最终交付依据

---

## 3. 这份作业最终要达到什么结果

### 3.1 核心目标

你需要基于 GLM-ASR 模型，自己实现若干关键 GPU kernel，使模型能够在 GPU 上完成从音频到文本的推理，并体现出一定的 GPU 优化能力。

### 3.2 最低正确性目标

在 `hw1-asr` 目录运行：

```bash
bash benchmark.sh glm_asr_triton_template
```

最终应该得到类似输出：

```text
Transcription: Concord returned to its place amidst the tents.
Accuracy: 100.0%
Status: PASS
```

`PASS` 是底线，不是全部。

### 3.3 最低优化目标

根据仓库中的 `hw1-asr/README.md` 和 `hw1-asr/GUIDE.md`，你的提交至少需要体现这 3 类优化：

1. `tile/block size` 调参  
   至少尝试 2 到 3 组配置，并说明最终为什么选择当前参数。

2. 至少 1 个 `kernel fusion`  
   目的是真正减少中间读写和 kernel launch 开销，而不是只在文字里说“做了融合”。

3. `FlashAttention` 风格的 attention  
   需要把 attention 从“三段式 `scores -> softmax -> output`”推进到 blockwise / streaming softmax 的实现思路。

---

## 4. 你真正需要修改哪些文件

只改 `Triton` 模板中的 3 个核心文件：

- `hw1-asr/glm_asr_triton_template/attention.py`
- `hw1-asr/glm_asr_triton_template/layers.py`
- `hw1-asr/glm_asr_triton_template/rope.py`

### 4.1 `layers.py` 负责什么

这个文件负责神经网络中最常见、最基础的一类算子，主要包括：

- `rmsnorm_kernel`
- `layernorm_kernel`
- `gelu_kernel`
- `silu_kernel`
- `linear_kernel_tf32`
- `softmax_kernel`

它们覆盖了：

- 归一化
- 激活函数
- 通用 softmax
- 线性层矩阵乘
- MLP 中的核心计算

此外，这个文件里还已经存在一些融合方向的基础设施，例如：

- `linear_gelu_kernel`
- `swiglu_fused_kernel`
- `MLP.FUSED`
- `EncoderMLP.FUSED`

这些是后续完成“至少一个融合优化”的重要抓手。

### 4.2 `attention.py` 负责什么

这个文件负责自注意力机制的核心路径，主要包括：

- `attention_scores_kernel`
- `softmax_inplace_kernel`
- `attention_output_kernel`

当前模板和 example 的基本结构是传统三段式 attention：

1. 先算 `Q @ K^T`
2. 再对分数做 softmax
3. 再与 `V` 相乘得到输出

这条路径适合先做“正确版本”。  
但如果你要满足课程对 `FlashAttention-style attention` 的要求，最终需要在这里继续重构。

### 4.3 `rope.py` 负责什么

这个文件负责 Rotary Position Embedding，也就是 RoPE 位置编码。  
你需要实现：

- `compute_freqs_kernel`

它负责根据位置和逆频率生成：

- `cos`
- `sin`

后续模型会利用这些缓存对 Q/K 做旋转位置编码。

---

## 5. 哪些文件不要动

以下文件属于公共基础设施，不建议修改：

- `hw1-asr/glm_asr_triton_template/model.py`
- `hw1-asr/glm_asr_triton_template/weight_loader.py`
- `hw1-asr/glm_asr_triton_template/conv.py`

原因：

- `model.py` 负责完整模型结构和推理流程
- `weight_loader.py` 负责从 HuggingFace 加载预训练权重
- `conv.py` 负责音频前端卷积/下采样

如果为了赶时间直接改这些文件，往往会让问题边界失控，最后变成“模型基础设施被改坏了”，而不是“kernel 没写对”。

---

## 6. 仓库整体项目框架

根目录主要结构如下：

| 路径 | 作用 |
|---|---|
| `README.md` | 仓库总入口，介绍课程、目录结构、两条轨道、环境搭建方式 |
| `triton-tutorial/` | Triton 教学路径，从环境检查到 attention，适合补 Triton 基础 |
| `cutile-tutorial/` | cuTile 教学路径，和 Triton 教程结构平行 |
| `hw1-asr/` | 本次作业主体目录，所有实际交付都在这里完成 |
| `utils/` | 环境安装与配置脚本 |
| `pylet_example/` | 额外示例目录，不是当前 HW1 主线 |
| `Teaching Cluster.md` | 集群使用说明的 Markdown 版本 |
| `Teaching Cluster.pdf` | 集群使用说明的 PDF 原版 |
| `requirements-blackwell.lock` | 某一 GPU 平台下的依赖快照 |

### 6.1 `triton-tutorial/` 是干什么的

这是课前或补基础用的 Triton 教程目录。其结构是分课次的：

- `0-environment`：环境检查
- `1-vectoradd`：向量加法入门
- `2-execution-model`：执行模型和 grid 概念
- `3-data-model`：数据类型
- `4-transpose`：转置
- `5-secret-notes`：高级说明
- `6-performance-tuning`：性能调优
- `7-attention`：注意力相关示例

如果你在实现 `linear_kernel_tf32`、softmax、attention 时卡住，这个目录是最直接的学习补充。

### 6.2 `cutile-tutorial/` 是干什么的

这是另一条 `cuTile` 教学路径，与 Triton 教程结构对应。  
你既然已经选 `Triton`，这个目录当前不是主线，但保留它有两个意义：

- 帮你理解课程是“双轨教学”
- 帮你确认自己当前不需要投入精力到 `cuTile` 模板

### 6.3 `utils/` 是干什么的

这是环境准备目录。

关键脚本包括：

- `utils/setup-triton.sh`  
  创建/激活 `mls` conda 环境，并安装 `torch`、`numpy`、`triton`、`cupy`、`datasets` 等 Triton 路线所需依赖。

- `utils/setup-cutile.sh`  
  为 cuTile 路线准备环境。

- `utils/setup-cutile-fix.sh`  
  cuTile 路线的增强/修正版环境脚本，文档里也有提到。

对当前任务来说，`setup-triton.sh` 是最重要的。

### 6.4 `pylet_example/` 是干什么的

这是一个额外示例目录，包含：

- `debate.py`
- `start_worker.sh`
- `README.md`

它不属于当前 HW1-ASR 作业主流程。  
除非你明确需要研究这个示例，否则当前可以忽略。

### 6.5 `Teaching Cluster.*` 是干什么的

这两个文件用于集群使用指导：

- `Teaching Cluster.md`
- `Teaching Cluster.pdf`

如果你在教学集群或 Slurm 环境上运行模型、启动 Streamlit、做端口转发，这两个文件有实际参考价值。

---

## 7. `hw1-asr` 目录框架说明

`hw1-asr/` 是这次作业的核心目录。

### 7.1 主要子目录

| 路径 | 作用 |
|---|---|
| `glm_asr_triton_example/` | Triton 参考实现，完整可运行，是你最重要的对照对象 |
| `glm_asr_triton_template/` | Triton 学生模板，你真正要修改和提交的代码 |
| `glm_asr_cutile_example/` | cuTile 参考实现 |
| `glm_asr_cutile_template/` | cuTile 学生模板 |
| `glm_asr_scratch/` | PyTorch 参考版本，便于理解模型结构和数据流 |

### 7.2 关键脚本与文件

| 路径 | 作用 |
|---|---|
| `benchmark.sh` | 外层 shell 包装脚本，负责调用 `benchmark_student.py` |
| `benchmark_student.py` | 端到端正确性与速度 benchmark 主程序 |
| `benchmark_detailed.sh` | 外层 shell 包装脚本，负责调用 `benchmark_detailed.py` |
| `benchmark_detailed.py` | 组件级 profiling 脚本，输出 encoder / projector / decoder 等耗时 |
| `demo.py` | Streamlit 交互式演示页面，可直接上传或测试音频 |
| `show_tunnel.sh` | 集群环境下为 Streamlit/服务端口生成 SSH 隧道命令 |
| `test_audio.wav` | 默认测试音频 |
| `test_audio.txt` | 参考文本 |
| `README.md` | 作业说明 |
| `GUIDE.md` | 更详细的实现指南和调试思路 |

---

## 8. `benchmark` 到底是干什么的

这是必须搞清楚的一部分，因为它不仅是“跑个脚本”，而是你整个开发流程的验收主线。

### 8.1 `benchmark.sh`

它是一个非常薄的 shell 包装器。

它做的事情主要是：

1. 检查你传入的目录名是否存在
2. 进入 `hw1-asr` 目录
3. 调用：

```bash
python benchmark_student.py <folder_name>
```

所以它的定位是：

- 便捷入口
- 统一命令格式
- 避免每次手敲 Python 命令

### 8.2 `benchmark_student.py`

这是最重要的正确性 benchmark 主程序。

它做的事情包括：

1. 加载测试音频
2. 加载你指定目录下的模型实现
3. 调用对应模型的 `generate`
4. 做 warmup
5. 跑多次 benchmark
6. 输出平均耗时、token 数、速度
7. 解码转写文本
8. 把转写文本与参考答案做对比
9. 给出 `PASS / FAIL`

它的作用可以概括为两句话：

- 它验证你的实现是否“真的能完成端到端转写”
- 它给出最基础的推理耗时数据

因此它是你的第一验收脚本。

### 8.3 `benchmark_detailed.sh`

它也是一个 shell 包装器，主要负责：

- 展示帮助信息
- 校验目录
- 调用：

```bash
python benchmark_detailed.py <folder_name>
```

或者在需要时附带 `--nsys`。

### 8.4 `benchmark_detailed.py`

这是性能分析脚本，不再只看最终一句转写对不对，而是把大模型推理拆开看。

它会重点分析：

- `audio_encoder`
- `multi_modal_projector`
- `decoder_prefill`
- `decode_step`
- 前几层 decoder layer 的单层耗时

它还会：

- 输出各组件均值、方差、最小/最大时间
- 给出大致的总耗时拆分
- 在指定 `--nsys` 时调用 `nsys profile`

它的意义是：

- 找热点
- 比较 example 与 template
- 比较优化前与优化后
- 为 tile/block 调参和 fusion 提供证据

### 8.5 `demo.py`

这是一个 Streamlit Web UI。

它不是评分主入口，但有几个非常实用的价值：

- 用真实交互方式测试转写结果
- 对比 `Triton Example`、`Triton Template`、`CuTile`、`Scratch` 等不同实现
- 验证模型加载、缓存和端到端推理是否稳定

如果你的 benchmark 能过但 demo 出现明显异常，通常说明还有边界条件或资源释放问题没有处理好。

### 8.6 `show_tunnel.sh`

这个脚本主要服务于教学集群 / Slurm 场景。

它做的事是：

1. 找到你当前运行中的作业节点
2. 读取你指定的端口
3. 生成一条本地机器应该执行的 SSH 端口转发命令

典型用途：

- 你在集群上运行了 `streamlit run demo.py`
- Streamlit 提示了一个远端端口
- 你在登录节点执行 `show_tunnel.sh <port>`
- 再把生成的 SSH 命令复制到本地电脑执行

这样你就可以在本地浏览器访问远端的 Web UI。

---

## 9. Triton 轨道中的参考关系

你在做 Triton 作业时，应该建立非常明确的参考链路：

| 目录 | 用途 |
|---|---|
| `glm_asr_triton_template/` | 你的提交实现 |
| `glm_asr_triton_example/` | 直接参考答案级别的工作实现 |
| `glm_asr_scratch/` | 理解模型结构、张量流和语义参考 |

推荐阅读顺序：

1. `hw1-asr/README.md`
2. `hw1-asr/GUIDE.md`
3. `hw1-asr/glm_asr_triton_template/*.py`
4. `hw1-asr/glm_asr_triton_example/*.py`
5. 必要时再看 `glm_asr_scratch/`

核心思路不是“盲写”，而是：

- 先知道模板接口长什么样
- 再知道 example 是怎么实现的
- 然后把模板补到正确
- 最后再在此基础上做超出 example 的性能优化

---

## 10. 模型与 kernel 的对应关系

为了避免写 kernel 时只见局部、不见全局，需要知道这些 kernel 在模型里分别服务于哪里。

### 10.1 模型主流程

整体数据流可以概括为：

```text
Audio (wav)
-> 音频处理器 / 特征提取
-> Mel 频谱特征
-> Conv Subsampler
-> Audio Encoder
-> Multi-modal Projector
-> Text Decoder
-> 输出 token
-> 文本转写
```

### 10.2 Kernel 到模块的映射

| Kernel | 主要服务位置 |
|---|---|
| `layernorm_kernel` | Audio Encoder |
| `rmsnorm_kernel` | Text Decoder |
| `gelu_kernel` | Audio Encoder、Projector |
| `silu_kernel` | Text Decoder 的 SwiGLU 路径 |
| `linear_kernel_tf32` | 各类线性层 |
| `softmax_kernel` | 通用 softmax |
| `attention_scores_kernel` | Attention 中 `QK^T` |
| `softmax_inplace_kernel` | Attention 权重归一化 |
| `attention_output_kernel` | `attention weights @ V` |
| `compute_freqs_kernel` | Q/K 的 RoPE cos/sin 缓存 |

---

## 11. 当前任务的完整执行计划

下面给出从零到最终交付的完整执行路线。  
这不是“可选参考”，而是建议你直接照着推进的主计划。

### 阶段 0：准备环境与运行前提

#### 目标

先确认问题不是环境造成的。

#### 要做的事

1. 在 Linux / WSL / 教学集群环境中使用 Bash
2. 从仓库根目录执行 Triton 环境脚本
3. 确认 CUDA、Torch、Triton 都可用

#### 推荐命令

```bash
source utils/setup-triton.sh
python triton-tutorial/0-environment/check.py
```

#### 完成标准

- `triton` 可导入
- `torch.cuda.is_available()` 为真
- 环境检查脚本不报错

#### 风险

- 本仓库脚本主要按 Linux/Bash 设计
- 如果你在纯 Windows PowerShell 下直接跑 shell 脚本，会出现兼容问题

---

### 阶段 1：先验证 Triton baseline

#### 目标

先证明仓库自带的 Triton 参考实现可以正常跑通。

#### 要做的事

执行：

```bash
cd hw1-asr
bash benchmark.sh glm_asr_triton_example
```

#### 为什么必须先做

因为如果 example 都跑不通，那么问题在：

- 环境
- 驱动
- 权重下载
- Python 依赖
- HuggingFace 访问

而不在你的 kernel 实现。

#### 完成标准

- 得到正确转写
- 输出 `Status: PASS`

#### 产出

- 一份 baseline 耗时结果
- 一条已验证的环境基线

---

### 阶段 2：补齐 template 的最小正确实现

#### 目标

让 `glm_asr_triton_template` 从“带 TODO 的模板”变成“功能正确的工作实现”。

#### 建议实现顺序

1. `gelu_kernel`
2. `silu_kernel`
3. `rmsnorm_kernel`
4. `layernorm_kernel`
5. `softmax_kernel`
6. `compute_freqs_kernel`
7. `linear_kernel_tf32`
8. `attention_scores_kernel`
9. `softmax_inplace_kernel`
10. `attention_output_kernel`

#### 为什么按这个顺序

- elementwise 最容易验证
- reduction 次之
- matmul 再次之
- attention 最复杂，应该放到后面

#### 这一阶段的原则

- 先做对，不先做快
- 尽量和 example 的接口、shape、dtype 行为保持一致
- 不要一上来就写 FlashAttention 版本，否则调试面会太大

#### 完成标准

- 所有功能性 `pass` 已消失
- 三个核心模板文件都能独立运行

---

### 阶段 3：做单文件级测试

#### 目标

在进入端到端 benchmark 之前，先保证每个模块基础可运行。

#### 推荐命令

```bash
cd hw1-asr/glm_asr_triton_template
python layers.py
python rope.py
python attention.py
```

#### 应重点关注什么

- 输出 shape 是否合理
- 是否出现 `illegal memory access`
- 是否出现 NaN / inf
- Triton 编译是否成功

#### 如果失败，优先排查

- mask 逻辑
- stride 使用
- `tl.load` / `tl.store` 的越界保护
- reduction 是否数值稳定
- dtype 是否一致

---

### 阶段 4：第一次端到端跑通

#### 目标

让完整模型先跑出正确文本。

#### 推荐命令

```bash
cd hw1-asr
bash benchmark.sh glm_asr_triton_template
```

#### 这一步要接受的现实

第一次成功跑通时，性能可能并不好。  
这完全正常。现在最重要的是先把“正确性主线”建立起来。

#### 完成标准

- benchmark 正常结束
- 文本输出正确
- `Status: PASS`

#### 如果失败，排查顺序

1. `attention.py`
2. `layers.py`
3. `rope.py`
4. 输入 mask / causal 路径
5. dtype / padding / reshape

---

### 阶段 5：完成至少 1 个 kernel fusion

#### 目标

满足作业“至少一个融合优化”的要求，并带来真实收益。

#### 最稳妥的做法

优先利用模板/参考实现中已经具备的融合方向：

- `linear_gelu_kernel`
- `swiglu_fused_kernel`

#### 建议执行方式

1. 先确认标准路径可以工作
2. 再确认 fused 路径可以被触发
3. 对比 fused 与 unfused 的耗时差异
4. 记录结果

#### 为什么这一步优先做现成融合

因为这是仓库已经预留好的性能优化点，风险低、收益明确，而且最容易形成可交付的优化证据。

#### 完成标准

- 至少一个 fused kernel 实际启用
- benchmark / profiling 能体现前后差异

---

### 阶段 6：把 attention 重构为 FlashAttention 风格

#### 目标

满足课程对 attention 优化的硬性要求。

#### 当前现状

当前 example 和模板使用的是传统三段式 attention：

1. 计算 `scores`
2. 单独 softmax
3. 单独乘 `V`

这种实现容易理解，但会物化完整的 `scores` 矩阵，内存流量较大。

#### 这一阶段建议的实现原则

- 保留原始路径作为 fallback
- 新增 FlashAttention 风格主路径
- 在 `scaled_dot_product_attention()` 统一入口里做分支选择

#### FlashAttention 风格实现要点

1. 以 block 为单位处理 K/V
2. 使用 online softmax / streaming softmax
3. 在累计过程中维护数值稳定性
4. 尽量避免完整物化 `scores`

#### 建议的策略

- 先覆盖模型实际最常见的 head 维度和序列范围
- 对复杂边界条件保留 fallback
- 先保证数值正确，再追求极限性能

#### 完成标准

- `attention.py` 中有一条实际可用的 FlashAttention 风格路径
- 与旧路径相比仍保持正确输出
- profiling 中可看到注意力路径的收益或至少结构改进

---

### 阶段 7：做 tile/block 调参与性能选择

#### 目标

满足“至少试 2 到 3 组配置”的优化要求，并找到适合当前 GPU 的参数。

#### 重点调哪些地方

- `linear_kernel_tf32`
- fused MLP kernel
- FlashAttention kernel

#### 推荐尝试的参数维度

- `BLOCK_M`
- `BLOCK_N`
- `BLOCK_K`
- `num_warps`
- `num_stages`

#### 推荐的起始组合

可以从下面几组开始试：

- `(64, 64, 32)`
- `(128, 64, 32)`
- `(64, 128, 32)`

然后再根据资源占用情况尝试：

- `num_warps = 4`
- `num_warps = 8`
- `num_stages = 2`
- `num_stages = 4`

#### 调参方法

1. 固定正确实现
2. 每次只改一组参数
3. 跑同一套 benchmark
4. 记录平均时间
5. 选最终配置

#### 完成标准

- 至少保留 2 到 3 组配置结果
- 明确当前最终选型

---

### 阶段 8：做详细 profiling 与结果整理

#### 目标

不仅知道“最后快不快”，还知道“哪里快、哪里慢、为什么慢”。

#### 推荐命令

```bash
cd hw1-asr
python benchmark_detailed.py glm_asr_triton_example --runs 5
python benchmark_detailed.py glm_asr_triton_template --runs 5
```

如果环境支持 Nsight Systems，再跑：

```bash
bash benchmark_detailed.sh glm_asr_triton_template --nsys
```

#### 应重点观察的指标

- `audio_encoder`
- `projector`
- `decoder_prefill`
- `decode_step`
- decoder 前几层 layer timing
- 总体估计推理时间

#### 这一阶段的产出

- baseline vs template 对比
- 融合前后对比
- 调参结果表
- FlashAttention 改造前后差异

---

### 阶段 9：最终验收与整理提交材料

#### 最终你应该同时具备

1. 一份可运行的 `glm_asr_triton_template`
2. `benchmark.sh glm_asr_triton_template` 的 `PASS`
3. 至少一个融合优化的证据
4. 至少 2 到 3 组 tile/block 调参记录
5. FlashAttention 风格 attention 的实现说明
6. 详细 profiling 结果

#### 如果要写报告，建议包含

- 你实现了哪些 kernel
- 你如何验证正确性
- 你做了哪些优化
- 调参尝试了哪些配置
- 最终配置是什么
- 与 baseline 相比的性能结果
- 还有哪些已知限制或 fallback 条件

---

## 12. 建议的开发顺序与命令清单

下面给出一条从开始到结束的建议命令路径。

### 12.1 环境与 baseline

```bash
source utils/setup-triton.sh
python triton-tutorial/0-environment/check.py
cd hw1-asr
bash benchmark.sh glm_asr_triton_example
```

### 12.2 单文件自测

```bash
cd hw1-asr/glm_asr_triton_template
python layers.py
python rope.py
python attention.py
```

### 12.3 端到端验证

```bash
cd hw1-asr
bash benchmark.sh glm_asr_triton_template
```

### 12.4 详细 profiling

```bash
cd hw1-asr
python benchmark_detailed.py glm_asr_triton_template --runs 5
python benchmark_detailed.py glm_asr_triton_example --runs 5
```

### 12.5 Web 演示

```bash
cd hw1-asr
streamlit run demo.py
```

如果在教学集群上运行，并且 Streamlit 给出端口号：

```bash
bash show_tunnel.sh <port>
```

---

## 13. 推荐的实现优先级

如果你想知道“先盯住什么最划算”，优先级如下：

### 第一优先级：必须完成

- `layers.py` 中所有 TODO kernel
- `attention.py` 中所有 TODO kernel
- `rope.py` 中 `compute_freqs_kernel`
- 端到端 `PASS`

### 第二优先级：必须体现优化

- 启用至少一个 fused kernel
- 完成 FlashAttention 风格 attention
- 做参数调优并记录结果

### 第三优先级：增强交付质量

- 用 `benchmark_detailed.py` 做组件级分析
- 用 `demo.py` 做交互验证
- 用 `nsys` 做更完整的系统级 profile

---

## 14. 常见风险与排查重点

### 14.1 benchmark 跑不通

优先怀疑：

- 环境没有配好
- 权重没有正确下载
- CUDA / Triton 不可用
- baseline 都没跑通

### 14.2 模型能跑但输出错误

优先怀疑：

- attention 数值不对
- softmax 不稳定
- layernorm / rmsnorm 实现错误
- RoPE cos/sin 生成不对

### 14.3 出现 NaN 或 inf

优先检查：

- 是否减去最大值后再 softmax
- 是否给归一化加入 `eps`
- 是否有越界 load/store

### 14.4 性能没有改善

优先检查：

- fused 路径是否真的被启用
- attention 是否仍然走旧路径
- tile/block 是否只是改了参数但没有带来吞吐提升
- benchmark 是否在比较相同输入和相同 runs 条件

---

## 15. 最终完成标准

满足以下条件，才能认为这项任务“完整完成”：

- 只保留 `Triton` 轨道作为主线
- `glm_asr_triton_template` 中核心 kernel 已实现
- `benchmark.sh glm_asr_triton_template` 输出 `PASS`
- 至少一个融合优化已实际生效
- FlashAttention 风格 attention 已落地
- 至少 2 到 3 组 tile/block 配置已测试并记录
- `benchmark_detailed.py` 已完成结果对比
- 未修改 `model.py`、`weight_loader.py`、`conv.py`

---

## 16. 一句话总览

这次作业的正确做法不是“把 TODO 填完就结束”，而是：

先用 `Triton example` 建立正确基线，再把 `Triton template` 补成可运行实现，随后补齐融合优化、FlashAttention 风格 attention 和参数调优，最后用 `benchmark`、`benchmark_detailed` 和 `demo` 完成正确性、性能与交付说明三条主线。
