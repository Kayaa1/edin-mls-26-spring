# GLM-ASR Triton 调参教学

## 1. 先说清楚这份作业里的“调参”是什么

这里说的调参，不是训练超参数，不是学习率，也不是模型结构搜索。  
这份作业里要做的是 Triton kernel 的性能参数调优，也就是：

- tile/block shape
- `num_warps`
- `num_stages`
- 是否启用 fused kernel
- 是否启用 FlashAttention 路径

课程计划文件对这一点写得很明确：至少要尝试 `2-3` 组配置，并说明你为什么选当前方案。

---

## 2. 先看当前仓库里哪些东西是可调的

结合当前仓库，最重要的可调位置有这几类。

| 位置 | 当前可调项 | 默认状态 | 说明 |
|---|---|---|---|
| `glm_asr_triton_template/layers.py` 的 `Linear` | `TILE_M/TILE_N/TILE_K` | `64/64/32` | 对 `linear_kernel_tf32` 的 tile 大小有直接影响 |
| `glm_asr_triton_template/layers.py` 的 `MLP` | `FUSED`、`TILE_M/TILE_N/TILE_K` | `True`、`64/64/32` | 控制 `swiglu_fused_kernel` 是否启用以及 tile 选择 |
| `glm_asr_triton_template/layers.py` 的 `EncoderMLP` | `FUSED`、`TILE_M/TILE_N/TILE_K` | `True`、`64/64/32` | 控制 `linear_gelu_kernel` |
| 未来的 FlashAttention kernel | `BLOCK_M/BLOCK_N/BLOCK_D`、`num_warps`、`num_stages` | 需要你新增 | 这是 attention 优化的主战场 |
| `softmax` / `norm` 类 kernel | `BLOCK_SIZE` | 一般按 shape 自动取 | 优先级低于 matmul 和 FlashAttention |

但是有一个很关键的现实：

`hw1-asr/glm_asr_triton_template/__init__.py` 当前把模板包默认配置成：

```python
layers.Linear.BACKEND = "cublas"
layers.MLP.FUSED = False
layers.EncoderMLP.FUSED = False
```

这意味着：

- 你虽然实现了 Triton matmul/fused kernel
- 但模板默认还是基线配置
- 如果你不改这个配置，很多“你写好的优化路径”端到端根本不会被使用

所以调参前一定要先确认：你调的那条路径，真的在跑。

---

## 3. 正确的调参顺序

最常见的失败方式是：功能还没稳定，就开始一顿乱调。  
正确顺序应该是：

1. 先固定正确性
2. 再确定哪些 kernel 真正被调用
3. 先做热点算子的局部对比
4. 再做端到端 benchmark
5. 最后才选最终参数

在这个仓库里，推荐顺序如下：

### 第一步：先让 template 正确跑通

先确认：

```bash
cd hw1-asr/glm_asr_triton_template
python layers.py
python rope.py
python attention.py
```

再确认：

```bash
cd hw1-asr
bash benchmark.sh glm_asr_triton_template
```

如果这一步都不稳定，就不要调参。

### 第二步：确认优化路径真的会被触发

你至少要明确三件事：

1. `Linear.BACKEND` 是不是仍然停在 `cublas`
2. `MLP.FUSED` / `EncoderMLP.FUSED` 是不是还是 `False`
3. attention 现在走的是旧三段式，还是你新加的 Flash 路径

如果不确认这一点，后面所有“调参结果”都可能是假的。

---

## 4. 当前仓库里最值得优先调的地方

不要平均发力。  
对 GLM-ASR 这种模型，真正值得优先调的通常是：

1. `linear_kernel_tf32`
2. `swiglu_fused_kernel`
3. `linear_gelu_kernel`
4. FlashAttention kernel

原因很简单：

- matmul 类通常最占时间
- MLP 是大头
- attention 是课程要求的重点
- elementwise kernel 再怎么调，收益通常也不如大矩阵计算

`softmax_kernel`、`rmsnorm_kernel`、`layernorm_kernel` 当然也可以调，但优先级通常低一些。

---

## 5. 在这个仓库里，调参前要先做的配置整理

如果你准备认真做调参，我建议先把参数显式化，而不是把常量散落在代码里。

例如在 `Linear` 里除了：

```python
TILE_M = 64
TILE_N = 64
TILE_K = 32
```

再加上：

```python
NUM_WARPS = 4
NUM_STAGES = 2
```

launch 时改成：

```python
linear_kernel_tf32[grid](
    ...,
    BLOCK_M=self.TILE_M,
    BLOCK_N=self.TILE_N,
    BLOCK_K=self.TILE_K,
    num_warps=self.NUM_WARPS,
    num_stages=self.NUM_STAGES,
)
```

同样的方法也适用于：

- `MLP`
- `EncoderMLP`
- FlashAttention wrapper

这样你调参时只改类常量，不用到处改 launch 代码。

---

## 6. 一个适合当前仓库的基础调参流程

下面是一套很适合这份作业的最小调参流程。

### 阶段 A：做基线

先记录基线配置的结果。  
这个基线至少要包括：

- `benchmark.sh glm_asr_triton_template`
- `benchmark_detailed.py glm_asr_triton_template --runs 5`

同时记下：

- 当前 `Linear.BACKEND`
- 当前 `MLP.FUSED`
- 当前 `EncoderMLP.FUSED`
- 当前 attention 是 legacy 还是 flash

### 阶段 B：只改一个维度

每轮只改一个东西。  
比如：

- 只改 `TILE_M/TILE_N/TILE_K`
- 或只开 fused
- 或只换 FlashAttention block 配置

不要一轮里同时改：

- tile
- warps
- stages
- fused
- backend

否则你最后不知道究竟是谁带来的变化。

### 阶段 C：固定输入和测试条件

为了让结果可比较，必须固定：

- 同一份音频
- 同一个模型
- 同一组 runs
- 同样的 warmup 次数

不要拿一次 `runs=1` 的偶然值做结论。

### 阶段 D：同时看局部和端到端

局部快，不一定端到端快。  
所以每组参数至少看两类结果：

1. `benchmark_detailed.py` 的组件时间
2. `benchmark.sh` 的整条推理时间

最后以端到端结果为主，组件 profiling 为辅。

---

## 7. 一套建议的尝试顺序

### 7.1 先做 fused 的 A/B

这一步通常最容易出结果，因为仓库里已经有现成 fused kernel。

你可以做两组对比：

#### 配置 A：基线

```python
Linear.BACKEND = "cublas"
MLP.FUSED = False
EncoderMLP.FUSED = False
```

#### 配置 B：只开融合

```python
Linear.BACKEND = "cublas"
MLP.FUSED = True
EncoderMLP.FUSED = True
```

这一步的目的不是追求最优，而是先证明：

- fused 路径能被触发
- fused 路径不会破坏正确性
- fused 路径有可测收益

### 7.2 再做 Triton linear 的 tile 调整

如果你要调 `linear_kernel_tf32`，先确认你真的让：

```python
Linear.BACKEND = "triton"
```

否则你改 `TILE_M/TILE_N/TILE_K` 不会有任何端到端影响。

建议至少试这三组：

1. `(64, 64, 32)`
2. `(128, 64, 32)`
3. `(64, 128, 32)`

如果你的 GPU 比较强，再加：

4. `(128, 128, 32)`

### 7.3 最后调 FlashAttention

FlashAttention 通常建议先固定 `BLOCK_D = head_dim_padded`，主要调：

- `BLOCK_M`
- `BLOCK_N`
- `num_warps`
- `num_stages`

推荐起始点：

1. `BLOCK_M=32, BLOCK_N=64, num_warps=4, num_stages=2`
2. `BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=2`
3. `BLOCK_M=32, BLOCK_N=128, num_warps=8, num_stages=2`

如果 `head_dim=128` 时寄存器压力太大，就优先减小 `BLOCK_M`，不要盲目增大所有 block。

---

## 8. 你应该怎样记录结果

最少要留下一个结果表。  
建议像下面这样记：

| 配置编号 | Linear 后端 | Fused | Flash | Tile/Block | Warps | Stages | benchmark 平均时间 | 组件热点变化 | 正确性 |
|---|---|---|---|---|---|---|---|---|---|
| A | cublas | off | off | baseline | - | - |  |  | PASS/FAIL |
| B | cublas | on | off | baseline | - | - |  |  | PASS/FAIL |
| C | triton | on | off | 64/64/32 | 4 | 2 |  |  | PASS/FAIL |
| D | triton | on | off | 128/64/32 | 4 | 2 |  |  | PASS/FAIL |
| E | triton | on | on | flash 32/64 | 4 | 2 |  |  | PASS/FAIL |

这个表非常重要，因为课程要求不是“你说自己调过”，而是“你能展示自己调过”。

---

## 9. 一个简单可靠的测速原则

无论你是写额外小脚本，还是直接用仓库脚本，测速都要遵守下面几条：

1. 先 warmup
2. 每轮正式测前都同步 GPU
3. 不要只测 1 次
4. 用同一输入重复测
5. 正确性不通过的结果一律作废

如果你自己写小基准，典型结构如下：

```python
import time
import torch

def bench(fn, warmup=10, runs=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times), min(times), max(times)
```

如果是端到端对比，优先用仓库自己的：

```bash
python benchmark_detailed.py glm_asr_triton_template --runs 5
bash benchmark.sh glm_asr_triton_template
```

---

## 10. 为什么局部最优不一定是最终最优

这是调参时非常容易忽略的一点。

举个典型例子：

- 某个 `Linear` tile 配置在单层 matmul 上最快
- 但它寄存器占用太高
- 结果让其他 kernel 的并发性变差
- 端到端不一定最好

所以最终选型时，建议遵守这个优先级：

1. 正确性
2. 端到端时间
3. 关键热点组件时间
4. 参数是否稳定、是否容易解释

不要只因为某个局部 kernel 快了 `3%`，就忽视整个 pipeline 反而更慢的事实。

---

## 11. Triton 里最常见的调参误区

### 11.1 调了参数，但压根没走到这条路径

这是当前仓库最容易发生的错误。  
尤其是：

- `Linear.BACKEND` 还在 `cublas`
- fused flag 还在 `False`
- attention 还在 legacy 路径

这时你改再多 tile 都没意义。

### 11.2 一次改太多参数

如果你同时改：

- block size
- warps
- stages
- backend
- fused 开关

最后你几乎无法解释结果。

### 11.3 只看最快一次

单次最快值经常只是噪声。  
至少看平均值，最好同时看最小/最大值。

### 11.4 只看 microbenchmark，不看端到端

课程最后交付看的仍然是整条模型推理。

### 11.5 忽视数值正确性

某组参数更快，但输出开始漂、NaN、或者 benchmark FAIL，这组参数没有价值。

---

## 12. 对当前仓库最现实的调参建议

如果你的目标是“先把作业交付做完整”，那我建议按下面的优先级推进：

### 优先级 1：先让优化真正启用

先把模板从基线模式切到你要测试的模式，例如：

- `Linear.BACKEND = "triton"` 或保留 `cublas` 做 A/B
- `MLP.FUSED = True`
- `EncoderMLP.FUSED = True`

### 优先级 2：先做 3 组可解释配置

比如只围绕 matmul/FlashAttention 留下三组：

1. 保守配置
2. 中间配置
3. 激进配置

只要你能说明：

- 为什么试这三组
- 结果分别如何
- 为什么最终选某一组

这就已经满足作业要求了。

### 优先级 3：把 profiling 结果补齐

最终你需要的不只是“我调过了”，而是：

- 哪个组件变快
- 变快了多少
- 为什么选这个配置

---

## 13. 一个适合交作业的最小调参方案

如果你想要一条最稳的交付路线，可以这样做：

### 方案一：先完成融合 + FlashAttention + 3 组参数

1. 开 `MLP.FUSED`
2. 开 `EncoderMLP.FUSED`
3. 实现 FlashAttention 风格 attention
4. 对 FlashAttention 试 3 组 block 配置
5. 用 `benchmark_detailed.py` 和 `benchmark.sh` 留结果

这是最贴合作业要求的一条路线。

### 方案二：如果 Triton linear 还不稳定

那就不要强行把所有线性层都切到 Triton。  
可以先：

1. 保持 `Linear.BACKEND = "cublas"`
2. 把 fused MLP 和 FlashAttention 做扎实
3. 把调参重点放在 FlashAttention 上

这条路线也很合理，因为课程要求的重点是：

- 至少一个 fusion
- FlashAttention-style attention
- 至少 2-3 组参数尝试

并没有强制你所有线性层都必须用 Triton matmul。

---

## 14. 最后你应该能交出的东西

如果调参做完，你最好能给出下面这些材料：

1. 一张参数对比表
2. 一组最终配置
3. 一条解释为什么选它
4. 一组 `benchmark.sh` 结果
5. 一组 `benchmark_detailed.py` 结果
6. 一句说明当前有哪些 fallback 或限制

---

## 15. 一句话版调参建议

对当前仓库，调参最重要的不是“试很多数字”，而是：

先确认优化路径真的被启用，再围绕 `fused MLP` 和 `FlashAttention` 做 `2-3` 组有控制变量的配置实验，最后用 `benchmark_detailed.py` 和端到端 benchmark 共同决定最终参数。
