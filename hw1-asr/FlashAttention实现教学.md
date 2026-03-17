# GLM-ASR Triton FlashAttention 实现教学

## 1. 这份文档解决什么问题

这份文档专门对应当前仓库里的 `hw1-asr/glm_asr_triton_template/attention.py`，目标不是泛泛介绍 FlashAttention，而是回答下面这几个和作业最相关的问题：

1. 现在这份 `attention.py` 到底卡在哪一步。
2. 在这个仓库里应该怎样把“三段式 attention”改成 FlashAttention 风格。
3. 代码该怎么分层、怎么验证、怎么逐步落地。
4. 哪些边界条件先支持，哪些情况先 fallback。

如果你只想先做一版能交作业、能解释、能 benchmark 的实现，这份文档可以直接当成开发步骤使用。

---

## 2. 先看当前仓库的现状

当前 `hw1-asr/glm_asr_triton_template/attention.py` 里的主路径是传统三段式 attention：

1. `attention_scores_kernel` 先算 `Q @ K^T`
2. `softmax_inplace_kernel` 再对 `scores` 做 softmax
3. `attention_output_kernel` 最后做 `attn_weights @ V`

在 `scaled_dot_product_attention()` 里，这条路径会：

- 先分配完整 `scores` 张量
- 做 mask / causal 处理
- 再单独 softmax
- 再单独和 `V` 相乘

这个实现的优点是好理解、好调试，缺点是会把完整的 `scores` 矩阵物化到显存里，内存读写很多。  
课程计划文件也明确说了：这一步之后还需要继续推进到 FlashAttention 风格，也就是 blockwise + streaming softmax 的实现。

---

## 3. FlashAttention 到底在优化什么

标准 attention 的公式是：

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d)) V
```

普通三段式实现的问题不是公式错，而是中间量太大：

- `QK^T` 的形状是 `(seq_q, seq_k)`
- 如果序列长，这个矩阵会很大
- 你不仅要把它写到显存里，还要再读出来做 softmax
- softmax 后又要再读一遍去乘 `V`

FlashAttention 风格的核心思想是：

- 不再把完整的 `scores` 矩阵写回显存
- 只处理一个 `Q` block 对应的若干个 `K/V` block
- 在扫描这些 block 的过程中，在线维护 softmax 的统计量
- 最后直接得到输出 `O`

也就是说，FlashAttention 优化的核心不是“数学公式变了”，而是“数据流变了”。

---

## 4. 你必须掌握的 online softmax

这一部分是整个实现的核心。

### 4.1 为什么不能直接分块 softmax

如果你把 `scores` 分成很多个 `K` block，单独对每个 block 做 softmax 再拼起来，这是错的。  
因为 softmax 的归一化分母是对整行全部 `seq_k` 求和，不是对每个 block 各算各的。

### 4.2 正确做法：维护三类量

对一个 `Q` block 的每一行，我们维护：

- `m_i`：到目前为止看到的最大 score
- `l_i`：到目前为止的 softmax 分母累计值
- `o_i`：到目前为止的加权输出分子累计值

其中：

```text
m_i = max(scores_seen_so_far)
l_i = sum(exp(scores - m_i))
o_i = sum(exp(scores - m_i) * V)
```

最后输出：

```text
output_i = o_i / l_i
```

### 4.3 块更新公式

假设当前处理一个新的 `K/V` block，得到这个 block 的局部分数 `s_ij`。  
对这一行：

```text
m_ij = max(s_ij)
p_ij = exp(s_ij - m_ij)
l_ij = sum(p_ij)
o_ij = p_ij @ V_j
```

然后把旧统计量和新 block 统计量合并：

```text
m_new = max(m_i, m_ij)
alpha = exp(m_i  - m_new)
beta  = exp(m_ij - m_new)

l_new = alpha * l_i + beta * l_ij
o_new = alpha * o_i + beta * o_ij
```

循环结束后：

```text
output = o_new / l_new
```

这套更新就是 FlashAttention 风格实现最关键的数值稳定技巧。

---

## 5. 结合当前仓库，最稳妥的改造路线

不要直接把现在的 `scaled_dot_product_attention()` 整个推翻。  
最稳的路线是：

1. 保留现有三段式实现作为 reference/fallback。
2. 新增一个 FlashAttention 风格 kernel。
3. 新增一个 Python 包装函数，比如 `flash_scaled_dot_product_attention(...)`。
4. 在统一入口里按条件分发。

推荐的代码结构：

```text
scaled_dot_product_attention(...)
|- if use_flash:
|  `- return flash_scaled_dot_product_attention(...)
`- else:
   `- return scaled_dot_product_attention_legacy(...)
```

这样做的好处：

- 调试时有参考答案
- 某些复杂边界条件可以先 fallback
- benchmark 时可以做 A/B 对比

---

## 6. 当前仓库里哪些条件最适合先支持

这门作业不要求你第一版就覆盖一切。

推荐第一版 FlashAttention 主路径优先支持：

- `q/k/v` 在 CUDA 上
- `head_dim` 为模型常见值，例如 `64` 或 `128`
- `attention_mask is None`
- `is_causal` 为 `False` 或 `True`
- `num_heads == num_kv_heads` 的张量形状

这里最后一条并不意味着你不支持 GQA。  
因为当前 `MultiHeadAttention.__call__()` 已经在 Python 侧通过 `_expand_kv()` 把 `k/v` 扩成和 `q` 同样的 head 数，所以 kernel 本身仍然可以假设输入是：

```text
q: (batch, heads, seq_q, head_dim)
k: (batch, heads, seq_k, head_dim)
v: (batch, heads, seq_k, head_dim)
```

推荐第一阶段先这样处理：

- 如果 `attention_mask is not None`，先 fallback 到旧路径
- 先把无 mask 和 causal 做对
- 等正确性建立后，再考虑把通用 additive mask 也搬进 Flash kernel

这是当前仓库里最现实的推进顺序。

---

## 7. Kernel 设计应该长什么样

一个适合当前作业的 kernel 思路如下：

### 7.1 建议的 grid

- `axis 0`: `batch * num_heads`
- `axis 1`: `seq_q` 方向上的 query block

也就是一块程序负责：

- 一个 `batch_head`
- 一组连续的 query 行

比如：

```text
grid = (batch * num_heads, ceil_div(seq_q, BLOCK_M))
```

### 7.2 建议的编译期参数

- `BLOCK_M`: 一个程序处理多少个 query 位置
- `BLOCK_N`: 一次扫描多少个 key/value 位置
- `BLOCK_D`: head_dim 的 padded 大小
- `CAUSAL`: 是否因果 mask

### 7.3 建议的运行时参数

- `seq_q`
- `seq_k`
- `head_dim`
- `scale`
- 各个 stride

---

## 8. 一个适合当前作业的 kernel 骨架

下面给的是“开发骨架”，不是最终可直接提交的完整版本。  
你实现时可以按这个结构去写。

```python
@triton.jit
def flash_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    seq_q,
    seq_k,
    head_dim,
    scale,
    stride_q0, stride_q1, stride_q2,
    stride_k0, stride_k1, stride_k2,
    stride_v0, stride_v1, stride_v2,
    stride_o0, stride_o1, stride_o2,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_mask = (offs_m[:, None] < seq_q) & (offs_d[None, :] < head_dim)
    q = tl.load(
        q_ptr + pid_bh * stride_q0 + offs_m[:, None] * stride_q1 + offs_d[None, :] * stride_q2,
        mask=q_mask,
        other=0.0,
    )
    q = q.to(tl.float32)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    o_i = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for start_n in range(0, seq_k, BLOCK_N):
        n_idx = start_n + offs_n

        k_mask = (n_idx[:, None] < seq_k) & (offs_d[None, :] < head_dim)
        v_mask = (n_idx[:, None] < seq_k) & (offs_d[None, :] < head_dim)

        k = tl.load(
            k_ptr + pid_bh * stride_k0 + n_idx[:, None] * stride_k1 + offs_d[None, :] * stride_k2,
            mask=k_mask,
            other=0.0,
        ).to(tl.float32)
        v = tl.load(
            v_ptr + pid_bh * stride_v0 + n_idx[:, None] * stride_v1 + offs_d[None, :] * stride_v2,
            mask=v_mask,
            other=0.0,
        ).to(tl.float32)

        qk = tl.dot(q, tl.trans(k)) * scale

        if CAUSAL:
            causal = offs_m[:, None] >= n_idx[None, :]
            qk = tl.where(causal, qk, -float("inf"))

        qk = tl.where(n_idx[None, :] < seq_k, qk, -float("inf"))

        m_ij = tl.max(qk, axis=1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        o_ij = tl.dot(p, v)

        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        beta = tl.exp(m_ij - m_new)

        l_i = alpha * l_i + beta * l_ij
        o_i = alpha[:, None] * o_i + beta[:, None] * o_ij
        m_i = m_new

    out = o_i / l_i[:, None]
    out_mask = (offs_m[:, None] < seq_q) & (offs_d[None, :] < head_dim)
    tl.store(
        o_ptr + pid_bh * stride_o0 + offs_m[:, None] * stride_o1 + offs_d[None, :] * stride_o2,
        out,
        mask=out_mask,
    )
```

这个骨架最重要的不是每一行字面代码，而是下面三个结构：

1. `q` 只加载一次
2. `k/v` 按 block 扫描
3. softmax 统计量在循环里在线更新

---

## 9. 在当前文件里建议怎么落代码

推荐按下面顺序改 `hw1-asr/glm_asr_triton_template/attention.py`。

### 第一步：把旧路径单独提出来

把当前 `scaled_dot_product_attention()` 的三段式实现抽成：

```python
def scaled_dot_product_attention_legacy(...):
    ...
```

这样之后总入口会更清晰。

### 第二步：新增 Flash kernel 和 wrapper

新增：

```python
@triton.jit
def flash_attention_kernel(...):
    ...

def flash_scaled_dot_product_attention(...):
    ...
```

Python wrapper 要负责：

- reshape 成 `(batch * heads, seq, dim)`
- 对 `head_dim` 做 padding
- 分配输出张量
- 设置 grid
- 选择 `BLOCK_M/BLOCK_N`
- 调 kernel
- 裁掉 padding
- reshape 回 `(batch, heads, seq_q, head_dim)`

### 第三步：加统一入口分发

可以加一个类似开关：

```python
USE_FLASH_ATTENTION = True
```

然后在总入口里做：

```python
if use_flash_attention(...):
    return flash_scaled_dot_product_attention(...)
return scaled_dot_product_attention_legacy(...)
```

`use_flash_attention(...)` 可以先写成保守版本：

- 必须 `q.is_cuda`
- 必须 `attention_mask is None`
- 必须 `head_dim_padded <= 128` 或 `<= 256`
- 必须 `seq_k` 不超过你当前 kernel 能稳定覆盖的范围

---

## 10. 为什么我建议先对 `attention_mask` 做 fallback

当前仓库里最难的不是 online softmax 本身，而是“通用 mask + causal + padding + GQA”一次全混在一起。

如果你一开始就把所有 mask 逻辑都塞进 Flash kernel，调试难度会非常高。  
更现实的顺序是：

1. 先支持无 mask
2. 再支持 causal
3. 再支持 padding 情况
4. 最后再支持通用 `attention_mask`

对当前作业来说，这种分阶段策略更容易成功。

如果你确实要把 additive mask 也搬进 kernel，通常做法是：

- 先把 mask reshape 成 `(batch * heads, seq_q, seq_k)`
- 在每个 `K` block 里 load 当前 mask tile
- 在 `qk` 上直接做加法

但第一版完全可以先不做这一步。

---

## 11. 一个适合当前仓库的最小可行实现顺序

建议按下面五步推进：

### 阶段 A：建立 reference

先把当前三段式代码保留下来，并单独命名成 legacy 路径。  
这样你任何时候都可以拿它做数值对照。

### 阶段 B：只支持无 mask、非 causal

这是最容易打通的一版。  
先验证：

- shape 对
- 没有越界
- 数值接近 legacy 路径

### 阶段 C：支持 causal

在 kernel 里通过 query 绝对位置和 key 绝对位置比较来加 causal mask。

判断逻辑一定要用绝对位置，而不是 block 内相对位置。  
也就是说比较的是：

```text
global_q_index >= global_k_index
```

### 阶段 D：把 fallback 接回统一入口

对于你还没覆盖的情况，继续走 legacy：

- 一般 `attention_mask`
- 特殊 dtype
- 超出你当前 block 设计上限的 shape

### 阶段 E：再考虑性能调优

先把结果做对，再调：

- `BLOCK_M`
- `BLOCK_N`
- `num_warps`
- `num_stages`

---

## 12. 你应该怎样验证这条新路径

### 12.1 单元级对照

先写一个小对照，比较 Flash 路径和旧路径输出误差：

```python
with torch.no_grad():
    y_ref = scaled_dot_product_attention_legacy(q, k, v, attention_mask, is_causal, scale)
    y_new = flash_scaled_dot_product_attention(q, k, v, attention_mask, is_causal, scale)

max_diff = (y_ref - y_new).abs().max().item()
mean_diff = (y_ref - y_new).abs().mean().item()
print("max_diff =", max_diff)
print("mean_diff =", mean_diff)
```

建议至少测这些 case：

- 非 causal
- causal
- `head_dim = 64`
- `head_dim = 128`
- `seq_q != seq_k`
- GQA 路径经过 `_expand_kv()` 后的输入

### 12.2 运行文件自测

```bash
cd hw1-asr/glm_asr_triton_template
python attention.py
```

### 12.3 跑端到端 benchmark

```bash
cd hw1-asr
bash benchmark.sh glm_asr_triton_template
```

### 12.4 跑详细 profiling

```bash
cd hw1-asr
python benchmark_detailed.py glm_asr_triton_template --runs 5
```

你最终要的不只是“能跑”，还要能说明：

- 旧路径和新路径都正确
- 新路径减少了 `scores` 物化
- profiling 里 attention 组件更合理

---

## 13. 最常见的坑

### 13.1 `o_i` 和 `l_i` 更新错位

这是最常见的错误。  
你要么：

- 始终把 `o_i` 看成未归一化分子，最后再除 `l_i`

要么：

- 每一步都把归一化因子写清楚

千万不要一半按分子逻辑写，一半按归一化输出逻辑写。

### 13.2 没有用 `float32` 做累计

`q/k/v` 可以是半精度，但：

- `qk`
- `m_i`
- `l_i`
- `o_i`

都应该优先用 `float32` 累计。

### 13.3 causal mask 用了相对下标

错误写法会把 block 边界搞乱。  
一定要比较全局 token 位置。

### 13.4 忘了处理越界

所有 `tl.load` / `tl.store` 都要带 mask。  
尤其是：

- `seq_q` 不是 `BLOCK_M` 整倍数
- `seq_k` 不是 `BLOCK_N` 整倍数
- `head_dim` 做了 padding

### 13.5 一开始就把所有 mask 都塞进去

这会让 debug 面爆炸。  
对当前作业，先把无 mask 和 causal 做好，已经是非常合理的路线。

---

## 14. 一个适合当前作业的参数起点

第一版可以从下面这些组合开始：

### 对 `head_dim = 64`

- `BLOCK_M = 32`
- `BLOCK_N = 64`
- `num_warps = 4`
- `num_stages = 2`

### 对 `head_dim = 128`

- `BLOCK_M = 32`
- `BLOCK_N = 64`
- `num_warps = 4` 或 `8`
- `num_stages = 2`

如果序列偏长，再尝试：

- `BLOCK_N = 128`

不要一开始就把 block 开太大。  
FlashAttention kernel 很容易因为寄存器压力过高导致 occupancy 掉下去。

---

## 15. 最终交付时你最好能回答的 5 个问题

如果你把这部分做完，报告里最好能回答下面 5 个问题：

1. 旧 attention 路径为什么不是 FlashAttention。
2. 你的新 kernel 如何避免完整物化 `scores`。
3. 你如何实现 online softmax。
4. 你目前哪些情况走 Flash 路径，哪些情况走 fallback。
5. 你用什么实验说明它是正确且更合理的。

---

## 16. 一句话版落地建议

对当前仓库，最稳妥的做法不是“重写全部 attention”，而是：

保留旧三段式路径作为 fallback，在 `attention.py` 里新增一个 blockwise + online softmax 的 FlashAttention 风格 kernel，先覆盖无 mask 和 causal 的主场景，跑通数值对照、单文件测试和端到端 benchmark 后，再进入调参阶段。
