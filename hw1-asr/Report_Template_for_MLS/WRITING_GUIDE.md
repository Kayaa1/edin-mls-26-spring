# 报告写作指南 — HW1-ASR MLS 2026

> **每次动 report.tex 之前先读这个文件。**

---

## 0. 基本规则

- 报告上限 **8 页**（正文），参考文献不计
- 双盲评审，**不写姓名**
- 提交到 OpenReview，代码作为附件一起 zip 上传
- 截止时间：**2026-03-30 12:00 noon**
- **不删除模板说明文字**，只在说明文字下面添加内容

---

## 1. 页数分配计划

| 章节 | 目标页数 | 状态 |
|---|---|---|
| §1 Introduction | 1.5 页 | ✅ 已写完 |
| §2.1 Implementation Summary | 0.4 页 | ⚠️ 待精简为表格 |
| §2.2 Design Choices | 1.1 页 | ⏳ 待写 |
| §3.1 Setup | 0.2 页 | ⏳ 待写 |
| §3.2 End-to-end Evaluation | 0.3 页 | ⏳ 需实验数据 |
| §3.3 Per-operator Breakdown | 0.5 页 | ⏳ 需实验数据 |
| §4.1 Compute vs Memory | 0.5 页 | ⏳ 待写 |
| §4.2 Overall Bottleneck | 0.25 页 | ⏳ 待写 |
| §5.1 Tile Tuning | 0.4 页 | ⏳ 需实验数据 |
| §5.2 Kernel Fusion | 0.3 页 | ⏳ 待写 |
| §5.3 FlashAttention | 0.6 页 | ⏳ 待写 |
| §5.4 Other (可选) | 0.2 页 | ⏳ |
| §6.1 Data Collection | 0.1 页 | ⏳ |
| §6.2 End-to-end Comparison | 0.3 页 | ⏳ 需实验数据 |
| §6.3 Per-operator Comparison | 0.4 页 | ⏳ 需实验数据 |
| §6.4 Root Cause Analysis | 0.2 页 | ⏳ |
| §7 Conclusion | 0.75 页 | ⏳ 待写 |
| **合计** | **≈ 8.0 页** | |

---

## 2. 各章节写法要点

### §1 Introduction（已完成，勿动）

**1.1 Motivation** — 三段：
1. GLM-ASR pipeline（已含架构数字）
2. Why GPU kernel matters（已含 decoder 28层 per-step 论点）
3. Challenges（已含 non-power-of-2 / 异构结构 / GQA+FlashAttention）

**1.2 System Overview** — 已含：
- Triton track 声明
- 10-kernel 表格（核查通过）
- 三个优化描述
- 延迟结果（TODO，等实验）
- 图（占位符，等替换）

⚠️ **TODO**：
- 填入真实延迟数字（跑完实验后）
- 替换 `figure/workflow.png` 为真实 pipeline 图

---

### §2.1 Implementation Summary（待精简）

**当前问题**：写了5段，约0.6页，太占空间。

**目标格式**：改成一张5行表格 + 一句注释，约0.2-0.3页：

```
Phase | Kernels | Computation | Bound
1 Element-wise | silu, gelu | pointwise σ/tanh | Memory
2 Reductions | softmax, rmsnorm, layernorm | row max/sum/norm | Memory
3 Matmul | linear_kernel_tf32 | tiled A@B (TF32) | Compute
4 Attention | scores, softmax_inplace, output | QK^T→softmax→V | Memory→Compute*
5 RoPE | compute_freqs | cos/sin cache | Memory
```
\* Phase 4 naive 是 memory-bound（O(T²)写回）；FlashAttention 后移向 compute-bound。

---

### §2.2 Design Choices（待写，目标1.1页）

**Block/tile sizes 部分：**
- Element-wise BLOCK_SIZE = 1024（layers.py 默认）
- linear_kernel_tf32：TILE_M=64, TILE_N=64, TILE_K=32（默认，§5调参后更新）
- num_warps / num_stages：写实验结论（等跑完 autotune）

**Data layout 部分：**
- 多维张量 flatten 成 `[batch*heads, seq, dim]` 再用 stride 寻址
- GQA head mapping：`kv_head_idx = q_head_idx // (num_q_heads // num_kv_heads)`，即 28Q/4KV → 每个 KV 头服务 7 个 Q 头

---

### §3 Performance Profiling（需实验数据）

**3.1 Setup**（一段话）：
- 硬件：NVIDIA H200，Teaching Cluster (saxa)
- 命令：`./benchmark.sh` (3 runs) / `./benchmark_detailed.sh`
- 对比对象：`glm_asr_triton_example` vs `glm_asr_triton_template`

**3.2 End-to-end**：一张表，`example | template` 延迟 + 准确率

**3.3 Per-operator**：`benchmark_detailed.sh` 输出整理成表格

---

### §4 Bottleneck Analysis（可提前写理论部分）

每个 kernel 的**算术强度**（FLOP/byte）与 H200 ridge point 对比：
- H200 峰值：~3958 TFLOPS (TF32), 带宽 ~3.35 TB/s
- ridge point ≈ 3958×10¹² / (3.35×10¹²) ≈ **1181 FLOP/byte**
- linear_kernel_tf32：算术强度 ∝ TILE_SIZE，大 tile 时超过 ridge point → compute-bound
- norm/softmax/gelu：< 10 FLOP/byte → memory-bound
- attention（naive）：受 O(T²) score 矩阵读写限制 → memory-bound

---

### §5 Optimization（重点章节）

**格式模板**：每个优化用 Hypothesis → Change → Result

**5.1 Tile/Block Size Tuning**
- Hypothesis：更大的 TILE_M/N 提高 Tensor Core 利用率
- Change：遍历 {64,128} × {64,128} × {32,64} + {4,8} warps + {2,3} stages
- Result：填实验表格

**5.2 Kernel Fusion**
- linear_gelu_kernel：Linear+GELU 融合，省一次 HBM 写/读
- swiglu_fused_kernel：两个 Linear projection + SiLU gate 融合
- Result：timing before/after（需实验数据）

**5.3 FlashAttention-style Attention**（重点，0.6页）
- Hypothesis：naive 方案写 O(T²) score 矩阵到 HBM，随序列长度平方增长
- Change：blockwise Q/K/V tiling，online softmax（维护 running max + log-sum-exp），accumulate weighted V in registers
- Result：不再写 O(T²) 中间结果 → 内存节省 + 速度提升

---

### §6 Comparison

- 6.2：准确率必须 100%（否则实现有 bug）
- 6.3：逐算子对比（attention, linear, norm 分开）
- 6.4：Root cause 重点解释 attention 的改进

---

### §7 Conclusion（5个要点）

1. 最有影响的优化（FlashAttention / tile tuning）及原因
2. memory-bound vs compute-bound 的核心教训
3. 注意 attention 为何最难优化
4. Profiling 中意外的发现
5. 下一步想做什么 / 组员分工

---

## 3. 代码事实速查（防止报告写错）

| 事项 | 正确值 | 来源 |
|---|---|---|
| Conv下采样倍数 | **2×**（conv1 stride=1, conv2 stride=2） | model.py L143-146 |
| Projector帧拼接 | **4×**（非Conv下采样！） | model.py Projector |
| Audio Encoder 层数 | 32 | GlmAsrConfig |
| Audio hidden size | 1280 | GlmAsrConfig |
| Audio heads / head_dim | 20 / 64 | GlmAsrConfig |
| Audio intermediate | 5120 | GlmAsrConfig |
| Audio norm | LayerNorm + GELU | model.py |
| Partial RoPE factor | 50% of head_dim | GlmAsrConfig |
| Text Decoder 层数 | 28 | GlmAsrConfig |
| Text hidden size | 3584 | GlmAsrConfig |
| Text Q-heads / KV-heads | 28 / 4 (GQA) | GlmAsrConfig |
| Text head_dim | 128 | GlmAsrConfig |
| Text intermediate | 18944 | GlmAsrConfig |
| Text norm | RMSNorm + SwiGLU | model.py |
| RoPE base (text) | 500000.0 | GlmAsrConfig |
| softmax_kernel 用于哪里 | Text Decoder output (standalone) | layers.py |
| RMSNorm/LayerNorm 何时 fallback PyTorch | hidden_size 非 2 的幂时 | layers.py _is_power_of_two |
| Linear.BACKEND 默认 | "torch"（需手动切 "triton"） | layers.py |
| MLP.FUSED 默认 | True | layers.py |

---

## 4. 实验数据 TODO 清单

跑完实验后填入报告的数字：

- [ ] `./benchmark.sh glm_asr_triton_example` → baseline 延迟
- [ ] `./benchmark.sh glm_asr_triton_template` → 我们的延迟
- [ ] `./benchmark_detailed.sh glm_asr_triton_example` → 逐算子 baseline
- [ ] `./benchmark_detailed.sh glm_asr_triton_template` → 逐算子我们
- [ ] Tile size 扫描实验（TILE_M/N/K × num_warps × num_stages）
- [ ] FlashAttention before/after timing
- [ ] Kernel fusion before/after timing

---

## 5. 关键文件路径

| 文件 | 用途 |
|---|---|
| `report.tex` | 主报告 |
| `mybib.bib` | 参考文献（FlashAttention-2 论文要加） |
| `figure/workflow.png` | 待替换的 pipeline 图占位符 |
| `../glm_asr_triton_template/layers.py` | Phase 1-3 kernel |
| `../glm_asr_triton_template/attention.py` | Phase 4 kernel |
| `../glm_asr_triton_template/rope.py` | Phase 5 kernel |
| `../glm_asr_triton_example/` | Reference 实现（可对比） |
| `../benchmark.sh` | 端到端测试 |
| `../benchmark_detailed.sh` | 逐算子 profiling |

---

## 6. 参考文献需要加的

`mybib.bib` 里需要确认有（§5.3 要 cite FlashAttention-2）：

```bibtex
@article{dao2023flashattention2,
  title={FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning},
  author={Dao, Tri},
  journal={arXiv preprint arXiv:2307.08691},
  year={2023}
}
```
