# HW1-ASR 项目总览与执行计划

> **截止日期：报告 2026-03-30 中午12点 | 互评 2026-04-13 中午12点**  
> **课程：INFR11269 Machine Learning Systems, UoE 2025/26 Spring**

---

## 一句话总结

为 **GLM-ASR** 语音识别模型手写 GPU kernel（使用 **Triton** 轨道），让模型能在 GPU 上将音频转成正确文本，并完成性能优化，最终写 8 页学术报告。

---

## 模型端到端流程

```
音频 WAV
  → Mel Spectrogram（128 频带）
  → Conv 子采样（4x 降采样）                    ← 不需要改
  → Audio Encoder（32层: LayerNorm + GELU + Linear + Attention + RoPE）
  → Projector（池化4帧 + MLP + GELU）
  → Text Decoder（28层: RMSNorm + SiLU/SwiGLU + Linear + Attention + RoPE）
  → 文本输出
```

**正确性目标输出：**
```
Transcription: Concord returned to its place amidst the tents.
Accuracy: 100.0%
Status: PASS
```

---

## 需要实现的 Kernel（按顺序）

| 阶段 | Kernel | 文件 | 说明 |
|---|---|---|---|
| 1 | `silu_kernel`、`gelu_kernel` | `layers.py` | 逐元素激活函数 |
| 2 | `softmax_kernel`、`rmsnorm_kernel`、`layernorm_kernel` | `layers.py` | 归约/归一化 |
| 3 | `linear_kernel_tf32` | `layers.py` | 分块矩阵乘法 |
| 4 | `attention_scores_kernel`、`softmax_inplace_kernel`、`attention_output_kernel` | `attention.py` | 注意力机制 |
| 5 | `compute_freqs_kernel` | `rope.py` | RoPE 位置编码 |

**只能改这3个文件：**
- `glm_asr_triton_template/layers.py`
- `glm_asr_triton_template/attention.py`
- `glm_asr_triton_template/rope.py`

**禁止改动：** `model.py`、`weight_loader.py`、`conv.py`

---

## 三个强制优化要求（缺一不可）

1. **tile/block size 调参**
   - 至少测试 2-3 组配置（`BLOCK_M`、`BLOCK_N`、`BLOCK_K`、`num_warps`、`num_stages`）
   - 记录每组配置的 latency，说明为什么选最终参数

2. **至少 1 个 kernel fusion**
   - 推荐候选：`linear + GELU`（`linear_gelu_kernel`）或 `SwiGLU`（`swiglu_fused_kernel`）
   - 目标：减少中间 tensor 的读写次数和 kernel launch 开销

3. **FlashAttention 风格的 attention**
   - 分块计算（blockwise QK^T）
   - 数值稳定 softmax（streaming max + online normalizer）
   - 再乘 V
   - 不能是朴素三段式（scores → softmax → output）

---

## 快速命令参考

```bash
# 从仓库根目录运行所有命令
cd /Users/zihuanzhang/EDB/edin-mls-26-spring

# 环境激活
source utils/setup-triton.sh

# 验证 baseline（先确认环境没问题）
cd hw1-asr && ./benchmark.sh glm_asr_triton_example

# 单文件单元测试（在 glm_asr_triton_template/ 里）
cd glm_asr_triton_template
python layers.py
python attention.py
python rope.py

# 端到端正确性测试
cd .. && ./benchmark.sh glm_asr_triton_template

# 性能 profiling（对比 template 和 example）
./benchmark_detailed.sh glm_asr_triton_template
./benchmark_detailed.sh glm_asr_triton_example

# 交互式 demo
streamlit run demo.py
```

---

## 执行阶段 Checklist

### Phase 0：环境验收
- [ ] `source utils/setup-triton.sh` 成功
- [ ] `./benchmark.sh glm_asr_triton_example` → Status: PASS

### Phase 1：Element-wise Kernel
- [ ] 实现 `silu_kernel`（`x * sigmoid(x)`）
- [ ] 实现 `gelu_kernel`（`0.5x(1+tanh(√(2/π)(x+0.044715x³)))`）
- [ ] `python layers.py` 无崩溃，shape 正确

### Phase 2：Reduction / Normalization Kernel
- [ ] 实现 `softmax_kernel`（`exp(x-max)/sum(...)`）
- [ ] 实现 `rmsnorm_kernel`（`x/sqrt(mean(x²)+eps)*w`）
- [ ] 实现 `layernorm_kernel`（`(x-mean)/sqrt(var+eps)*w+b`）
- [ ] `python layers.py` 无 NaN/Inf

### Phase 3：Matmul Kernel
- [ ] 实现 `linear_kernel_tf32`（分块 `A @ B`，TF32 累加）
- [ ] `python layers.py` 数值稳定，shape 正确

### Phase 4：Attention Kernel
- [ ] 实现 `attention_scores_kernel`（`Q @ K^T * scale`）
- [ ] 实现 `softmax_inplace_kernel`（原位 softmax）
- [ ] 实现 `attention_output_kernel`（`attn_weights @ V`）
- [ ] `python attention.py` basic/causal/masked/GQA 全部通过

### Phase 5：RoPE Kernel
- [ ] 实现 `compute_freqs_kernel`（生成 cos/sin 频率缓存）
- [ ] `python rope.py` shape/value 正确

### Phase 6：端到端 + 优化
- [ ] `./benchmark.sh glm_asr_triton_template` → Status: PASS（正确性基线）
- [ ] 尝试 tile/block size 多组配置，记录数据
- [ ] 实现 kernel fusion（≥1个），前后数据对比
- [ ] 实现 FlashAttention-style attention，验证正确性不下降
- [ ] 跑 `benchmark_detailed.sh`，收集 per-operator profiling 数据

### Phase 7：报告写作（截止 2026-03-30）
- [ ] Section 1 Introduction（10%）：GLM-ASR动机、端到端流程、系统概述
- [ ] Section 2 Implementation（18%）：5阶段实现总结、设计选择、内存访问模式
- [ ] Section 3 Performance Profiling（14%）：Benchmark 环境、端到端+算子级数据
- [ ] Section 4 Bottleneck Analysis（14%）：算术强度分析、compute-bound vs memory-bound
- [ ] Section 5 Optimization Attempts（18%）：假设→改动→结果（包括失败的）
- [ ] Section 6 Comparison（16%）：与 baseline 的逐算子+端到端对比，根因分析
- [ ] Section 7 Conclusion（10%）：总结、未来方向、经验教训
- [ ] 提交至 OpenReview：[INFR11269](https://openreview.net/group?id=ed.ac.uk/University_of_Edinburgh/2026/Semester_2/INFR11269)
- [ ] 代码打包成 zip 作为 supplementary material

### Phase 8：互评（截止 2026-04-13）
- [ ] 阅读分配到的报告
- [ ] 写 overall summary（120-180词）
- [ ] 至少3个 major strengths（附具体证据）
- [ ] 至少3个 major weaknesses（附改进建议）
- [ ] 覆盖 technical checks：正确性、profiling质量、bottleneck推理、优化方法论、baseline公平性

---

## 报告评分结构

| 章节 | 权重 | 关键评估点 |
|---|---|---|
| 1. Introduction | 10% | ASR 动机准确且具体、流程清晰、kernel列表、核心结论有信息量 |
| 2. Implementation | 18% | 5阶段实现逻辑、tile/block 选择理由、内存访问模式分析 |
| 3. Performance Profiling | 14% | Benchmark 可复现、端到端+算子级 profiling 数据 |
| 4. Bottleneck Analysis | 14% | 算术强度计算、峰值 TFLOPS/带宽、瓶颈诊断有说服力 |
| 5. Optimization Attempts | 18% | 每个优化用"假设→改动→结果"分析，失败的也算分 |
| 6. Comparison | 16% | 公平对比 baseline，逐算子+端到端，根因分析清晰 |
| 7. Conclusion | 10% | 综合发现、下一步方向、经验反思、表达专业 |

> 报告正文最多 **8 页**，参考文献和附录不计页数。双盲提交，不能在正文中透露身份。

---

## 参考文件索引

| 文件 | 用途 |
|---|---|
| [README.md](README.md) | 快速入门、目录结构、任务概述 |
| [GUIDE.md](GUIDE.md) | 详细实现指南、kernel公式、调试技巧 |
| [codex.md](codex.md) | AI协作执行手册（分阶段任务+验收标准） |
| [Triton任务计划与项目框架总说明.md](Triton任务计划与项目框架总说明.md) | Triton轨道详细执行计划 |
| `glm_asr_triton_example/` | **Triton 完整参考实现**（用于学习和对比） |
| `glm_asr_scratch/` | PyTorch CPU 参考（理解模型结构用） |
| `accessingpdf/` | 评分标准 PDF、报告模板 PDF |
