# HW1-ASR Codex 协作执行手册（Triton）

## Summary
- 目标：给你和 Codex 一份可直接执行的协作手册，覆盖任务清单、分阶段计划、命令模板、测试验收和结果记录格式。
- 位置：`hw1-asr/codex.md`。
- 协作方式：Codex 负责读代码和改代码，你负责运行命令、回传日志、确认结果。

## 1. 角色与边界（Roles And Boundaries）

### 你的职责（Your Role）
- 准备并激活 `environment`。
- 在本地或集群运行命令与测试。
- 失败时贴完整日志（尤其 `traceback`），成功时贴关键 `benchmark` 输出。
- 确认每个阶段是否达到验收标准。

### Codex 的职责（Codex Role）
- 阅读当前代码并给出最小改动方案。
- 按阶段实现代码修改。
- 提供下一步精确命令。
- 根据 `traceback` 和性能结果定位问题并提出优化策略。

### 代码改动约束（Edit Constraints）
- 允许改动范围：`glm_asr_triton_template/layers.py`、`glm_asr_triton_template/attention.py`、`glm_asr_triton_template/rope.py`。
- 禁止改动：`model.py`、`weight_loader.py`、`conv.py`。

## 2. 项目目标与完成标准（Goals And Criteria）

### 正确性目标（Correctness）
- `./benchmark.sh glm_asr_triton_template` 结果为 `Status: PASS`。

### 硬性优化目标（Required Optimization）
- 调参：至少尝试 2-3 组 `tile/block` 配置。
- 融合：至少实现 1 个 `fused kernel`。
- 注意力：实现 `FlashAttention-style attention`（分块计算、数值稳定 `softmax`、再乘 `V`）。

### 最终交付（Deliverables）
- 更新后的模板代码。
- 调参对比记录。
- 最终 `benchmark` 与 `benchmark_detailed` 日志/截图。

## 3. 分阶段任务清单（Phase Checklist）

### Phase 0：环境验收（Environment Validation）
- 命令：
```bash
source utils/setup-triton.sh
cd hw1-asr
./benchmark.sh glm_asr_triton_example
```
- 验收标准：
- `baseline` 必须跑通。
- 如果 `baseline` 失败，先修环境，不进入模板开发。

### Phase 1：基础 Elementwise Kernel
- 在 `layers.py` 实现：
- `silu_kernel`
- `gelu_kernel`
- 本地测试：
```bash
cd glm_asr_triton_template
python layers.py
```
- 验收标准：
- 无崩溃。
- 输出 `shape` 正确。

### Phase 2：Reduction 与 Normalization Kernel
- 在 `layers.py` 实现：
- `softmax_kernel`
- `rmsnorm_kernel`
- `layernorm_kernel`
- 本地测试：
```bash
python layers.py
```
- 验收标准：
- 无 `NaN/inf`。
- `shape` 与 `dtype` 行为正确。

### Phase 3：Matmul Kernel
- 在 `layers.py` 实现：
- `linear_kernel_tf32`
- 本地测试：
```bash
python layers.py
```
- 验收标准：
- 层级测试通过。
- 数值结果稳定且合理。

### Phase 4：Attention Kernel
- 在 `attention.py` 实现：
- `attention_scores_kernel`
- `softmax_inplace_kernel`
- `attention_output_kernel`
- 本地测试：
```bash
python attention.py
```
- 验收标准：
- `basic/causal/masked/GQA attention` 测试通过。

### Phase 5：RoPE Kernel
- 在 `rope.py` 实现：
- `compute_freqs_kernel`
- 本地测试：
```bash
python rope.py
```
- 验收标准：
- RoPE 测试路径通过，`shape/value` 正确。

### Phase 6：端到端与优化（End-To-End And Optimization）
- 命令：
```bash
cd ../
./benchmark.sh glm_asr_triton_template
./benchmark_detailed.sh glm_asr_triton_template
./benchmark_detailed.sh glm_asr_triton_example
```
- 验收标准：
- 端到端正确性通过。
- 完成详细性能剖析（profile）。
- 记录调参与融合决策。

## 4. 标准协作循环（Collaboration Loop）
- 你每轮需要提供：
- 当前 `branch` 和改动文件。
- 实际执行过的命令。
- 完整报错或关键输出（`Status`、`latency`、`tokens`、`accuracy`）。
- Codex 每轮会输出：
- 下一步最小补丁（minimal patch）。
- 需要执行的精确命令。
- 预期输出与失败分支处理。
- 协作节奏：
- 每完成一个 `phase` 做一次 `checkpoint`（正确性 + 性能）。

## 5. 测试与验收模板（Templates）

### 5.1 Unit Test 记录表
| Phase | Command | Result (PASS/FAIL) | Key Output / Error |
|---|---|---|---|
| Layers | `python layers.py` |  |  |
| Attention | `python attention.py` |  |  |
| RoPE | `python rope.py` |  |  |

### 5.2 End-To-End Benchmark 记录表
| Command | Mean Latency (ms) | Tokens | Accuracy | Status | Notes |
|---|---:|---:|---:|---|---|
| `./benchmark.sh glm_asr_triton_template` |  |  |  |  |  |
| `./benchmark_detailed.sh glm_asr_triton_template` |  |  |  |  |  |
| `./benchmark_detailed.sh glm_asr_triton_example` |  |  |  |  |  |

### 5.3 Optimization 调参记录表
| Config Name | `BLOCK_M` | `BLOCK_N` | `BLOCK_K` | `num_warps` | `num_stages` | Result | Decision |
|---|---:|---:|---:|---:|---:|---|---|
| cfg_a |  |  |  |  |  |  |  |
| cfg_b |  |  |  |  |  |  |  |
| cfg_c |  |  |  |  |  |  |  |

## 6. 常见失败与快速处理（Failure Modes）
- `illegal memory access`：
- 先检查越界：`mask`、`offset`、指针计算。
- `shape mismatch`：
- 核对 `flatten/reshape` 和 `stride/index` 映射。
- `NaN/inf`：
- `softmax` 先减 `max`；`norm` 分母保留 `epsilon`。
- `baseline` 也失败：
- 停止 `kernel` 开发，先修 `environment` 或 `slurm` 资源配置。

## Test Plan
- 每个阶段至少执行 1 次局部测试，并写 1 条阶段总结。
- 完成全部 TODO 后，必须执行：
```bash
./benchmark.sh glm_asr_triton_template
./benchmark_detailed.sh glm_asr_triton_template
```
- 与 `baseline` 在同条件下对比，输出最终结论。

## Assumptions
- 路线固定为 Triton。
- 执行环境是学校 Slurm 集群。
- 本文档先定义完整流程和规范，代码实现在后续回合推进。
