# GLM-ASR Triton 调参实验记录

## Step 1：Fusion A/B

### 修改的参数

| 配置 ID | Backend | `MLP.FUSED` | `EncoderMLP.FUSED` | `USE_FLASH_ATTENTION` | `Linear.TILE_M/N/K` | Flash 参数 |
|---|---|---|---|---|---|---|
| A | `cublas` | `False` | `False` | `False` | `64/64/32` | off |
| B | `cublas` | `True` | `True` | `False` | `64/64/32` | off |

### 该参数的影响

| 参数 | 对性能的影响 |
|---|---|
| `Linear.BACKEND` | 固定为 `cublas` 后，`Linear` 层使用库级 GEMM，通常是较强的吞吐基线，这样端到端变化主要来自 fusion，而不是 matmul 后端差异。 |
| `MLP.FUSED` | 打开后会减少 decoder MLP 中间张量读写和 kernel launch 次数，理论上更容易改善 `decoder_prefill` 与 `decode_step`；如果 fused kernel 自身寄存器压力、padding 或额外搬运开销更大，也可能让端到端变慢。 |
| `EncoderMLP.FUSED` | 打开后理论上可减少 encoder 侧 `Linear + GELU` 的访存与 launch 开销，更可能影响 `audio_encoder`；若主路径未大量使用该 fused 路径，则端到端收益通常有限。 |
| `USE_FLASH_ATTENTION` | 固定关闭后，attention 路径保持不变，能把 `A/B` 的性能差异限制在 fusion 本身，避免 FlashAttention 对 `decoder_prefill` 和端到端时间造成干扰。 |

### 实验结果

| 配置 ID | E2E Time (ms) | E2E Speed (ms/token) | Audio Encoder (ms) | Projector (ms) | Decoder Prefill (ms) | Decode Step (ms) | Accuracy | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A | 45640.8 | 3510.83 | 1198.66 | 6.35 | 1826.37 | 516.06 | 100.0 | PASS |
| B | 48060.2 | 3696.94 | 1134.41 | 6.82 | 1828.91 | 494.63 | 100.0 | PASS |

## Step 2：Triton Linear Tile

### 修改的参数

| 配置 ID | Backend | `MLP.FUSED` | `EncoderMLP.FUSED` | `USE_FLASH_ATTENTION` | `Linear.TILE_M/N/K` | Flash 参数 |
|---|---|---|---|---|---|---|
| T1 | `triton` | `True` | `True` | `False` | `64/64/32` | off |
| T2 | `triton` | `True` | `True` | `False` | `128/64/32` | off |
| T3 | `triton` | `True` | `True` | `False` | `64/128/32` | off |

### 该参数的影响

| 参数 | 对性能的影响 |
|---|---|
| `Linear.BACKEND` | 切到 `triton` 后，`Linear` 的性能开始明显依赖 tile 形状；如果 tile 与实际矩阵形状、GPU 资源匹配得好，可能接近或超过 `cublas`，否则容易在端到端上回退。 |
| `Linear.TILE_M` | 增大 `TILE_M` 会让一个 program 一次处理更多输出行，通常能减少 launch 开销、提高输入重用；但如果实际 `M` 偏小，或寄存器/共享资源占用升高，反而会拖慢 `decode_step` 和端到端。 |
| `Linear.TILE_N` | 增大 `TILE_N` 会让一个 program 一次处理更多输出列，通常更有利于权重复用和大输出维吞吐；但 tile 过大时会降低 occupancy，增加片上资源压力。 |
| `Linear.TILE_K` | `TILE_K` 决定每次 reduction 吃多少 `K` 维数据；更大通常能减少循环次数，但会增加寄存器和共享内存压力。当前固定为 `32`，主要是为了把性能变化集中到 `M/N` 两个方向。 |
| `USE_FLASH_ATTENTION` | 固定关闭后，attention 路径的耗时变化不会混入这一步结果，便于把 `T1/T2/T3` 的差异归因到 Triton linear 的 tile 选择。 |

### 实验结果

| 配置 ID | E2E Time (ms) | E2E Speed (ms/token) | Audio Encoder (ms) | Projector (ms) | Decoder Prefill (ms) | Decode Step (ms) | Accuracy | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| T1 | 46321.3 | 3563.18 | 1783.67 | 7.83 | 1582.22 | 498.65 | 100.0 | PASS |
| T2 | 48070.3 | 3697.71 | 1767.59 | 6.47 | 1860.78 | 519.41 | 100.0 | PASS |
| T3 | 57869.7 | 4451.52 | 2094.48 | 6.33 | 1842.62 | 535.17 | 100.0 | PASS |

## Step 2B：Triton Linear Tile（Fused Off）

### 修改的参数

| 配置 ID | Backend | `MLP.FUSED` | `EncoderMLP.FUSED` | `USE_FLASH_ATTENTION` | `Linear.TILE_M/N/K` | Flash 参数 |
|---|---|---|---|---|---|---|
| U1 | `triton` | `False` | `False` | `False` | `64/64/32` | off |
| U2 | `triton` | `False` | `False` | `False` | `128/64/32` | off |
| U3 | `triton` | `False` | `False` | `False` | `64/128/32` | off |

### 该参数的影响

| 参数 | 对性能的影响 |
|---|---|
| `Linear.BACKEND` | 固定为 `triton` 后，端到端速度会直接受到自写 matmul kernel 质量影响；这一组可以单独观察 Triton linear 本身是否有收益，不再混入 fused MLP 的额外代价。 |
| `MLP.FUSED` | 固定关闭后，decoder MLP 回到标准路径，能把实验重点放在线性层 tile 上；如果之前的 fused kernel 本身带来了寄存器压力、padding 或额外搬运开销，那么关闭它后端到端有机会更快。 |
| `EncoderMLP.FUSED` | 固定关闭后，encoder 侧不会再尝试 fused `Linear + GELU` 路径，可避免把 `audio_encoder` 的变化误归因到 linear tile。 |
| `Linear.TILE_M` | 增大 `TILE_M` 更可能改善大 `M` 场景下的吞吐，但如果 decode 阶段 `M` 很小，tile 过大反而会让 occupancy 下降。 |
| `Linear.TILE_N` | 增大 `TILE_N` 更可能改善大输出维场景下的权重复用，但 tile 过大时也更容易带来资源浪费，拖慢端到端。 |
| `Linear.TILE_K` | 保持 `32` 可以把变化集中在 `M/N` 两个方向，避免同时改动 reduction 粒度导致结果难归因。 |
| `USE_FLASH_ATTENTION` | 固定关闭后，attention 路径保持不变，便于把这一组的性能差异归因到 `triton + fused off` 的 linear 配置。 |

### 实验结果

| 配置 ID | E2E Time (ms) | E2E Speed (ms/token) | Audio Encoder (ms) | Projector (ms) | Decoder Prefill (ms) | Decode Step (ms) | Accuracy | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| U1 | 54828.0 | 4217.54 | 1161.10 | 6.47 | 1789.75 | 480.31 | 100.0 | PASS |
| U2 | 213098.4 | 16392.19 | 1236.47 | 6.33 | 2648.50 | 734.09 | 100.0 | PASS |
| U3 | 45012.6 | 3462.51 | 1012.97 | 6.91 | 1767.62 | 503.65 | 100.0 | PASS |

## Step 3：FlashAttention

### 修改的参数

| 配置 ID | Backend | `MLP.FUSED` | `EncoderMLP.FUSED` | `USE_FLASH_ATTENTION` | `Linear.TILE_M/N/K` | Flash 参数 |
|---|---|---|---|---|---|---|
| F1 | `cublas` | `False` | `False` | `True` | `64/64/32` | `32/64, 4w, 2s` |
| F2 | `cublas` | `False` | `False` | `True` | `64/64/32` | `64/64, 4w, 2s` |
| F3 | `cublas` | `False` | `False` | `True` | `64/64/32` | `64/64, 4w, 1s` |

### 该参数的影响

| 参数 | 对性能的影响 |
|---|---|
| `USE_FLASH_ATTENTION` | 打开后，attention 不再显式物化完整 score 矩阵，理论上可减少显存读写并改善长序列 `decoder_prefill`；如果 block 配置不合适，也可能因为资源占用过高而变慢或直接 OOR。 |
| `BLOCK_M` | `BLOCK_M` 越大，一个 kernel 一次处理的 query token 越多，通常有利于提高吞吐、摊薄 launch 开销；但也会抬高寄存器和共享内存占用。 |
| `BLOCK_N` | `BLOCK_N` 越大，一次流式处理的 key/value block 越大，通常更有利于 K/V 复用和带宽效率；但这是最容易把 shared memory 顶满的参数之一。 |
| `num_warps` | 更大的 `num_warps` 往往适合更大的 block，可提升并行度和吞吐；如果 block 本身不够大，warp 过多会增加调度与资源占用，未必带来收益。 |
| `num_stages` | 更大的 `num_stages` 可以更深地预取和流水化，常见收益是隐藏访存延迟；代价是 shared memory 占用上升，因此过大时最容易触发 `OutOfResources`。 |

### 实验结果

| 配置 ID | E2E Time (ms) | E2E Speed (ms/token) | Audio Encoder (ms) | Projector (ms) | Decoder Prefill (ms) | Decode Step (ms) | Accuracy | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| F1 | 27696.7 | 2130.52 | 1402.32 | 6.70 | 2228.55 | 452.08 | 100.0 | PASS |
| F2 | N/A | N/A | 2663.49 | 6.84 | OOR | OOR | N/A | INVALID |
| F3 | 25836.2 | 1987.40 | 2448.10 | 7.76 | 1884.88 | 524.78 | 100.0 | PASS |

## 汇总表

| 配置 ID | Backend | Fused | Flash | Linear Tile | Flash 参数 | E2E Time(ms) | E2E Speed(ms/token) | Audio Encoder(ms) | Projector(ms) | Decoder Prefill(ms) | Decode Step(ms) | Accuracy | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A | `cublas` | off | off | `64/64/32` | off | 45640.8 | 3510.83 | 1198.66 | 6.35 | 1826.37 | 516.06 | 100.0 | PASS |
| B | `cublas` | on | off | `64/64/32` | off | 48060.2 | 3696.94 | 1134.41 | 6.82 | 1828.91 | 494.63 | 100.0 | PASS |
| T1 | `triton` | on | off | `64/64/32` | off | 46321.3 | 3563.18 | 1783.67 | 7.83 | 1582.22 | 498.65 | 100.0 | PASS |
| T2 | `triton` | on | off | `128/64/32` | off | 48070.3 | 3697.71 | 1767.59 | 6.47 | 1860.78 | 519.41 | 100.0 | PASS |
| T3 | `triton` | on | off | `64/128/32` | off | 57869.7 | 4451.52 | 2094.48 | 6.33 | 1842.62 | 535.17 | 100.0 | PASS |
| U1 | `triton` | off | off | `64/64/32` | off | 54828.0 | 4217.54 | 1161.10 | 6.47 | 1789.75 | 480.31 | 100.0 | PASS |
| U2 | `triton` | off | off | `128/64/32` | off | 213098.4 | 16392.19 | 1236.47 | 6.33 | 2648.50 | 734.09 | 100.0 | PASS |
| U3 | `triton` | off | off | `64/128/32` | off | 45012.6 | 3462.51 | 1012.97 | 6.91 | 1767.62 | 503.65 | 100.0 | PASS |
| F1 | `cublas` | off | on | `64/64/32` | `32/64, 4w, 2s` | 27696.7 | 2130.52 | 1402.32 | 6.70 | 2228.55 | 452.08 | 100.0 | PASS |
| F2 | `cublas` | off | on | `64/64/32` | `64/64, 4w, 2s` | N/A | N/A | 2663.49 | 6.84 | OOR | OOR | N/A | INVALID |
| F3 | `cublas` | off | on | `64/64/32` | `64/64, 4w, 1s` | 25836.2 | 1987.40 | 2448.10 | 7.76 | 1884.88 | 524.78 | 100.0 | PASS |
