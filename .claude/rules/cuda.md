---
paths:
  - "src/cuda/**"
  - "**/*.cu"
  - "**/*.cuh"
---

# CUDA Implementation Rules

## Target: Jetson AGX Orin (sm_87, Ampere)

- コンパイル: `nvcc -arch=sm_87`
- Ampere固有機能を活用: Shared Memory hardware atomic, async copy (cp.async)
- Tensor Core不使用（ADR-005参照）

## カーネル実装の鉄則

1. Pythonリファレンス（src/python/reference.py）が正解。実装前に必ず参照
2. テストベクタ（tests/test_vectors/）でFP32 tolerance ~1e-5を検証
3. A行列は必ずA[38][39]でパディング。変更禁止
4. μ更新後は必ず `__syncthreads()`
5. B行列はConstant Memory。`cudaMemcpyToSymbol` で転送
6. `policy_offset` / `policy_count` 引数を削除しない

## プロファイリング（M2以降）

記録すべきメトリクス:
- `sm__throughput.avg.pct_of_peak_sustained_elapsed`
- `dram__throughput.avg.pct_of_peak_sustained_elapsed`
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum`
