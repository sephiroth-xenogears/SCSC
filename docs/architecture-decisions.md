# SCSC Architecture Decision Records

設計判断の記録。CLAUDE.md から参照される。
各ADRは「何を決めたか」「なぜそうしたか」「何を却下したか」を記録。

---

## ADR-001: Kernel Fusion（3ステージ統合）

**決定**: VFE計算→G(π)計算→行動選択を1カーネルに統合。

**理由**: Arithmetic Intensity分析の結果、本システムは完全にメモリバウンド
（~0.47 ops/byte、Orin Ridge Point ~26 ops/byteの約1/50）。
DRAM往復を1回に削減することが性能改善の最大レバー。

**却下した代替案**: 3つの独立カーネル。カーネル間でDRAM往復が発生し、
メモリバウンドのシステムでは致命的。

---

## ADR-002: A行列のBank Conflict回避

**決定**: A[38][38] → A[38][39] に1列パディング。

**理由**: stride 38 → gcd(38, 32) = 2 → 2-way bank conflict。
stride 39 → gcd(39, 32) = 1 → conflict-free。
コスト +152 bytes（無視できる）。

**却下した代替案**: パディングなし。2-way bank conflictは
Shared Memoryアクセスのスループットを半減させる。

---

## ADR-003: μベクトルのShared Memory配置

**決定**: μをShared Memoryに配置。

**理由**: Row-parallel MatVec（s[i] = Σ_j A[i][j] · μ[j]）では、
全スレッドがμの全38要素を読む必要がある。
レジスタはスレッドプライベートなので不可。

**制約**: VFEループ内の `__syncthreads()` は毎反復必要
（thread iがμ[i]を更新した後、次反復でthread jがμ[i]を読むため）。

---

## ADR-004: G(π)リダクション方式

**決定**: `atomicAdd(&G_shared[k], g_local)` を採用。

**理由**: Ampere GPUのShared Memory atomicはハードウェアサポートあり。
38次元×K=10ではコンテンション限定的。

**却下した代替案**: Warp shuffle。38 > 32で2ワープにまたがり、
コード複雑化に対しメリットが薄い。
ただしボトルネック判明時にwarp shuffleへ差し替え可能。

---

## ADR-005: Tensor Core不使用（Phase 7）

**決定**: 現時点ではTensor Core活用を見送り。

**理由**: 38次元を64次元にパディングすると、実データ35%、
無駄な演算65%。Tensor Coreのスループットで補えるか不明。

**将来の適用先**: Phase 8のZMP軌道計画・全身協調制御。
行列サイズがTensor Coreに自然にフィットする計算で投入予定。

---

## ADR-006: C行列のブロック対角構造

**決定**: C行列をブロック対角（サーボブロック + カメラブロック）で実装。

**理由**: サーボ系とカメラ系の観測は物理的に独立。
結合がないことを前提にすると、Shared Memory使用量を削減可能。
密行列への変更は `create_dense_C()` で対応可能。

**再検討条件**: 理論学習（第4章「生成モデル」）の進捗次第で見直し。

---

## ADR-007: Plan A/Plan B拡張インターフェース

**決定**: カーネル引数に `policy_offset` と `policy_count` を持たせ、
任意のポリシースライスを評価可能にする。

**理由**: Plan A（K=10固定）からPlan B（階層的探索）への移行時、
カーネル内部コードの変更が不要。呼び出し側の変更のみで対応。

**Plan B仕様**: Tier 1: 10粗い → Tier 2: top3×10=30細かい。
Kernel launch 2回（Tier間でcudaMemcpyToSymbol）。

---

## ADR-008: RoadRunner/Condor Cluster統合アプローチ（Phase 2クラスタ）

**決定**: Phase 2のクラスタ設計は「第3の道」を採用。
RoadRunnerの段階的構築 + Condor Clusterの制約駆動効率を統合。

**方針**:
- 通信最小化設計: A,B行列は各ノードに複製、ポリシー評価のみ分散
- 段階的構築: 2台で検証→通信パターン実測→ネットワーク設計→スケール
- ネットワーク: 1GbEで開始、ボトルネック判明後に検討

**理由**: 能動的推論の計算構造を活用し、ノード間通信を最小化できる。
汎用MPI通信に頼るRoadRunner型は、SCSCのドメインでは過剰。

# Architecture Decision Records — Phase 7-8+ Session 2

Appended to `docs/architecture-decisions.md` (ADR 001-008 from Phase 7-8)

---

## ADR 009: Dual-Mode Kernel (`if constexpr<PERSISTENT>`)

**Status**: Accepted
**Date**: 2026-04-04

**Context**: Persistent kernel achieves minimum latency by eliminating launch overhead, but Nsight Compute cannot profile non-terminating kernels. We need both profilability (M2) and production-grade latency (M3).

**Decision**: Use C++17 `if constexpr` template parameter to compile two modes from identical core computation code. Profile mode (`PERSISTENT=false`) is a regular kernel fully compatible with Nsight Compute. Production mode (`PERSISTENT=true`) adds poll/signal loop.

**Rejected alternatives**:
- Persistent-only: Cannot profile with Nsight Compute → violates RHPC principle 6
- Regular-only: ~5μs launch overhead per frame → suboptimal latency
- Runtime branch (`if (persistent)`): Dead code remains, consumes registers, prevents compiler optimization

**Consequences**: Core computation PTX is bit-identical between modes. M2 profiling results apply directly to M3 production kernel. Deadlock prevention via stream separation and Pinned Memory (ADR 014).

---

## ADR 010: C Matrix Padding C[38][39]

**Status**: Accepted
**Date**: 2026-04-04

**Context**: C matrix is accessed in two patterns within Stage 1 compute_vfe():
1. Row access: `C @ mu` — thread i reads C[i][0..37]
2. Column access: `C^T @ (Pi_o * eps_o)` — thread i reads C[0..37][i]

With C[38][38] (stride 38), gcd(38, 32) = 2 → 2-way bank conflict on both patterns.

**Decision**: Pad C to C[38][39] (stride 39). gcd(39, 32) = 1 → conflict-free on both row and column access. Cost: +152 bytes (0.08% of SM capacity).

**Rejected alternatives**:
- C[38][38] without padding: 2-way bank conflicts on every MatVec involving C (20+ per inference step)
- Store both C and C^T: eliminates stride issue but doubles memory (+5,776 bytes) with no additional benefit since padding solves both directions

**Consequences**: Same padding technique as ADR 002 (A matrix). Unified approach: all [38×N] matrices in Shared Memory use [38][39] layout.

---

## ADR 011: Block Size = 64 (2 Warps)

**Status**: Accepted
**Date**: 2026-04-04

**Context**: STATE_DIM = 38 requires at least 38 threads for single-pass MatVec (1 thread per row).

**Decision**: BLOCK_SIZE = 64 (2 warps). Thread tid < 38 handles row tid. Threads 38-63 contribute zeros during warp shuffle reductions and are idle during MatVec.

**Rejected alternatives**:
- BLOCK_SIZE = 32: Requires two-pass MatVec (rows 0-31, then 32-37). Apparent 100% utilization is misleading — second pass uses only 6/32 threads. Adds indexing complexity.
- BLOCK_SIZE = 128: 70% idle threads, double warp occupancy cost, no benefit for 38-dim problem.

**Consequences**: 2 warps × 1 block = 4.2% warp occupancy on SM. Acceptable for dedicated robotics system. `__syncthreads()` required (not `__syncwarp()`). 45 sync points per inference step.

---

## ADR 012: Warp Shuffle for Scalar Reductions

**Status**: Accepted
**Date**: 2026-04-04

**Context**: F (free energy) and risk (per-policy) require sum reduction of 38 float values. Two approaches: shared memory reduction or warp shuffle.

**Decision**: Use `__shfl_down_sync()` within each warp, with one cross-warp shared memory exchange (1 float). This avoids shared memory bank contention and extra sync points.

**Rejected alternatives**:
- `atomicAdd` to shared memory: serializes writes, slower for small reductions
- Full shared memory tree reduction: requires log2(N) sync points and N/2 shared memory slots
- Thread 0 sequential sum: wastes parallelism (38 sequential additions)

**Consequences**: Warp shuffle is register-to-register — zero shared memory overhead. Cross-warp exchange uses s_temp[0] (1 float, already allocated). Total cost: 5 shuffle steps per warp + 1 shared read.

---

## ADR 013: s_temp[38] Working Buffer for C^T Multiply

**Status**: Accepted
**Date**: 2026-04-04

**Context**: `compute_vfe()` requires C^T @ (Pi_o * eps_o). Each thread computes `Pi_o[tid] * eps_o[tid]` in registers, but the transpose multiply needs ALL threads' values. Registers are private per-thread.

**Decision**: Allocate s_temp[38] (152 bytes) in Shared Memory as intermediate buffer. Threads write Pi_o * eps_o to s_temp, sync, then read s_temp for column-access MatVec. Buffer is reused for mu_pred in Stage 2 and cross-warp reduction exchange.

**Rejected alternatives**:
- Register-only: Structurally impossible — transpose MatVec requires cross-thread data visibility
- Dedicated separate buffers for each use: wastes shared memory; sequential reuse is safe with proper sync

**Consequences**: 152 bytes additional shared memory. Triple-purpose buffer (C^T intermediate, mu_pred staging, warp reduction exchange) with clear ownership boundaries enforced by `__syncthreads()`.

---

## ADR 014: Pinned Memory for Host↔Device Communication

**Status**: Accepted
**Date**: 2026-04-04

**Context**: Persistent kernel creates deadlock risk if CUDA synchronous APIs are called while kernel is running. Need zero-CUDA-API frame loop design.

**Decision**: Use `cudaHostAlloc()` with `cudaHostAllocMapped` for observation input and result output. Host writes observations directly to pinned memory; GPU reads via mapped pointer. Flag-based signaling (volatile int) for synchronization. No `cudaMemcpy`, `cudaDeviceSynchronize`, or any CUDA API calls during frame loop.

**Rejected alternatives**:
- `cudaMemcpyAsync` per frame: requires CUDA API call, creates deadlock surface with persistent kernel
- Unified Memory (`cudaMallocManaged`): page migration overhead unpredictable, unsuitable for real-time
- Regular `cudaMemcpy` (synchronous): guaranteed deadlock with persistent kernel

**Consequences**: Frame loop is CUDA-API-free by design. Deadlock is structurally impossible. `__threadfence_system()` required after GPU writes to ensure host visibility. Watchdog timeout (ADR 009) provides failsafe for communication loss.
