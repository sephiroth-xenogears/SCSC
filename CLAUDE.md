# SCSC - Super Cybernetics Control System

Jetson AGX Orin (Ampere, sm_87) + Teensy 4.1 で動作する能動的推論ロボット制御システム。
38次元の状態空間に対するフュージョンCUDAカーネルを開発中。

## Current Phase: 7 (M1-b: CUDA kernel implementation)

@docs/scsc-phase7-8-handoff.md に全体設計あり。必ず参照すること。
@docs/m1b-kernel-spec.md にカーネル実装仕様あり。Stage 1-3の疑似コード・Shared Memoryレイアウト・同期ポイントの詳細はここを参照。
@docs/architecture-decisions.md にADR 001-014の全設計判断あり。判断理由と却下した代替案を確認すること。

## Commands

- `python3 src/python/test_reference.py` — Pythonリファレンスのテスト（10テスト）
- `python3 src/python/test_reference.py --export tests/test_vectors` — CUDA検証用テストベクタ生成
- `python3 src/python/reference.py` — リファレンス実行（参考値出力）
- `mkdir build && cd build && cmake .. -DCMAKE_CUDA_ARCHITECTURES=87 && make -j$(nproc)` — CUDAビルド
- `cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_FLAGS="-G -lineinfo"` — Nsight Compute用デバッグビルド

## Architecture

```
src/python/          # Pythonリファレンス実装（CUDA検証の正解）
  gen_model.py       # 生成モデル定義 → gen_model.cuh と対応
  reference.py       # 3ステージパイプライン（VFE → G(π) → softmax）
  test_reference.py  # テストスイート + テストベクタエクスポート
src/cuda/            # CUDAカーネル実装
  gen_model.cuh      # メモリ階層配置定義（Shared/Constant/Register）
  kernel.cu          # フュージョンカーネル本体（M1-b）
  kernel.cuh         # カーネル宣言 + launch wrapper
  host_runner.cpp    # Host側ランナー（Profile mode）
  host_persistent.cpp # Host側ランナー（Persistent mode、M3）
tests/test_vectors/  # FP32バイナリテストベクタ（metadata.json参照）
docs/                # 設計ドキュメント
  scsc-phase7-8-handoff.md
  m1b-kernel-spec.md
  architecture-decisions.md  # ADR 001-014
```

## Kernel Design (IMPORTANT)

Dual-modeフュージョンカーネル。`if constexpr<PERSISTENT>` でProfile/Production切替（ADR 009）。
M1-bではProfile mode（`PERSISTENT=false`）で実装し、全テストPASS後にNsight Compute計測。

```
template<bool PERSISTENT>
active_inference_kernel(policy_offset, policy_count, ...)
├── Prologue: DRAM → Shared Memory（persistent時は初回のみ）
├── Stage 1: VFE minimization（N=10反復、20 MatVec、20 sync）
├── Stage 2: G(π) evaluation（K=10 policies、22 MatVec、22 sync）
├── Stage 3: Action selection（softmax → action、1 MatVec、1 sync）
└── Epilogue: DRAM write / persistent時はPinned Mem + flag signal
```

## Memory Hierarchy (MUST FOLLOW)

### Shared Memory（13.1 KB / 192 KB = 6.8%）
- **A[38][39]**: 5,928 B — +1列パディング、gcd(39,32)=1 → bank conflict回避（ADR 002）
- **C[38][39]**: 5,928 B — +1列パディング、行/列アクセス両方向conflict-free（ADR 010）
- **mu[38]**: 152 B — 信念状態（RW、persistent時はframe間保持）
- **D[38]**: 152 B — 事前期待（RO）
- **Pi_o[38]**: 152 B — 観測精度（RO）
- **Pi_x[38]**: 152 B — 状態精度（RO）
- **o[38]**: 152 B — 現在の観測（per frame更新）
- **temp[38]**: 152 B — C^T中間バッファ / mu_pred / warp間交換（ADR 013）
- **o_pref[38]**: 152 B — C@D（Stage2で1回計算）
- **A_mu[38]**: 152 B — A@mu（Stage2で1回計算）
- **G[10]**: 40 B — ポリシースコア
- **F**: 4 B — 自由エネルギー

### Constant Memory
- **B[10][38][38]**: 57,760 B = 56.4KB / 64KB (88%)

### Registers（~30 regs/thread、spill riskなし）
- 勾配 grad_i、予測誤差 eps_o_i / eps_x_i、MatVecアキュムレータ acc、F寄与 f_local

## Thread Configuration

- Row-parallel: thread tid が行 tid を担当（tid < 38 のみactive）
- **Block size: 64 threads（2 warps）**（ADR 011）
  - Warp 0 (tid 0-31): 全active
  - Warp 1 (tid 32-37): 6 active、tid 38-63 は `if(tid<38)` でguard
- mu更新後は必ず `__syncthreads()`（2 warps間の可視性保証）
- **F / riskリダクション: `__shfl_down_sync()` + cross-warp shared exchange**（ADR 012）

## Coding Rules

- FP32 only（FP64不使用）
- Tensor Core不使用（38次元では非効率、Phase 8のZMP制御まで待つ）
- `policy_offset` / `policy_count` 引数を維持（Plan B拡張用インターフェース）
- CUDA出力はPythonリファレンスとFP32 tolerance ~1e-5で一致すること
- テストベクタは `tests/test_vectors/` のバイナリを `fread()` で読み込む
- Stage 2最適化: A@mu と C@D は事前計算してs_A_mu / s_o_prefに格納（per-policy再計算禁止）

## Persistent Mode Rules (M3以降)

- Host↔Device通信: Pinned Memory（`cudaHostAllocMapped`）経由（ADR 014）
- フレームループ中にCUDA APIを一切呼ばない（デッドロック防止の構造的保証）
- Persistent kernel専用stream + comm_streamを分離
- ウォッチドッグタイムアウト: ~100ms でフェイルセーフ停止（ゼロ速度指令）
- グレースフルシャットダウン: `flag_shutdown` → mu書き出し → return

## Git

- Host PC: ドキュメント管理
- Jetson: コード開発
- `git push` 前に必ずテスト通過を確認

## DO NOT

- A行列のパディング（A[38][39]）を変更しない
- C行列のパディング（C[38][39]）を変更しない（ADR 010）
- Constant Memoryの B matrices 配置を変更しない
- `s_temp[38]` を別用途のバッファに分割しない（triple-purpose設計、ADR 013）
- Profile modeでのM1-b実装中にPersistent mode固有コードを有効にしない
- warp shuffleリダクションをatomicAddに変更しない（ADR 012）
- ブロックサイズ64を変更しない（ADR 011。変更はM2のNsight Compute結果に基づくADR更新が必要）
