# SCSC - Super Cybernetics Control System

Jetson AGX Orin (Ampere, sm_87) + Teensy 4.1 で動作する能動的推論ロボット制御システム。
38次元の状態空間に対するフュージョンCUDAカーネルを開発中。

## Current Phase: 7 (M1-b: CUDA kernel implementation)

@docs/scsc-phase7-8-handoff.md に詳細設計あり。必ず参照すること。

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
tests/test_vectors/  # FP32バイナリテストベクタ（metadata.json参照）
docs/                # 設計ドキュメント
```

## Kernel Design (IMPORTANT)

フュージョンカーネルは3ステージ構成。DRAM往復を1回に削減する設計。

```
active_inference_step(policy_offset, policy_count, ...)
├── Prologue: DRAM → Shared Memory（1回のみ）
├── Stage 1: VFE minimization（N=10反復、μ=Shared Mem、勾配=Register）
├── Stage 2: G(π) evaluation（K=10 policies、B行列=Constant Mem）
├── Stage 3: Action selection（softmax → action、結果をDRAM書き出し）
└── Epilogue: DRAM write（action, μ, F のみ）
```

## Memory Hierarchy (MUST FOLLOW)

- **Shared Memory**: A[38][39]（+1列パディング、gcd(39,32)=1でbank conflict回避）、μ[38]、σ[38]、G[10]、C,D,o
- **Constant Memory**: B[10][38][38] = 56.4KB/64KB (88%)
- **Registers**: 勾配 ∂F/∂μ[tid]、予測誤差 ε[tid]、作業変数

## Thread Configuration

- Row-parallel: 1 thread = 1 state dimension
- Block size: 64 threads（2 warps）暫定。M2のNsight Computeで最終決定
- μ更新後は必ず `__syncthreads()`
- G(π)リダクション: `atomicAdd(&G_shared[k], g_local)` — Ampereのhw atomic

## Coding Rules

- FP32 only（FP64不使用）
- Tensor Core不使用（38次元では非効率、Phase 8のZMP制御まで待つ）
- `policy_offset` / `policy_count` 引数を維持（Plan B拡張用インターフェース）
- CUDA出力はPythonリファレンスとFP32 tolerance ~1e-5で一致すること
- テストベクタは `tests/test_vectors/` のバイナリを `fread()` で読み込む

## Git

- Host PC: ドキュメント管理
- Jetson: コード開発
- `git push` 前に必ずテスト通過を確認

## DO NOT

- A行列のパディング（A[38][39]）を変更しない
- Constant Memoryの B matrices 配置を変更しない
- Persistent Kernel化はまだ行わない（ハードウェア到着後に判断）
- ブロックサイズ64を最終値として扱わない（M2で確定）
