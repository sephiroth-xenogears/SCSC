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
