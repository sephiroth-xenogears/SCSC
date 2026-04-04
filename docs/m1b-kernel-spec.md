# M1-b Kernel Implementation Spec

SCSC Active Inference Fused Kernel — Detailed Design for CUDA Implementation

**Date**: 2026-04-04
**Phase**: 7-8+ / M1-b
**Target**: Jetson AGX Orin 64GB (GA10B, CC 8.7)
**Reference**: `reference.py` (all 10 tests PASS)

---

## 1. Kernel signature

```cpp
template<bool PERSISTENT>
__global__ void active_inference_kernel(
    // Model parameters (DRAM → Shared Memory)
    const float* __restrict__ A_flat,     // [38 × 38] → padded to [38][39]
    const float* __restrict__ C_flat,     // [38 × 38] → padded to [38][39]
    const float* __restrict__ D,          // [38]
    const float* __restrict__ Pi_o,       // [38]
    const float* __restrict__ Pi_x,       // [38]

    // Per-frame I/O
    const float* __restrict__ obs,        // [38] observation
    float*       __restrict__ mu_io,      // [38] belief state (in/out)
    InferenceOutput* __restrict__ out,    // results

    // Plan B interface (ADR 007)
    int policy_offset,
    int policy_count,

    // Persistent mode only
    volatile int* flag_new_obs,           // Host → GPU signal
    volatile int* flag_result_ready,      // GPU → Host signal
    volatile int* flag_shutdown,          // graceful shutdown
    int max_frames                        // iteration limit
);
```

Launch configuration: `<<<1, 64, 0, stream>>>`

---

## 2. Shared Memory layout (13,112 bytes = 12.8 KB)

```
Offset   Size      Name           Access    Purpose
──────   ────      ────           ──────    ───────
0x0000   5,928 B   s_A[38][39]    RO        State transition matrix (padded)
0x1728   5,928 B   s_C[38][39]    RO        Observation matrix (padded, ADR 010)
0x2E50     152 B   s_mu[38]       RW        Belief state (persists across frames)
0x2EE8     152 B   s_D[38]        RO        Prior mean
0x2F80     152 B   s_Pi_o[38]     RO        Observation precision
0x3018     152 B   s_Pi_x[38]     RO        State precision
0x30B0     152 B   s_o[38]        RW        Current observation (per frame)
0x3148     152 B   s_temp[38]     RW        Working buffer (C^T multiply, mu_pred)
0x31E0     152 B   s_o_pref[38]   RO*       C @ D (computed once in Stage 2)
0x3278     152 B   s_A_mu[38]     RO*       A @ mu (computed once in Stage 2)
0x3310      40 B   s_G[10]        RW        G(π) scores
0x3338       4 B   s_F            RW        Free energy scalar
──────
Total:  13,112 B   (6.7% of 192 KB SM capacity)

*RO after initial computation
```

All arrays are `float` (4 bytes per element). The `[39]` padding columns store zero and are never read — they exist solely to make gcd(stride, 32) = 1 for bank-conflict-free access.

---

## 3. Constant Memory

```cpp
__constant__ float d_B[10][38][38];   // 57,760 bytes (88% of 64 KB)
```

Loaded once via `cudaMemcpyToSymbol()` before kernel launch. Accessed in Stage 2 with broadcast pattern (all threads in warp read same B[k][tid][j] for varying j).

---

## 4. Register budget per thread (tid < 38)

```
Variable         Regs   Lifetime
────────         ────   ────────
acc (MatVec)       1    within each MatVec loop
eps_o_i            1    Stage 1 iteration
eps_x_i            1    Stage 1 iteration
grad_s_i           1    Stage 1, after C^T multiply
grad_i             1    Stage 1, combined gradient
f_local            1    F reduction
mu_pred_i          1    Stage 2, per policy
o_pred_i           1    Stage 2, per policy
o_diff_i           1    Stage 2, per policy
risk_i             1    Stage 2, per policy
b_mu_i             1    Stage 2, per policy
loop vars          3    iter counter, j counter, k counter
compiler temp     ~15   address calc, predication, spill buffer
────────         ────
Estimated:       ~30    (11.8% of 255 max → no spill risk)
```

---

## 5. Prologue

```
All threads participate in cooperative loading.
Thread tid loads elements at stride-64 offsets.

// A matrix: 38 × 38 = 1,444 elements → padded to 38 × 39 = 1,482
for (int idx = tid; idx < 38 * 38; idx += BLOCK_SIZE) {
    int row = idx / 38;
    int col = idx % 38;
    s_A[row][col] = A_flat[idx];       // data column
}
// Padding column: zero (handled by initialization or explicit write)
if (tid < 38) s_A[tid][38] = 0.0f;

// C matrix: same pattern
for (int idx = tid; idx < 38 * 38; idx += BLOCK_SIZE) {
    int row = idx / 38;
    int col = idx % 38;
    s_C[row][col] = C_flat[idx];
}
if (tid < 38) s_C[tid][38] = 0.0f;

// Vectors: direct load (tid < 38 only)
if (tid < 38) {
    s_D[tid]    = D[tid];
    s_Pi_o[tid] = Pi_o[tid];
    s_Pi_x[tid] = Pi_x[tid];
    s_o[tid]    = obs[tid];
    s_mu[tid]   = mu_io[tid];
}
__syncthreads();   // SYNC PROLOGUE
```

Total DRAM reads: 12,464 bytes (one-time in persistent mode).

---

## 6. Stage 1: VFE minimization

### 6.1 Per-iteration pseudocode

```
for (int iter = 0; iter < VFE_ITERS; iter++) {

    float o_pred_i = 0.0f;
    float eps_o_i, eps_x_i, grad_s_i, grad_i, f_local;

    if (tid < 38) {
        // ---- MatVec: C @ mu → o_pred (register) ----
        // Thread tid computes row tid of C @ mu
        float acc = 0.0f;
        for (int j = 0; j < 38; j++) {
            acc += s_C[tid][j] * s_mu[j];
            // s_C access: stride 39, bank = (tid*39 + j) % 32
            // s_mu access: broadcast (all threads read same s_mu[j])
        }
        o_pred_i = acc;

        // ---- Prediction errors (registers) ----
        eps_o_i = s_o[tid] - o_pred_i;
        eps_x_i = s_mu[tid] - s_D[tid];

        // ---- Write Pi_o * eps_o to shared for C^T multiply ----
        s_temp[tid] = s_Pi_o[tid] * eps_o_i;

        // ---- F contribution (register, for later reduction) ----
        f_local = s_Pi_o[tid] * eps_o_i * eps_o_i
                + s_Pi_x[tid] * eps_x_i * eps_x_i;
    } else {
        f_local = 0.0f;   // idle threads contribute 0 to reduction
    }

    __syncthreads();   // SYNC #1: s_temp[] visible to all threads

    if (tid < 38) {
        // ---- MatVec: C^T @ s_temp → grad_sensory (register) ----
        // Thread tid computes COLUMN tid of C^T @ s_temp
        // = row tid of (C^T @ v) = Σ_j C[j][tid] * s_temp[j]
        float acc = 0.0f;
        for (int j = 0; j < 38; j++) {
            acc += s_C[j][tid] * s_temp[j];
            // s_C column access: address = j*39 + tid
            // bank = (j*39 + tid) % 32, gcd(39,32)=1 → conflict-free
        }
        grad_s_i = -acc;

        // ---- Combined gradient (register) ----
        grad_i = grad_s_i + s_Pi_x[tid] * eps_x_i;

        // ---- mu update (shared memory write) ----
        s_mu[tid] -= VFE_LR * grad_i;
    }

    // ---- F reduction via warp shuffle ----
    // Warp 0 (tid 0-31): reduce 32 values
    unsigned mask0 = 0xFFFFFFFF;
    for (int offset = 16; offset > 0; offset >>= 1)
        f_local += __shfl_down_sync(mask0, f_local, offset);

    // Warp 1 (tid 32-63): reduce (6 active + 26 zeros)
    unsigned mask1 = 0xFFFFFFFF;
    for (int offset = 16; offset > 0; offset >>= 1)
        f_local += __shfl_down_sync(mask1, f_local, offset);

    // Cross-warp: warp 1 leader → shared → warp 0 leader
    if (tid == 32) s_temp[0] = f_local;  // reuse s_temp[0]
    __syncthreads();   // SYNC #2: mu visible + s_temp[0] visible

    if (tid == 0) {
        s_F = 0.5f * (f_local + s_temp[0]);
    }
}
```

### 6.2 Sync count: 2 per iteration × 10 = 20

### 6.3 MatVec count: 2 per iteration × 10 = 20 MatVecs

### 6.4 Correctness check

After Stage 1: `s_mu[]` should match `reference.py` `result.mu` within tolerance ~1e-5. `s_F` should match `result.F`.

---

## 7. Stage 2: G(π) evaluation

### 7.1 Precomputation (once)

```
if (tid < 38) {
    // ---- MatVec: A @ mu → s_A_mu (shared) ----
    float acc = 0.0f;
    for (int j = 0; j < 38; j++) {
        acc += s_A[tid][j] * s_mu[j];
    }
    s_A_mu[tid] = acc;

    // ---- MatVec: C @ D → s_o_pref (shared) ----
    acc = 0.0f;
    for (int j = 0; j < 38; j++) {
        acc += s_C[tid][j] * s_D[j];
    }
    s_o_pref[tid] = acc;
}
__syncthreads();   // SYNC: precomputed values visible

// ---- Ambiguity (scalar, computed by tid 0) ----
float ambiguity = 0.0f;
if (tid == 0) {
    float amb_sum = 0.0f;
    for (int i = 0; i < 38; i++) {
        amb_sum += 1.0f / s_Pi_o[i];
    }
    ambiguity = 0.5f * amb_sum;
    // Store in shared or register; only tid 0 needs it
}
// Broadcast ambiguity to all threads via shared
if (tid == 0) s_temp[0] = ambiguity;
__syncthreads();
ambiguity = s_temp[0];   // all threads read
```

### 7.2 Per-policy loop

```
for (int k = 0; k < policy_count; k++) {
    int actual_k = policy_offset + k;

    float risk_local = 0.0f;

    if (tid < 38) {
        // ---- MatVec: B[k] @ mu → b_mu (register) ----
        // B is in Constant Memory: d_B[actual_k][tid][j]
        float acc = 0.0f;
        for (int j = 0; j < 38; j++) {
            acc += d_B[actual_k][tid][j] * s_mu[j];
        }
        float b_mu_i = acc;

        // ---- Predicted next state (register) ----
        float mu_pred_i = s_A_mu[tid] + b_mu_i;

        // ---- Write mu_pred to shared for C @ mu_pred ----
        s_temp[tid] = mu_pred_i;
    }
    __syncthreads();   // SYNC: mu_pred visible

    if (tid < 38) {
        // ---- MatVec: C @ mu_pred → o_pred (register) ----
        float acc = 0.0f;
        for (int j = 0; j < 38; j++) {
            acc += s_C[tid][j] * s_temp[j];
        }
        float o_pred_i = acc;

        // ---- Risk contribution (register) ----
        float o_diff_i = o_pred_i - s_o_pref[tid];
        risk_local = s_Pi_o[tid] * o_diff_i * o_diff_i;
    }
    // else: risk_local stays 0

    // ---- Warp reduction for risk ----
    for (int offset = 16; offset > 0; offset >>= 1)
        risk_local += __shfl_down_sync(0xFFFFFFFF, risk_local, offset);
    // Cross-warp
    if (tid == 32) s_temp[0] = risk_local;
    __syncthreads();   // SYNC: warp1 partial sum visible

    if (tid == 0) {
        float total_risk = risk_local + s_temp[0];
        s_G[k] = ambiguity + 0.5f * total_risk;
    }
}
```

### 7.3 Sync count: 1 (precompute) + 1 (ambiguity broadcast) + 2 per policy × 10 = 22

### 7.4 MatVec count: 2 (precompute) + 2 per policy × 10 = 22

### 7.5 Correctness check

`s_G[k]` for k=0..9 should match `reference.py` `result.G[k]` within ~1e-5.

---

## 8. Stage 3: Action selection

```
// ---- Softmax (thread 0 only, 10 elements is trivial) ----
if (tid == 0) {
    float max_neg_G = -s_G[0];
    for (int k = 1; k < policy_count; k++) {
        float neg_Gk = -s_G[k];
        if (neg_Gk > max_neg_G) max_neg_G = neg_Gk;
    }
    float sum_exp = 0.0f;
    float probs[10];   // local array, small enough for registers
    for (int k = 0; k < policy_count; k++) {
        probs[k] = expf(-s_G[k] - max_neg_G);
        sum_exp += probs[k];
    }
    int best_k = 0;
    for (int k = 0; k < policy_count; k++) {
        probs[k] /= sum_exp;
        out->probs[k] = probs[k];
        if (probs[k] > probs[best_k]) best_k = k;
    }
    out->action = best_k;
    out->F = s_F;
    // Copy G values
    for (int k = 0; k < policy_count; k++) {
        out->G[k] = s_G[k];
    }
}
__syncthreads();   // SYNC: out->action visible for B@mu below

// ---- Action vector: B[action] @ mu ----
int selected = out->action + policy_offset;
if (tid < 38) {
    float acc = 0.0f;
    for (int j = 0; j < 38; j++) {
        acc += d_B[selected][tid][j] * s_mu[j];
    }
    // Write action vector is optional (for Teensy command)
    // out->mu[tid] = s_mu[tid];   // also copy final beliefs
}
```

### 8.1 Sync count: 1

---

## 9. Epilogue

```
// ---- Write results to DRAM ----
if (tid < 38) {
    out->mu[tid] = s_mu[tid];
    mu_io[tid] = s_mu[tid];     // persist for next call
}

// ---- Persistent mode: signal and loop ----
if constexpr (PERSISTENT) {
    if (tid == 0) {
        atomicExch((int*)flag_new_obs, 0);
        __threadfence_system();     // ensure DRAM writes visible to host
        atomicExch((int*)flag_result_ready, 1);
    }
    __syncthreads();
    // Check shutdown
    if (*flag_shutdown) return;
    // Loop back to observation poll
}
```

---

## 10. Sync budget summary

| Phase | Syncs | MatVecs | Notes |
|-------|-------|---------|-------|
| Prologue | 1 | 0 | DRAM → SM load |
| Stage 1 (×10 iter) | 20 | 20 | C@mu + C^T@temp per iter |
| Stage 2 precompute | 2 | 2 | A@mu + C@D |
| Stage 2 (×10 policy) | 20 | 20 | B@mu + C@mu_pred per policy |
| Stage 3 | 1 | 1 | softmax + B[action]@mu |
| Epilogue | 1 | 0 | DRAM write + signal |
| **Total** | **45** | **43** | per inference step |

---

## 11. Verification plan

### Test 1: Prologue data integrity
Load model, verify s_A, s_C, s_mu match host-side values.

### Test 2: Single VFE iteration
Run 1 iteration, compare mu and F against reference.py iteration 0.

### Test 3: Full VFE (10 iterations)
Compare final mu and F against reference.py Stage 1 output.
Tolerance: |cuda - ref| < 1e-5 per element.

### Test 4: G(π) single policy
Compute G[0] only, compare against reference.py.

### Test 5: G(π) all policies
Compare G[0..9] against reference.py Stage 2 output.

### Test 6: Action selection
Compare softmax probabilities and selected action.

### Test 7: Full pipeline
Run complete active_inference_step, compare all outputs.
Must match reference values:
- F: 1.40636 → 0.27435
- Best policy: 2
- Tolerance: ~1e-5

### Test 8: Bank conflict verification
Use Nsight Compute to verify zero shared memory bank conflicts on A and C matrix access patterns.

### Test 9: Persistent mode basic
Run 3 frames with synthetic observations, verify frame-to-frame mu persistence.

### Test 10: Persistent mode timeout
Verify watchdog timeout triggers graceful shutdown.

---

## 12. ADR updates from this design

| ADR | Decision | Rejected alternative |
|-----|----------|---------------------|
| 009 | Dual-mode kernel (if constexpr) | Persistent-only, Regular-only |
| 010 | C[38][39] padding | C[38][38] (2-way bank conflict) |
| 011 | BLOCK_SIZE = 64 (2 warps) | 32 (two-pass), 128 (excessive) |
| 012 | Warp shuffle for F/risk reduction | atomicAdd to shared (slower) |
| 013 | s_temp[38] for C^T intermediate | Register-only (impossible for transpose) |
| 014 | Pinned Memory for host↔device | cudaMemcpy per frame (deadlock risk) |

---

## 13. Files to create (M1-b deliverables)

```
src/
  kernel.cu          — Kernel implementation (this spec)
  kernel.cuh         — Kernel declarations + launch wrapper
  host_runner.cpp    — Host-side runner (Profile mode)
  host_persistent.cpp — Host-side runner (Persistent mode)
tests/
  test_kernel.cu     — Tests 1-10 from verification plan
  test_reference.py  — Updated with CUDA output comparison
docs/
  architecture-decisions.md  — ADR 009-014 appended
```
