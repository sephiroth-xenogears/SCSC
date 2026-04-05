/**
 * SCSC Active Inference Fused Kernel — Implementation
 * =====================================================
 * M1-b: Profile mode (PERSISTENT=false)
 *
 * 3-stage pipeline fused into a single kernel:
 *   Prologue → Stage 1 (VFE) → Stage 2 (G(π)) → Stage 3 (Action) → Epilogue
 *
 * Launch: <<<1, 64, 0, stream>>>
 * Thread mapping: tid < 38 active (row-parallel MatVec), tid 38-63 idle/reduction helpers
 *
 * See docs/m1b-kernel-spec.md for full design.
 */

#include "active_inference_kernel.cuh"

// ============================================================
// Kernel implementation
// ============================================================
template<bool PERSISTENT>
__global__ void active_inference_kernel(
    const float* __restrict__ A_flat,
    const float* __restrict__ C_flat,
    const float* __restrict__ D,
    const float* __restrict__ Pi_o,
    const float* __restrict__ Pi_x,
    const float* __restrict__ obs,
    float*       __restrict__ mu_io,
    InferenceOutput* __restrict__ out,
    int policy_offset,
    int policy_count,
    volatile int* flag_new_obs,
    volatile int* flag_result_ready,
    volatile int* flag_shutdown,
    int max_frames)
{
    const int tid = threadIdx.x;

    // ============================================================
    // Shared Memory layout (13,112 bytes = 12.8 KB)
    // ============================================================
    __shared__ float s_A[STATE_DIM][A_PADDED_COLS];     // 5,928 B
    __shared__ float s_C[STATE_DIM][A_PADDED_COLS];     // 5,928 B (ADR 010)
    __shared__ float s_mu[STATE_DIM];                    // 152 B
    __shared__ float s_D[STATE_DIM];                     // 152 B
    __shared__ float s_Pi_o[STATE_DIM];                  // 152 B
    __shared__ float s_Pi_x[STATE_DIM];                  // 152 B
    __shared__ float s_o[STATE_DIM];                     // 152 B
    __shared__ float s_temp[STATE_DIM];                  // 152 B (ADR 013: triple-purpose)
    __shared__ float s_o_pref[STATE_DIM];                // 152 B
    __shared__ float s_A_mu[STATE_DIM];                  // 152 B
    __shared__ float s_G[NUM_POLICIES];                  // 40 B
    __shared__ float s_F;                                // 4 B

    // ============================================================
    // Prologue: DRAM → Shared Memory
    // ============================================================
    // Cooperative loading: all 64 threads participate at stride-64

    // A matrix: 38 × 38 = 1,444 elements → padded [38][39]
    for (int idx = tid; idx < STATE_DIM * STATE_DIM; idx += BLOCK_SIZE) {
        int row = idx / STATE_DIM;
        int col = idx % STATE_DIM;
        s_A[row][col] = A_flat[idx];
    }
    // Padding column: zero
    if (tid < STATE_DIM) {
        s_A[tid][STATE_DIM] = 0.0f;
    }

    // C matrix: same pattern (ADR 010)
    for (int idx = tid; idx < STATE_DIM * STATE_DIM; idx += BLOCK_SIZE) {
        int row = idx / STATE_DIM;
        int col = idx % STATE_DIM;
        s_C[row][col] = C_flat[idx];
    }
    if (tid < STATE_DIM) {
        s_C[tid][STATE_DIM] = 0.0f;
    }

    // Vectors: direct load
    if (tid < STATE_DIM) {
        s_D[tid]    = D[tid];
        s_Pi_o[tid] = Pi_o[tid];
        s_Pi_x[tid] = Pi_x[tid];
        s_o[tid]    = obs[tid];
        s_mu[tid]   = mu_io[tid];
    }

    __syncthreads();  // SYNC PROLOGUE

    // ============================================================
    // Stage 1: VFE Minimization (N=10 iterations)
    // ============================================================
    for (int iter = 0; iter < VFE_ITERS; iter++) {

        float f_local = 0.0f;

        if (tid < STATE_DIM) {
            // MatVec: C @ mu → o_pred (register)
            float acc = 0.0f;
            for (int j = 0; j < STATE_DIM; j++) {
                acc += s_C[tid][j] * s_mu[j];
            }
            float o_pred_i = acc;

            // Prediction errors (registers)
            float eps_o_i = s_o[tid] - o_pred_i;
            float eps_x_i = s_mu[tid] - s_D[tid];

            // Write Pi_o * eps_o to shared for C^T multiply
            s_temp[tid] = s_Pi_o[tid] * eps_o_i;

            // F contribution (register)
            f_local = s_Pi_o[tid] * eps_o_i * eps_o_i
                    + s_Pi_x[tid] * eps_x_i * eps_x_i;
        }

        __syncthreads();  // SYNC #1: s_temp[] visible

        if (tid < STATE_DIM) {
            // MatVec: C^T @ s_temp → grad_sensory (register)
            // Thread tid computes column tid of C^T @ s_temp
            float acc = 0.0f;
            for (int j = 0; j < STATE_DIM; j++) {
                acc += s_C[j][tid] * s_temp[j];
            }
            float grad_s_i = -acc;

            // Read eps_x again (recompute — cheaper than extra shared mem)
            float eps_x_i = s_mu[tid] - s_D[tid];

            // Combined gradient
            float grad_i = grad_s_i + s_Pi_x[tid] * eps_x_i;

            // mu update
            s_mu[tid] -= VFE_LR * grad_i;
        }

        // F reduction via warp shuffle (ADR 012)
        // Warp 0 (tid 0-31)
        unsigned mask0 = 0xFFFFFFFF;
        for (int offset = 16; offset > 0; offset >>= 1) {
            f_local += __shfl_down_sync(mask0, f_local, offset);
        }

        // Warp 1 (tid 32-63)
        unsigned mask1 = 0xFFFFFFFF;
        float f_w1 = f_local;  // each thread in warp 1 has its own value
        for (int offset = 16; offset > 0; offset >>= 1) {
            f_w1 += __shfl_down_sync(mask1, f_w1, offset);
        }

        // Cross-warp exchange via shared memory
        if (tid == 32) {
            s_temp[0] = f_w1;
        }

        __syncthreads();  // SYNC #2: mu visible + s_temp[0] visible

        if (tid == 0) {
            s_F = 0.5f * (f_local + s_temp[0]);
        }
    }

    // ============================================================
    // Stage 2: G(π) Evaluation (K=10 policies)
    // ============================================================

    // --- Precomputation (once) ---
    if (tid < STATE_DIM) {
        // MatVec: A @ mu → s_A_mu
        float acc = 0.0f;
        for (int j = 0; j < STATE_DIM; j++) {
            acc += s_A[tid][j] * s_mu[j];
        }
        s_A_mu[tid] = acc;

        // MatVec: C @ D → s_o_pref
        acc = 0.0f;
        for (int j = 0; j < STATE_DIM; j++) {
            acc += s_C[tid][j] * s_D[j];
        }
        s_o_pref[tid] = acc;
    }
    __syncthreads();  // SYNC: precomputed values visible

    // Ambiguity (scalar, computed by tid 0)
    float ambiguity;
    if (tid == 0) {
        float amb_sum = 0.0f;
        for (int i = 0; i < STATE_DIM; i++) {
            amb_sum += 1.0f / s_Pi_o[i];
        }
        ambiguity = 0.5f * amb_sum;
        s_temp[0] = ambiguity;
    }
    __syncthreads();  // SYNC: ambiguity broadcast
    ambiguity = s_temp[0];

    // --- Per-policy loop ---
    for (int k = 0; k < policy_count; k++) {
        int actual_k = policy_offset + k;
        float risk_local = 0.0f;

        if (tid < STATE_DIM) {
            // MatVec: B[k] @ mu → b_mu (register)
            // B is in Constant Memory: d_B[actual_k][tid][j]
            float acc = 0.0f;
            for (int j = 0; j < STATE_DIM; j++) {
                acc += d_B[actual_k][tid][j] * s_mu[j];
            }
            float b_mu_i = acc;

            // Predicted next state
            float mu_pred_i = s_A_mu[tid] + b_mu_i;

            // Write mu_pred to shared for C @ mu_pred
            s_temp[tid] = mu_pred_i;
        }
        __syncthreads();  // SYNC: mu_pred visible

        if (tid < STATE_DIM) {
            // MatVec: C @ mu_pred → o_pred (register)
            float acc = 0.0f;
            for (int j = 0; j < STATE_DIM; j++) {
                acc += s_C[tid][j] * s_temp[j];
            }
            float o_pred_i = acc;

            // Risk contribution
            float o_diff_i = o_pred_i - s_o_pref[tid];
            risk_local = s_Pi_o[tid] * o_diff_i * o_diff_i;
        }

        // Warp reduction for risk (ADR 012)
        for (int offset = 16; offset > 0; offset >>= 1) {
            risk_local += __shfl_down_sync(0xFFFFFFFF, risk_local, offset);
        }
        // Cross-warp
        if (tid == 32) {
            s_temp[0] = risk_local;
        }
        __syncthreads();  // SYNC: warp1 partial sum visible

        if (tid == 0) {
            float total_risk = risk_local + s_temp[0];
            s_G[k] = ambiguity + 0.5f * total_risk;
        }
    }

    // ============================================================
    // Stage 3: Action Selection (softmax + B[action] @ mu)
    // ============================================================
    if (tid == 0) {
        // Softmax over -G (lower G = higher probability)
        float max_neg_G = -s_G[0];
        for (int k = 1; k < policy_count; k++) {
            float neg_Gk = -s_G[k];
            if (neg_Gk > max_neg_G) max_neg_G = neg_Gk;
        }

        float sum_exp = 0.0f;
        float probs[NUM_POLICIES];
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
        for (int k = 0; k < policy_count; k++) {
            out->G[k] = s_G[k];
        }
    }
    __syncthreads();  // SYNC: out->action visible

    // ============================================================
    // Epilogue: Write results to DRAM
    // ============================================================
    if (tid < STATE_DIM) {
        out->mu[tid] = s_mu[tid];
        mu_io[tid] = s_mu[tid];
    }

    // Persistent mode epilogue (M3)
    if constexpr (PERSISTENT) {
        if (tid == 0) {
            atomicExch((int*)flag_new_obs, 0);
            __threadfence_system();
            atomicExch((int*)flag_result_ready, 1);
        }
        __syncthreads();
        if (*flag_shutdown) return;
    }
}

// ============================================================
// Explicit template instantiations
// ============================================================
template __global__ void active_inference_kernel<false>(
    const float*, const float*, const float*, const float*, const float*,
    const float*, float*, InferenceOutput*,
    int, int,
    volatile int*, volatile int*, volatile int*, int);

template __global__ void active_inference_kernel<true>(
    const float*, const float*, const float*, const float*, const float*,
    const float*, float*, InferenceOutput*,
    int, int,
    volatile int*, volatile int*, volatile int*, int);
