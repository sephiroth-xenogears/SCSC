/**
 * SCSC Generative Model - CUDA Header
 * =====================================
 * Memory layout and matrix definitions for the fused kernel.
 * Corresponds to Python gen_model.py.
 *
 * Memory Hierarchy (from Phase 7-8 handoff Section 1.2.5):
 *
 *   Registers (per thread):  ~128 B / thread (42% utilization)
 *     - gradient dF/dmu[tid]:  4 B
 *     - prediction error:      4 B
 *     - work variables:       ~120 B
 *
 *   Shared Memory (per SM):  ~6.3 KB / 48 KB (13% utilization)
 *     - A[38][39]:  5,928 B  (+1 col pad, bank conflict free)
 *     - mu[38]:       152 B
 *     - sigma[38]:    152 B
 *     - G[10]:         40 B
 *     - C, D, o:   ~1,500 B
 *
 *   Constant Memory:  56.4 KB / 64 KB (88% utilization)
 *     - B[10][38][38]:  57,760 B
 */

#ifndef SCSC_GEN_MODEL_CUH
#define SCSC_GEN_MODEL_CUH

// ============================================================
// Dimensions
// ============================================================
constexpr int STATE_DIM     = 38;
constexpr int NUM_SERVO     = 17;
constexpr int NUM_CAMERA    = 2;
constexpr int NUM_POLICIES  = 10;   // K=10 (Plan A)
constexpr int VFE_ITERS     = 10;
constexpr float VFE_LR      = 0.01f;

// Memory layout constants
constexpr int A_PADDED_COLS = 39;   // +1 for bank conflict avoidance
                                     // gcd(39, 32) = 1 -> conflict free

// Thread configuration (tentative, final value from M2 Nsight Compute)
constexpr int BLOCK_SIZE    = 64;   // 2 warps

// ============================================================
// Constant Memory: B matrices
// ============================================================
// B[NUM_POLICIES][STATE_DIM][STATE_DIM] = 10 * 38 * 38 * 4 = 57,760 bytes
// Constant Memory capacity: 64 KB = 65,536 bytes
// Utilization: 88%, remaining ~7.6 KB for gamma, priors, etc.
__constant__ float d_B[NUM_POLICIES][STATE_DIM][STATE_DIM];

// ============================================================
// Host-side model structure
// ============================================================
struct GenerativeModelHost {
    float A[STATE_DIM][STATE_DIM];
    float B[NUM_POLICIES][STATE_DIM][STATE_DIM];
    float C[STATE_DIM][STATE_DIM];
    float D[STATE_DIM];
    float Pi_o[STATE_DIM];
    float Pi_x[STATE_DIM];
};

// ============================================================
// Kernel output structure
// ============================================================
struct InferenceOutput {
    float mu[STATE_DIM];
    float F;
    float G[NUM_POLICIES];
    float probs[NUM_POLICIES];
    int   action;
};

// ============================================================
// TODO (M1-b): Kernel declarations
// ============================================================
// __global__ void active_inference_step(
//     const float* __restrict__ A,       // Shared Memory
//     const float* __restrict__ o,       // observation
//     const float* __restrict__ C,
//     const float* __restrict__ D,
//     const float* __restrict__ Pi_o,
//     const float* __restrict__ Pi_x,
//     float* __restrict__ mu,            // in/out
//     InferenceOutput* __restrict__ out,
//     int policy_offset,                  // Plan B support
//     int policy_count                    // Plan B support
// );

#endif // SCSC_GEN_MODEL_CUH
