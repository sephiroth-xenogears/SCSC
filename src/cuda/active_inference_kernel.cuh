/**
 * SCSC Active Inference Fused Kernel — Declarations
 * ===================================================
 * Kernel template declarations and launch wrapper for Profile/Persistent modes.
 * See m1b-kernel-spec.md for full design.
 */

#ifndef SCSC_ACTIVE_INFERENCE_KERNEL_CUH
#define SCSC_ACTIVE_INFERENCE_KERNEL_CUH

#include "gen_model.cuh"

// ============================================================
// Kernel template declaration
// ============================================================
// PERSISTENT=false : Profile mode (single launch, Nsight Compute compatible)
// PERSISTENT=true  : Production mode (poll/signal loop, M3)
template<bool PERSISTENT>
__global__ void active_inference_kernel(
    // Model parameters (DRAM → Shared Memory)
    const float* __restrict__ A_flat,     // [38 × 38]
    const float* __restrict__ C_flat,     // [38 × 38]
    const float* __restrict__ D,          // [38]
    const float* __restrict__ Pi_o,       // [38]
    const float* __restrict__ Pi_x,       // [38]

    // Per-frame I/O
    const float* __restrict__ obs,        // [38]
    float*       __restrict__ mu_io,      // [38] in/out
    InferenceOutput* __restrict__ out,    // results

    // Plan B interface (ADR 007)
    int policy_offset,
    int policy_count,

    // Persistent mode only (ignored when PERSISTENT=false)
    volatile int* flag_new_obs,
    volatile int* flag_result_ready,
    volatile int* flag_shutdown,
    int max_frames
);

// ============================================================
// Launch wrapper (Profile mode)
// ============================================================
inline void launch_active_inference(
    const float* A_flat,
    const float* C_flat,
    const float* D,
    const float* Pi_o,
    const float* Pi_x,
    const float* obs,
    float*       mu_io,
    InferenceOutput* out,
    int policy_offset,
    int policy_count,
    cudaStream_t stream = 0)
{
    active_inference_kernel<false><<<1, BLOCK_SIZE, 0, stream>>>(
        A_flat, C_flat, D, Pi_o, Pi_x,
        obs, mu_io, out,
        policy_offset, policy_count,
        nullptr, nullptr, nullptr, 0
    );
}

#endif // SCSC_ACTIVE_INFERENCE_KERNEL_CUH
