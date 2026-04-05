#pragma once
// ============================================================================
// Phase 7: Full-Body 17-Axis Active Inference — CUDA Primitives
// Warp-level shuffle, block reduction, binary search, clamp, sigmoid
// Migrated from Phase 6 — functionally identical
// ============================================================================

#include <cuda_runtime.h>
#include <cfloat>

namespace phase7 {

// ── Warp-level Reduce Sum ──
__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// ── Warp-level Reduce Max ──
__inline__ __device__ float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// ── Block-level Reduce Sum ──
__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    int num_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

// ── Block-level Reduce Max ──
__inline__ __device__ float blockReduceMax(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;

    val = warpReduceMax(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    int num_warps = (blockDim.x + 31) >> 5;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : -FLT_MAX;
    if (wid == 0) val = warpReduceMax(val);
    return val;
}

// ── Binary search in CDF ──
__inline__ __device__ int findBin(const float* cdf, float u, int n) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (cdf[mid] < u) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// ── Clamp ──
__inline__ __device__ float clampf(float val, float lo, float hi) {
    return fminf(fmaxf(val, lo), hi);
}

// ── Sigmoid ──
__inline__ __device__ float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

} // namespace phase7
