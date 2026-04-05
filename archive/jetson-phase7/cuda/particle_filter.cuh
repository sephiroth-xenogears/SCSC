#pragma once
// ============================================================================
// Phase 7: Full-Body 17-Axis Active Inference — Particle Filter (M5)
// Kernels: initRng, initParticles, predict, likelihood, normalize,
//          buildCdf, resample, estimateState, ESS
// Expanded from Phase 6: STATE_DIM 8→38, ACTION_DIM 2→17
// Ref: SCSC-P7-ARCH-001 §5.1, §5.2, §5.3
// ============================================================================

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "primitives.cuh"
#include "gen_model.cuh"
#include "../include/types.h"
#include "../include/body_config.h"

namespace phase7 {

// ── CUDA Launch Config ──
static constexpr int PF_BLOCK_SIZE = 256;

// ============================================================================
// [0] RNG Initialization
// ============================================================================
__global__ void initRngKernel(curandState* states, unsigned long long seed, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) curand_init(seed, tid, 0, &states[tid]);
}

// ============================================================================
// [9] Initialize Particles — 38-dim
// ============================================================================
__global__ void initParticlesKernel(float* particles, curandState* rng, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    curandState local_rng = rng[tid];
    float* s = &particles[tid * STATE_DIM];

    // Target: small spread around center
    s[state::TARGET_X]  = curand_normal(&local_rng) * 0.2f;
    s[state::TARGET_Y]  = curand_normal(&local_rng) * 0.2f;
    s[state::TARGET_VX] = curand_normal(&local_rng) * 0.01f;
    s[state::TARGET_VY] = curand_normal(&local_rng) * 0.01f;

    // All joints: small perturbation around zero
    for (int i = 4; i < STATE_DIM; i += 2) {
        s[i]     = curand_normal(&local_rng) * 0.05f;  // θ
        s[i + 1] = 0.0f;                                 // ω
    }

    rng[tid] = local_rng;
}

// ============================================================================
// [1] Predict Kernel — 38-dim state transition (§5.2)
// ============================================================================
__global__ void predictKernel(
    float* particles,
    const float* action,           // [NUM_JOINTS]
    const body_config_t* cfg,
    const GenModelParams* params,
    curandState* rng_states,
    int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    curandState local_rng = rng_states[tid];
    float* s = &particles[tid * STATE_DIM];

    // ── Target motion ──
    predictTarget(s, params->group[0].dt, local_rng,
                  params->target_pos_noise, params->target_vel_noise);

    // ── Joint groups ──
    for (int g = 0; g < NUM_GROUPS; g++) {
        const auto& dyn = params->group[g];
        predictJointGroup(
            s, action,
            cfg->group_offset[g],
            cfg->group_action_offset[g],
            cfg->group_size[g],
            dyn.k, dyn.d, dyn.sigma_proc, dyn.dt,
            local_rng,
            cfg->joint_min, cfg->joint_max, cfg->joint_vel_max);
    }

    // ── Inter-group coupling ──
    applyCoupling(s);

    rng_states[tid] = local_rng;
}

// ============================================================================
// [2] Likelihood Kernel — Integrated visual + servo (§5.3)
// ============================================================================
__global__ void likelihoodKernel(
    const float* particles,
    const gpu_observation_t* obs,
    const body_config_t* cfg,
    const GenModelParams* params,
    float* log_weights,
    int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    const float* s = &particles[tid * STATE_DIM];
    float log_L = 0.0f;

    // ── Visual likelihood ──
    if (obs->valid_mask & observation_t::VISUAL_VALID) {
        float detected = obs->visual[obs::DETECTED];
        float pred_cx, pred_cy, pred_w, pred_h;
        predictVisual(s, pred_cx, pred_cy, pred_w, pred_h,
                       params->default_face_size, params->face_aspect_ratio);

        if (detected > 0.5f) {
            float obs_cx = obs->visual[obs::BBOX_CX];
            float obs_cy = obs->visual[obs::BBOX_CY];
            float obs_w  = obs->visual[obs::BBOX_WIDTH];
            float obs_h  = obs->visual[obs::BBOX_HEIGHT];

            float inv_pos_var  = 1.0f / (params->sigma_visual * params->sigma_visual);
            float inv_size_var = 1.0f / (0.1f * 0.1f);  // size variance

            float dx = obs_cx - pred_cx;
            float dy = obs_cy - pred_cy;
            float dw = obs_w  - pred_w;
            float dh = obs_h  - pred_h;

            log_L += params->w_vis * (-0.5f * (dx*dx*inv_pos_var + dy*dy*inv_pos_var
                                              + dw*dw*inv_size_var + dh*dh*inv_size_var));
        } else {
            // Not detected: favor particles predicting face outside FOV
            float out_x = fmaxf(fabsf(pred_cx) - 1.0f, 0.0f);
            float out_y = fmaxf(fabsf(pred_cy) - 1.0f, 0.0f);
            float out_dist = fmaxf(out_x, out_y);
            float p_visible = 1.0f - sigmoidf(params->visibility_steepness * out_dist);
            log_L += params->w_vis * logf(1.0f - p_visible + 1e-6f);
        }
    }

    // ── Joint angle likelihood ──
    float inv_joint_var = 1.0f / (params->sigma_joint * params->sigma_joint);
    for (int j = 0; j < NUM_JOINTS; j++) {
        uint32_t servo_bit = observation_t::SERVO_VALID_BASE << j;
        if (obs->valid_mask & servo_bit) {
            int si = cfg->joint_to_state[j];
            float pred_theta = s[si];
            float obs_theta  = obs->joint_pos[j];
            float dq = obs_theta - pred_theta;
            log_L += params->w_joint * (-0.5f * dq * dq * inv_joint_var);
        }
    }

    // ── Joint current likelihood ──
    float inv_cur_var = 1.0f / (params->sigma_current * params->sigma_current);
    for (int j = 0; j < NUM_JOINTS; j++) {
        uint32_t servo_bit = observation_t::SERVO_VALID_BASE << j;
        if (obs->valid_mask & servo_bit) {
            // Simplified: predicted current ≈ 0 (static), deviation = external load
            float obs_cur = obs->joint_cur[j];
            log_L += params->w_cur * (-0.5f * obs_cur * obs_cur * inv_cur_var);
        }
    }

    log_weights[tid] = log_L;
}

// ============================================================================
// [3a] Find Max Weight
// ============================================================================
__global__ void findMaxWeightKernel(const float* log_weights, float* block_maxes, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < n) ? log_weights[tid] : -FLT_MAX;
    val = blockReduceMax(val);
    if (threadIdx.x == 0) block_maxes[blockIdx.x] = val;
}

// ============================================================================
// [3b] Normalize Weights
// ============================================================================
__global__ void normalizeWeightsKernel(const float* log_weights, float* weights,
                                        float* partial_sums, float max_log_w, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float w = (tid < n) ? expf(log_weights[tid] - max_log_w) : 0.0f;
    float block_sum = blockReduceSum(w);
    if (threadIdx.x == 0) partial_sums[blockIdx.x] = block_sum;
    if (tid < n) weights[tid] = w;
}

// ============================================================================
// [3c] Final Normalization
// ============================================================================
__global__ void finalNormalizeKernel(float* weights, const float* partial_sums,
                                      int num_blocks, int n)
{
    float total = 0.0f;
    for (int i = 0; i < num_blocks; i++) total += partial_sums[i];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n && total > 0.0f) weights[tid] /= total;
}

// ============================================================================
// [4a] Build CDF
// ============================================================================
__global__ void buildCdfKernel(const float* weights, float* cdf, int n)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) { sum += weights[i]; cdf[i] = sum; }
        if (sum > 0.0f) {
            float inv = 1.0f / sum;
            for (int i = 0; i < n; i++) cdf[i] *= inv;
        }
    }
}

// ============================================================================
// [4b] Systematic Resampling — 38-dim
// ============================================================================
__global__ void systematicResampleKernel(const float* cdf,
                                          const float* particles_in,
                                          float* particles_out,
                                          float u0, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    float u = (u0 + (float)tid) / (float)n;
    if (u >= 1.0f) u -= 1.0f;
    int idx = findBin(cdf, u, n);

    // Copy full 38-dim state
    for (int d = 0; d < STATE_DIM; d++) {
        particles_out[tid * STATE_DIM + d] = particles_in[idx * STATE_DIM + d];
    }
}

// ============================================================================
// [7] Weighted State Estimate — 38-dim
// ============================================================================
__global__ void estimateStateKernel(const float* particles, const float* weights,
                                     float* state_out, float* covariance_out,
                                     int num_particles)
{
    int tid = threadIdx.x;

    float mean[STATE_DIM] = {};
    float var[STATE_DIM]  = {};

    for (int i = tid; i < num_particles; i += blockDim.x) {
        float w = weights[i];
        const float* s = &particles[i * STATE_DIM];
        for (int d = 0; d < STATE_DIM; d++) mean[d] += w * s[d];
    }
    for (int d = 0; d < STATE_DIM; d++) mean[d] = blockReduceSum(mean[d]);

    __shared__ float s_mean[STATE_DIM];
    if (tid == 0) {
        for (int d = 0; d < STATE_DIM; d++) s_mean[d] = mean[d];
    }
    __syncthreads();

    for (int i = tid; i < num_particles; i += blockDim.x) {
        float w = weights[i];
        const float* s = &particles[i * STATE_DIM];
        for (int d = 0; d < STATE_DIM; d++) {
            float diff = s[d] - s_mean[d];
            var[d] += w * diff * diff;
        }
    }
    for (int d = 0; d < STATE_DIM; d++) var[d] = blockReduceSum(var[d]);

    if (tid == 0) {
        for (int d = 0; d < STATE_DIM; d++) {
            state_out[d]      = s_mean[d];
            covariance_out[d] = var[d];
        }
    }
}

// ============================================================================
// [8] ESS
// ============================================================================
__global__ void essKernel(const float* weights, float* block_ess, int n)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float w2 = (tid < n) ? weights[tid] * weights[tid] : 0.0f;
    w2 = blockReduceSum(w2);
    if (threadIdx.x == 0) block_ess[blockIdx.x] = w2;
}

} // namespace phase7
