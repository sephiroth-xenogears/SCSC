#pragma once
// ============================================================================
// Phase 7: Full-Body 17-Axis Active Inference — EFE Planner (M6)
// Hierarchical 2-stage Expected Free Energy computation
// Stage 1: Per-group independent evaluation (16 candidates × 5 groups = 80)
// Stage 2: Top-K combination → joint EFE (64 final candidates)
// Ref: SCSC-P7-ARCH-001 §5.4 — PATCHED for real Manoi PF01 (5 groups)
// ============================================================================

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "primitives.cuh"
#include "gen_model.cuh"
#include "../include/types.h"
#include "../include/body_config.h"

namespace phase7 {

// ── EFE Constants ──
static constexpr int CANDIDATES_PER_GROUP = 16;
static constexpr int TOP_K_PER_GROUP      = 4;
static constexpr int NUM_FINAL_CANDIDATES = 64;
static constexpr int EFE_BLOCK_SIZE       = 256;

// ============================================================================
// Stage 1: Per-Group EFE Evaluation
// Launch: <<<CANDIDATES_PER_GROUP, EFE_BLOCK_SIZE>>> per group (6 launches)
//   or <<<CANDIDATES_PER_GROUP * NUM_GROUPS, EFE_BLOCK_SIZE>>> combined
// Each block = 1 action candidate for 1 group
// ============================================================================
__global__ void efeGroupKernel(
    const float* particles,         // [N_P × STATE_DIM]
    const float* weights,           // [N_P]
    const float* group_candidates,  // [NUM_GROUPS × CANDIDATES_PER_GROUP × max_group_action]
    float* group_efe_values,        // [NUM_GROUPS × CANDIDATES_PER_GROUP]
    const body_config_t* cfg,
    const GenModelParams* params,
    int group_id,
    int num_particles)
{
    int cand_idx = blockIdx.x;  // which candidate within this group
    int tid = threadIdx.x;

    if (cand_idx >= CANDIDATES_PER_GROUP) return;

    int state_off  = cfg->group_offset[group_id];
    int action_off = cfg->group_action_offset[group_id];
    int act_sz     = cfg->group_action_size[group_id];
    int num_joints = cfg->group_size[group_id];

    const auto& dyn = params->group[group_id];

    // Load this candidate's action for this group
    const float* cand = &group_candidates[
        (group_id * CANDIDATES_PER_GROUP + cand_idx) * ACTION_DIM];

    float risk_sum     = 0.0f;
    float ambiguity_sum = 0.0f;

    for (int i = tid; i < num_particles; i += blockDim.x) {
        float w = weights[i];
        const float* s = &particles[i * STATE_DIM];

        // Predict one step for this group
        float pred_state[STATE_DIM];
        // Copy relevant portion
        for (int d = 0; d < STATE_DIM; d++) pred_state[d] = s[d];

        // Apply group transition
        for (int j = 0; j < num_joints; j++) {
            int si = state_off + j * 2;
            int ai = action_off + j;
            float theta = pred_state[si];
            float omega = pred_state[si + 1];
            float u     = cand[ai];

            float new_theta = theta + omega * dyn.dt;
            float new_omega = omega + (-dyn.k * theta - dyn.d * omega + u) * dyn.dt;
            new_theta = fminf(fmaxf(new_theta, cfg->joint_min[ai]), cfg->joint_max[ai]);

            pred_state[si]     = new_theta;
            pred_state[si + 1] = new_omega;
        }

        // Also predict target if this is the neck group (affects visual obs)
        if (group_id == 0) {
            pred_state[state::TARGET_X] += pred_state[state::TARGET_VX] * dyn.dt;
            pred_state[state::TARGET_Y] += pred_state[state::TARGET_VY] * dyn.dt;
        }

        // Compute risk
        if (group_id == 0) {
            // Neck: visual preference
            float pred_cx, pred_cy, pred_w, pred_h;
            predictVisual(pred_state, pred_cx, pred_cy, pred_w, pred_h,
                          params->default_face_size, params->face_aspect_ratio);
            float r = computeRisk(pred_cx, pred_cy, pred_w,
                                  params->pref_face_cx, params->pref_face_cy,
                                  params->pref_face_size);
            risk_sum += w * r;
            ambiguity_sum += w * computeAmbiguity(pred_cx, pred_cy);
        } else {
            // Limbs/waist: deviation from standing pose
            float r = 0.0f;
            for (int j = 0; j < num_joints; j++) {
                int si = state_off + j * 2;
                int ai = action_off + j;
                float diff = pred_state[si] - params->stand_pose[ai];
                r += diff * diff;
            }
            risk_sum += w * r;
        }
    }

    risk_sum     = blockReduceSum(risk_sum);
    ambiguity_sum = blockReduceSum(ambiguity_sum);

    if (tid == 0) {
        int out_idx = group_id * CANDIDATES_PER_GROUP + cand_idx;
        group_efe_values[out_idx] = params->efe_risk_weight * risk_sum
                                  + params->efe_ambiguity_weight * ambiguity_sum;
    }
}

// ============================================================================
// Stage 1.5: Select Top-K per group
// Launch: <<<1, NUM_GROUPS>>> or single thread
// ============================================================================
__global__ void selectTopKKernel(
    const float* group_efe_values,   // [NUM_GROUPS × CANDIDATES_PER_GROUP]
    int* top_k_indices,               // [NUM_GROUPS × TOP_K_PER_GROUP]
    float* top_k_efe)                 // [NUM_GROUPS × TOP_K_PER_GROUP]
{
    int g = threadIdx.x;
    if (g >= NUM_GROUPS) return;

    const float* efe = &group_efe_values[g * CANDIDATES_PER_GROUP];

    // Simple selection sort for top-K (K=4, N=16 — fast enough)
    bool used[CANDIDATES_PER_GROUP] = {};
    for (int k = 0; k < TOP_K_PER_GROUP; k++) {
        int best = -1;
        float best_val = FLT_MAX;
        for (int c = 0; c < CANDIDATES_PER_GROUP; c++) {
            if (!used[c] && efe[c] < best_val) {
                best_val = efe[c];
                best = c;
            }
        }
        used[best] = true;
        top_k_indices[g * TOP_K_PER_GROUP + k] = best;
        top_k_efe[g * TOP_K_PER_GROUP + k] = best_val;
    }
}

// ============================================================================
// Stage 2: Combined EFE — evaluate joint action from top-K combinations
// Launch: <<<NUM_FINAL_CANDIDATES, EFE_BLOCK_SIZE>>>
// Each block evaluates one combined candidate across all groups
// ============================================================================
__global__ void efeCombinedKernel(
    const float* particles,
    const float* weights,
    const float* group_candidates,    // [NUM_GROUPS × CANDIDATES_PER_GROUP × ACTION_DIM]
    const int* top_k_indices,          // [NUM_GROUPS × TOP_K_PER_GROUP]
    const int* combination_map,        // [NUM_FINAL_CANDIDATES × NUM_GROUPS] → candidate index
    float* combined_efe_values,        // [NUM_FINAL_CANDIDATES]
    float* combined_efe_per_group,     // [NUM_FINAL_CANDIDATES × NUM_GROUPS]
    const body_config_t* cfg,
    const GenModelParams* params,
    int num_particles)
{
    int combo_idx = blockIdx.x;
    int tid = threadIdx.x;

    if (combo_idx >= NUM_FINAL_CANDIDATES) return;

    // Assemble full action from combination map
    float full_action[ACTION_DIM] = {};
    for (int g = 0; g < NUM_GROUPS; g++) {
        int cand_local = combination_map[combo_idx * NUM_GROUPS + g];
        int cand_global = top_k_indices[g * TOP_K_PER_GROUP + cand_local];
        const float* ga = &group_candidates[
            (g * CANDIDATES_PER_GROUP + cand_global) * ACTION_DIM];

        int off = cfg->group_action_offset[g];
        int sz  = cfg->group_action_size[g];
        for (int j = 0; j < sz; j++) {
            full_action[off + j] = ga[off + j];
        }
    }

    // Evaluate full-body EFE
    float total_risk      = 0.0f;
    float total_ambiguity = 0.0f;
    float group_efe[NUM_GROUPS] = {};

    for (int i = tid; i < num_particles; i += blockDim.x) {
        float w = weights[i];
        const float* s = &particles[i * STATE_DIM];

        // Predict full state one step
        float pred[STATE_DIM];
        for (int d = 0; d < STATE_DIM; d++) pred[d] = s[d];

        // Target
        float dt0 = params->group[0].dt;
        pred[state::TARGET_X] += pred[state::TARGET_VX] * dt0;
        pred[state::TARGET_Y] += pred[state::TARGET_VY] * dt0;

        // All groups
        for (int g = 0; g < NUM_GROUPS; g++) {
            const auto& dyn = params->group[g];
            int state_off  = cfg->group_offset[g];
            int action_off = cfg->group_action_offset[g];
            int nj = cfg->group_size[g];

            float g_risk = 0.0f;
            for (int j = 0; j < nj; j++) {
                int si = state_off + j * 2;
                int ai = action_off + j;

                float theta = pred[si];
                float omega = pred[si + 1];
                float u     = full_action[ai];

                pred[si]     = theta + omega * dyn.dt;
                pred[si + 1] = omega + (-dyn.k * theta - dyn.d * omega + u) * dyn.dt;
                pred[si]     = fminf(fmaxf(pred[si], cfg->joint_min[ai]), cfg->joint_max[ai]);

                if (g == 0) {
                    // Neck: visual risk computed below
                } else {
                    float diff = pred[si] - params->stand_pose[ai];
                    g_risk += diff * diff;
                }
            }

            if (g == 0) {
                float pred_cx, pred_cy, pred_w, pred_h;
                predictVisual(pred, pred_cx, pred_cy, pred_w, pred_h,
                              params->default_face_size, params->face_aspect_ratio);
                g_risk = computeRisk(pred_cx, pred_cy, pred_w,
                                     params->pref_face_cx, params->pref_face_cy,
                                     params->pref_face_size);
                total_ambiguity += w * computeAmbiguity(pred_cx, pred_cy);
            }
            group_efe[g] += w * g_risk;
        }
    }

    // Reduce
    total_ambiguity = blockReduceSum(total_ambiguity);
    for (int g = 0; g < NUM_GROUPS; g++) {
        group_efe[g] = blockReduceSum(group_efe[g]);
        total_risk += group_efe[g];
    }

    if (tid == 0) {
        combined_efe_values[combo_idx] = params->efe_risk_weight * total_risk
                                       + params->efe_ambiguity_weight * total_ambiguity;
        for (int g = 0; g < NUM_GROUPS; g++) {
            combined_efe_per_group[combo_idx * NUM_GROUPS + g] = group_efe[g];
        }
    }
}

// ============================================================================
// [6] Softmax Action Selection — from final candidates
// Launch: <<<1, NUM_FINAL_CANDIDATES>>>
// ============================================================================
__global__ void softmaxActionKernel(
    const float* efe_values,
    const float* group_candidates,
    const int* top_k_indices,
    const int* combination_map,
    float* selected_action,           // [ACTION_DIM]
    float* selected_efe,
    float* selected_efe_per_group,    // [NUM_GROUPS]
    float* selected_confidence,
    const body_config_t* cfg,
    float temperature,
    curandState* rng,
    int num_candidates)
{
    int tid = threadIdx.x;
    if (tid >= num_candidates) return;

    __shared__ float s_neg_efe[NUM_FINAL_CANDIDATES];
    __shared__ float s_probs[NUM_FINAL_CANDIDATES];
    __shared__ float s_cdf[NUM_FINAL_CANDIDATES];

    s_neg_efe[tid] = -efe_values[tid] / temperature;
    __syncthreads();

    // Find max
    float max_val = s_neg_efe[0];
    for (int i = 1; i < num_candidates; i++)
        max_val = fmaxf(max_val, s_neg_efe[i]);

    float exp_val = expf(s_neg_efe[tid] - max_val);
    s_probs[tid] = exp_val;
    __syncthreads();

    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < num_candidates; i++) sum += s_probs[i];
        float inv_sum = 1.0f / (sum + 1e-8f);
        for (int i = 0; i < num_candidates; i++) s_probs[i] *= inv_sum;

        s_cdf[0] = s_probs[0];
        for (int i = 1; i < num_candidates; i++)
            s_cdf[i] = s_cdf[i-1] + s_probs[i];

        float u = curand_uniform(&rng[0]);
        int selected = 0;
        for (int i = 0; i < num_candidates; i++) {
            if (u <= s_cdf[i]) { selected = i; break; }
        }

        // Assemble selected full action
        for (int d = 0; d < ACTION_DIM; d++) selected_action[d] = 0.0f;
        for (int g = 0; g < NUM_GROUPS; g++) {
            int cand_local = combination_map[selected * NUM_GROUPS + g];
            int cand_global = top_k_indices[g * TOP_K_PER_GROUP + cand_local];
            const float* ga = &group_candidates[
                (g * CANDIDATES_PER_GROUP + cand_global) * ACTION_DIM];

            int off = cfg->group_action_offset[g];
            int sz  = cfg->group_action_size[g];
            for (int j = 0; j < sz; j++) {
                selected_action[off + j] = ga[off + j];
            }
        }

        *selected_efe = efe_values[selected];
        *selected_confidence = s_probs[selected];
    }
}

// ============================================================================
// Host helper: Generate action candidates for all groups
// ============================================================================
inline void generateGroupCandidates(
    float* h_candidates,           // [NUM_GROUPS × CANDIDATES_PER_GROUP × ACTION_DIM]
    const body_config_t& cfg,
    unsigned int seed)
{
    std::srand(seed);
    for (int g = 0; g < NUM_GROUPS; g++) {
        int off = cfg.group_action_offset[g];
        int sz  = cfg.group_action_size[g];

        for (int c = 0; c < CANDIDATES_PER_GROUP; c++) {
            int base = (g * CANDIDATES_PER_GROUP + c) * ACTION_DIM;

            // Zero out full action, then fill group dims
            for (int d = 0; d < ACTION_DIM; d++)
                h_candidates[base + d] = 0.0f;

            for (int j = 0; j < sz; j++) {
                float range = cfg.joint_vel_max[off + j];
                float val = ((float)std::rand() / RAND_MAX * 2.0f - 1.0f) * range;
                h_candidates[base + off + j] = val;
            }
        }
    }
}

// ============================================================================
// Host helper: Generate combination map for Stage 2
// Maps each of NUM_FINAL_CANDIDATES to a choice of top-K per group
// ============================================================================
inline void generateCombinationMap(
    int* h_map,                     // [NUM_FINAL_CANDIDATES × NUM_GROUPS]
    unsigned int seed)
{
    std::srand(seed);
    for (int c = 0; c < NUM_FINAL_CANDIDATES; c++) {
        for (int g = 0; g < NUM_GROUPS; g++) {
            h_map[c * NUM_GROUPS + g] = std::rand() % TOP_K_PER_GROUP;
        }
    }
}

} // namespace phase7
