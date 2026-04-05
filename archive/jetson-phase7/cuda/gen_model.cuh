#pragma once
// ============================================================================
// Phase 7: Full-Body 17-Axis Active Inference — Generative Model (M4)
// State transition, observation, and preference models (device code)
// Ref: SCSC-P7-ARCH-001 §5.2, §6.1, §6.2, §6.3
// ============================================================================

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "primitives.cuh"
#include "../include/types.h"
#include "../include/body_config.h"

namespace phase7 {

// ── Model hyperparameters (uploaded to constant memory) ──
struct GenModelParams {
    // Per-group dynamics (§6.1.1)
    struct GroupDynamics {
        float k;            // Stiffness
        float d;            // Damping
        float sigma_proc;   // Process noise
        float dt;           // Time step
    };
    GroupDynamics group[NUM_GROUPS];

    // Target motion model
    float target_pos_noise;
    float target_vel_noise;

    // Observation model (§5.3)
    float sigma_visual;     // Visual position noise
    float sigma_joint;      // Joint angle read noise
    float sigma_current;    // Current read noise
    float w_vis;            // Likelihood weight: visual
    float w_joint;          // Likelihood weight: joint
    float w_cur;            // Likelihood weight: current

    // Preference model (§6.3)
    float pref_face_cx;     // Preferred face center X
    float pref_face_cy;     // Preferred face center Y
    float pref_face_size;   // Preferred face size
    float sigma_pref_vis;   // Preference spread: visual
    float sigma_pref_stand; // Preference spread: standing pose

    // Standing pose (rest angles for legs, §6.3)
    float stand_pose[NUM_JOINTS];

    // EFE
    float efe_risk_weight;
    float efe_ambiguity_weight;
    float action_temperature;

    // Visibility
    float visibility_steepness;
    float default_face_size;
    float face_aspect_ratio;
};

// ── Default parameters ──
inline GenModelParams makeDefaultGenModelParams() {
    GenModelParams p = {};

    // Head (pan only — no tilt on real Manoi PF01)
    p.group[0] = {2.0f, 1.5f, 0.10f, 1.0f/30.0f};
    // Left Arm
    p.group[1] = {1.5f, 1.2f, 0.08f, 1.0f/100.0f};
    // Right Arm
    p.group[2] = {1.5f, 1.2f, 0.08f, 1.0f/100.0f};
    // Left Leg (5 axes incl. ankle roll)
    p.group[3] = {3.0f, 2.0f, 0.05f, 1.0f/100.0f};
    // Right Leg (5 axes incl. ankle roll)
    p.group[4] = {3.0f, 2.0f, 0.05f, 1.0f/100.0f};

    p.target_pos_noise = 0.02f;
    p.target_vel_noise = 0.05f;

    p.sigma_visual  = 0.05f;
    p.sigma_joint   = 0.009f;  // ±0.5° ≈ 0.0087 rad
    p.sigma_current = 50.0f;   // mA
    p.w_vis   = 1.0f;
    p.w_joint = 0.5f;
    p.w_cur   = 0.2f;

    p.pref_face_cx   = 0.0f;
    p.pref_face_cy   = 0.0f;
    p.pref_face_size = 0.25f;
    p.sigma_pref_vis   = 0.1f;
    p.sigma_pref_stand = 0.05f;

    // Standing pose: all zeros (upright)
    for (int i = 0; i < NUM_JOINTS; i++) p.stand_pose[i] = 0.0f;

    p.efe_risk_weight      = 1.0f;
    p.efe_ambiguity_weight = 1.0f;
    p.action_temperature   = 0.5f;

    p.visibility_steepness = 5.0f;
    p.default_face_size    = 0.25f;
    p.face_aspect_ratio    = 1.3f;

    return p;
}

// ============================================================================
// Device functions: State Transition Model (§6.1)
// ============================================================================

// Predict target (external world) — constant velocity + noise
__device__ inline void predictTarget(float* s, float dt, curandState& rng,
                                      float pos_noise, float vel_noise)
{
    s[state::TARGET_X]  += s[state::TARGET_VX] * dt + curand_normal(&rng) * pos_noise;
    s[state::TARGET_Y]  += s[state::TARGET_VY] * dt + curand_normal(&rng) * pos_noise;
    s[state::TARGET_VX] += curand_normal(&rng) * vel_noise;
    s[state::TARGET_VY] += curand_normal(&rng) * vel_noise;
}

// Predict a joint group — 2nd-order damped rotation (§6.1)
// θₙ₊₁ = θₙ + ωₙ · dt
// ωₙ₊₁ = ωₙ + (-k·θₙ - d·ωₙ + u) · dt + σ · ε
__device__ inline void predictJointGroup(
    float* s,                   // full state vector
    const float* action,        // action vector [NUM_JOINTS]
    int state_offset,           // first θ index in state
    int action_offset,          // first cmd index in action
    int num_joints,             // joints in this group
    float k, float d,           // stiffness, damping
    float sigma_proc, float dt, // noise, timestep
    curandState& rng,
    const float* joint_min,     // clamp limits
    const float* joint_max,
    const float* vel_max)
{
    for (int j = 0; j < num_joints; j++) {
        int si = state_offset + j * 2;     // θ index
        int ai = action_offset + j;         // action index

        float theta = s[si];
        float omega = s[si + 1];
        float u     = action[ai];

        // State transition
        float new_theta = theta + omega * dt;
        float new_omega = omega + (-k * theta - d * omega + u) * dt
                        + curand_normal(&rng) * sigma_proc;

        // Clamp
        new_theta = fminf(fmaxf(new_theta, joint_min[ai]), joint_max[ai]);
        new_omega = fminf(fmaxf(new_omega, -vel_max[ai]),  vel_max[ai]);

        s[si]     = new_theta;
        s[si + 1] = new_omega;
    }
}

// Inter-group coupling (§5.2): placeholder for kinematic coupling
// PATCHED: waist axis removed. Future: leg ankle roll → posture compensation
__device__ inline void applyCoupling(float* s) {
    // No-op — to be refined with real kinematics in Phase 8
    (void)s;
}

// ============================================================================
// Device functions: Observation Model (§6.2)
// ============================================================================

// Predicted visual observation from state
__device__ inline void predictVisual(const float* s,
                                      float& pred_cx, float& pred_cy,
                                      float& pred_w, float& pred_h,
                                      float default_size, float aspect)
{
    pred_cx = s[state::TARGET_X] - s[state::PAN_THETA];
    pred_cy = s[state::TARGET_Y];  // PATCHED: no tilt servo on real Manoi PF01
    pred_w  = default_size;
    pred_h  = default_size * aspect;
}

// ============================================================================
// Device functions: Preference Model (§6.3)
// ============================================================================

// Risk: deviation from preferred observation
__device__ inline float computeRisk(
    float pred_cx, float pred_cy, float pred_w,
    float pref_cx, float pref_cy, float pref_sz)
{
    float dx = pred_cx - pref_cx;
    float dy = pred_cy - pref_cy;
    float ds = pred_w  - pref_sz;
    return dx * dx + dy * dy + ds * ds;
}

// Ambiguity: observation uncertainty at FOV edge
__device__ inline float computeAmbiguity(float pred_cx, float pred_cy) {
    float edge_dist = fmaxf(fabsf(pred_cx), fabsf(pred_cy));
    return sigmoidf(3.0f * (edge_dist - 0.8f));
}

} // namespace phase7
