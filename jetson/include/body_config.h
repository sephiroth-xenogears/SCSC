#pragma once
// ============================================================================
// Phase 7: Full-Body 17-Axis Active Inference — Body Configuration (M9)
// Manoi PF01 + KRS-4024S HV servo parameters
// Ref: SCSC-P7-ARCH-001 §4.1.1, §9
// ============================================================================

#include "types.h"
#include <cmath>

namespace phase7 {

// ── Utility ──
static constexpr float DEG2RAD = static_cast<float>(M_PI) / 180.0f;

// ── Default Body Configuration ── PATCHED: real Manoi PF01 (5 groups)
inline body_config_t makeDefaultBodyConfig() {
    body_config_t cfg = {};

    // ── Group layout in STATE vector (θ,ω pairs) ──
    cfg.group_offset[0] = 4;   // Head   (1 joint × 2 = 2 states)
    cfg.group_offset[1] = 6;   // L Arm  (3 joints × 2 = 6 states)
    cfg.group_offset[2] = 12;  // R Arm  (3 joints × 2 = 6 states)
    cfg.group_offset[3] = 18;  // L Leg  (5 joints × 2 = 10 states)
    cfg.group_offset[4] = 28;  // R Leg  (5 joints × 2 = 10 states)

    cfg.group_size[0] = 1;
    cfg.group_size[1] = 3;
    cfg.group_size[2] = 3;
    cfg.group_size[3] = 5;
    cfg.group_size[4] = 5;

    cfg.group_action_offset[0] = 0;   // HEAD_PAN
    cfg.group_action_offset[1] = 1;   // L_SH_PITCH..L_ELBOW
    cfg.group_action_offset[2] = 4;   // R_SH_PITCH..R_ELBOW
    cfg.group_action_offset[3] = 7;   // L_HIP_PITCH..L_ANKLE_ROLL
    cfg.group_action_offset[4] = 12;  // R_HIP_PITCH..R_ANKLE_ROLL

    cfg.group_action_size[0] = 1;
    cfg.group_action_size[1] = 3;
    cfg.group_action_size[2] = 3;
    cfg.group_action_size[3] = 5;
    cfg.group_action_size[4] = 5;

    // ── Joint → State index mapping ──
    cfg.joint_to_state[HEAD_PAN]        = 4;
    cfg.joint_to_state[L_SH_PITCH]      = 6;
    cfg.joint_to_state[L_SH_ROLL]       = 8;
    cfg.joint_to_state[L_ELBOW]         = 10;
    cfg.joint_to_state[R_SH_PITCH]      = 12;
    cfg.joint_to_state[R_SH_ROLL]       = 14;
    cfg.joint_to_state[R_ELBOW]         = 16;
    cfg.joint_to_state[L_HIP_PITCH]     = 18;
    cfg.joint_to_state[L_HIP_ROLL]      = 20;
    cfg.joint_to_state[L_KNEE]          = 22;
    cfg.joint_to_state[L_ANKLE_PITCH]   = 24;
    cfg.joint_to_state[L_ANKLE_ROLL]    = 26;
    cfg.joint_to_state[R_HIP_PITCH]     = 28;
    cfg.joint_to_state[R_HIP_ROLL]      = 30;
    cfg.joint_to_state[R_KNEE]          = 32;
    cfg.joint_to_state[R_ANKLE_PITCH]   = 34;
    cfg.joint_to_state[R_ANKLE_ROLL]    = 36;

    // ── Joint Limits (rad) — KRS-4024S HV ──
    cfg.joint_min[HEAD_PAN]       = -1.57f;   cfg.joint_max[HEAD_PAN]       =  1.57f;  // ±90°

    cfg.joint_min[L_SH_PITCH]    = -2.00f;   cfg.joint_max[L_SH_PITCH]    =  2.00f;
    cfg.joint_min[L_SH_ROLL]     = -1.50f;   cfg.joint_max[L_SH_ROLL]     =  0.30f;
    cfg.joint_min[L_ELBOW]       = -2.40f;   cfg.joint_max[L_ELBOW]       =  0.00f;

    cfg.joint_min[R_SH_PITCH]    = -2.00f;   cfg.joint_max[R_SH_PITCH]    =  2.00f;
    cfg.joint_min[R_SH_ROLL]     = -0.30f;   cfg.joint_max[R_SH_ROLL]     =  1.50f;
    cfg.joint_min[R_ELBOW]       =  0.00f;   cfg.joint_max[R_ELBOW]       =  2.40f;

    cfg.joint_min[L_HIP_PITCH]   = -1.80f;   cfg.joint_max[L_HIP_PITCH]   =  0.80f;
    cfg.joint_min[L_HIP_ROLL]    = -0.50f;   cfg.joint_max[L_HIP_ROLL]    =  0.80f;
    cfg.joint_min[L_KNEE]        =  0.00f;   cfg.joint_max[L_KNEE]        =  2.40f;
    cfg.joint_min[L_ANKLE_PITCH] = -0.52f;   cfg.joint_max[L_ANKLE_PITCH] =  0.87f;  // -30°..+50°
    cfg.joint_min[L_ANKLE_ROLL]  = -0.44f;   cfg.joint_max[L_ANKLE_ROLL]  =  0.44f;  // ±25°

    cfg.joint_min[R_HIP_PITCH]   = -0.80f;   cfg.joint_max[R_HIP_PITCH]   =  1.80f;
    cfg.joint_min[R_HIP_ROLL]    = -0.80f;   cfg.joint_max[R_HIP_ROLL]    =  0.50f;
    cfg.joint_min[R_KNEE]        = -2.40f;   cfg.joint_max[R_KNEE]        =  0.00f;
    cfg.joint_min[R_ANKLE_PITCH] = -0.52f;   cfg.joint_max[R_ANKLE_PITCH] =  0.87f;
    cfg.joint_min[R_ANKLE_ROLL]  = -0.44f;   cfg.joint_max[R_ANKLE_ROLL]  =  0.44f;

    // ── Max angular velocity (rad/s) ──
    for (int i = 0; i < NUM_JOINTS; i++) {
        cfg.joint_vel_max[i] = 6.0f;
    }
    cfg.joint_vel_max[HEAD_PAN]      = 4.0f;
    cfg.joint_vel_max[L_ANKLE_ROLL]  = 3.0f;
    cfg.joint_vel_max[R_ANKLE_ROLL]  = 3.0f;

    // ── ICS Servo IDs ──
    for (int i = 0; i < NUM_JOINTS; i++) {
        cfg.ics_id[i] = static_cast<uint8_t>(i);
    }

    // ── UART channel mapping (2ch: CH1=left body, CH2=right body) ──
    cfg.uart_ch[HEAD_PAN]       = 0;
    cfg.uart_ch[L_SH_PITCH]    = 0;  cfg.uart_ch[L_SH_ROLL]    = 0;  cfg.uart_ch[L_ELBOW]       = 0;
    cfg.uart_ch[L_HIP_PITCH]   = 0;  cfg.uart_ch[L_HIP_ROLL]   = 0;  cfg.uart_ch[L_KNEE]        = 0;
    cfg.uart_ch[L_ANKLE_PITCH] = 0;  cfg.uart_ch[L_ANKLE_ROLL] = 0;
    cfg.uart_ch[R_SH_PITCH]    = 1;  cfg.uart_ch[R_SH_ROLL]    = 1;  cfg.uart_ch[R_ELBOW]       = 1;
    cfg.uart_ch[R_HIP_PITCH]   = 1;  cfg.uart_ch[R_HIP_ROLL]   = 1;  cfg.uart_ch[R_KNEE]        = 1;
    cfg.uart_ch[R_ANKLE_PITCH] = 1;  cfg.uart_ch[R_ANKLE_ROLL] = 1;

    return cfg;
}

// ── Safety Thresholds ──
struct SafetyConfig {
    float soft_limit_ratio   = 0.90f;
    float overcurrent_mA     = 1500.0f;
    float position_error_rad = 0.35f;
    float efe_anomaly_thresh = 5000.0f;  // PATCHED rev.3: loopback without face yields high EFE
    uint32_t heartbeat_timeout_ms = 200;
};

inline SafetyConfig makeDefaultSafetyConfig() {
    return SafetyConfig{};
}

} // namespace phase7