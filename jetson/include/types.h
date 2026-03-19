#pragma once
// ============================================================================
// Phase 7: Full-Body 17-Axis Active Inference — Type Definitions
// All structures are POD for safe GPU/CPU memory copy
// Ref: SCSC-P7-ARCH-001 §4.1
// ============================================================================

#include <cstdint>
#include <cstring>

namespace phase7 {

// ── Dimensions ──
static constexpr int NUM_JOINTS        = 17;
static constexpr int NUM_GROUPS        = 5;  // PATCHED: 6→5 (Waist group removed)
static constexpr int STATE_DIM         = 38;
static constexpr int ACTION_DIM        = 17;
static constexpr int OBS_DIM_VISUAL    = 5;   // detected, cx, cy, w, h
static constexpr int OBS_DIM_SERVO     = 34;  // 17 pos + 17 current
static constexpr int OBS_DIM_TOTAL     = OBS_DIM_VISUAL + OBS_DIM_SERVO; // 39

// ── State Vector Indices (STATE_DIM = 38) §4.1.2 ──
//  PATCHED: Real Manoi PF01 layout
//  [0..3]   External: target_x, target_y, target_vx, target_vy
//  [4..5]   Head:      pan_θ, pan_ω                           (1 joint)
//  [6..11]  Left Arm:  sh_p_θ/ω, sh_r_θ/ω, elb_θ/ω          (3 joints)
//  [12..17] Right Arm: (same structure as left arm)            (3 joints)
//  [18..27] Left Leg:  hip_p_θ/ω, hip_r_θ/ω, knee_θ/ω,
//                      ank_p_θ/ω, ank_r_θ/ω                   (5 joints)
//  [28..37] Right Leg: (same structure as left leg)            (5 joints)

// State segment indices
namespace state {
    // External world
    static constexpr int TARGET_X   = 0;
    static constexpr int TARGET_Y   = 1;
    static constexpr int TARGET_VX  = 2;
    static constexpr int TARGET_VY  = 3;

    // Head (1 axis: pan only — real Manoi PF01 has no tilt servo)
    static constexpr int PAN_THETA  = 4;
    static constexpr int PAN_OMEGA  = 5;

    // Group offsets (θ of first joint in each motor group)
    static constexpr int LEFT_ARM_START   = 6;
    static constexpr int RIGHT_ARM_START  = 12;
    static constexpr int LEFT_LEG_START   = 18;
    static constexpr int RIGHT_LEG_START  = 28;
}

// ── Observation Vector Indices (Visual part, Phase 6 compatible) ──
namespace obs {
    static constexpr int DETECTED    = 0;
    static constexpr int BBOX_CX     = 1;
    static constexpr int BBOX_CY     = 2;
    static constexpr int BBOX_WIDTH  = 3;
    static constexpr int BBOX_HEIGHT = 4;
}

// ── Joint Group Definition §4.1.1 ── PATCHED: 5 groups (Waist removed)
enum class JointGroup : int {
    HEAD      = 0,   // 1 axis: pan
    LEFT_ARM  = 1,   // 3 axes
    RIGHT_ARM = 2,   // 3 axes
    LEFT_LEG  = 3,   // 5 axes (ankle roll added)
    RIGHT_LEG = 4    // 5 axes (ankle roll added)
};

// ── Joint Index (maps to action_t / servo arrays) ── PATCHED: real Manoi PF01
enum JointIdx : int {
    HEAD_PAN        = 0,
    L_SH_PITCH      = 1,
    L_SH_ROLL       = 2,
    L_ELBOW         = 3,
    R_SH_PITCH      = 4,
    R_SH_ROLL       = 5,
    R_ELBOW         = 6,
    L_HIP_PITCH     = 7,
    L_HIP_ROLL      = 8,
    L_KNEE          = 9,
    L_ANKLE_PITCH   = 10,  // was L_ANKLE
    L_ANKLE_ROLL    = 11,  // NEW
    R_HIP_PITCH     = 12,
    R_HIP_ROLL      = 13,
    R_KNEE          = 14,
    R_ANKLE_PITCH   = 15,  // was R_ANKLE
    R_ANKLE_ROLL    = 16   // NEW (replaces WAIST_YAW)
};

// ── body_config_t §4.1.1 ──
struct body_config_t {
    int   group_offset[NUM_GROUPS];     // First index in state for each group
    int   group_size[NUM_GROUPS];       // #joints per group (for state: ×2 for θ,ω)
    int   group_action_offset[NUM_GROUPS]; // First index in action for each group
    int   group_action_size[NUM_GROUPS];   // #action dims per group

    float joint_min[NUM_JOINTS];        // Joint angle lower limit (rad)
    float joint_max[NUM_JOINTS];        // Joint angle upper limit (rad)
    float joint_vel_max[NUM_JOINTS];    // Max angular velocity (rad/s)

    uint8_t ics_id[NUM_JOINTS];         // ICS servo ID
    uint8_t uart_ch[NUM_JOINTS];        // Axis → UART channel mapping

    // Map: joint index → state index of its theta
    int   joint_to_state[NUM_JOINTS];
};

// ── state_t §4.1.2 — 38-dim flat state vector ──
struct state_t {
    float data[STATE_DIM];

    void clear() { std::memset(data, 0, sizeof(data)); }
};

// ── observation_t §4.1.3 ──
struct observation_t {
    // Visual
    bool  face_detected;
    float bbox_cx;            // Normalized [-1, 1]
    float bbox_cy;
    float bbox_w;
    float bbox_h;

    // Servo feedback
    float joint_pos[NUM_JOINTS];  // Current angle (rad)
    float joint_cur[NUM_JOINTS];  // Load current (mA)

    uint64_t timestamp_us;
    uint32_t valid_mask;      // Bit mask: bit0=visual, bit1..17=servo[0..16]

    static constexpr uint32_t VISUAL_VALID = 0x1;
    static constexpr uint32_t SERVO_VALID_BASE = 0x2; // bit 1..17

    bool isServoValid(int joint) const {
        return (valid_mask >> (joint + 1)) & 1;
    }

    void setServoValid(int joint) {
        valid_mask |= (1u << (joint + 1));
    }
};

// ── action_t §4.1.4 ──
struct action_t {
    float joint_cmd[NUM_JOINTS];     // Target angular velocity (rad/s)
    float efe_value;                 // Selected action EFE
    float efe_components[NUM_GROUPS]; // Per-group EFE breakdown
    float confidence;                // Softmax probability [0,1]
    bool  exploration_flag;
    uint64_t timestamp_us;

    void clear() {
        std::memset(joint_cmd, 0, sizeof(joint_cmd));
        efe_value = 0.0f;
        std::memset(efe_components, 0, sizeof(efe_components));
        confidence = 0.0f;
        exploration_flag = false;
        timestamp_us = 0;
    }
};

// ── servo_packet_t §4.1.5 — UART wire format ──
struct servo_packet_t {
    static constexpr uint8_t SYNC_0 = 0xAA;
    static constexpr uint8_t SYNC_1 = 0x55;
    static constexpr int MAX_PAYLOAD = 3 * NUM_JOINTS; // axis_id(1B) + data(2B)
    static constexpr int MAX_PACKET  = 2 + 1 + 1 + 1 + MAX_PAYLOAD + 1; // 57B

    uint8_t header[2];      // 0xAA, 0x55
    uint8_t cmd_type;       // Command type
    uint8_t seq_num;        // Sequence number (0-255 wrap)
    uint8_t axis_count;     // Number of axes
    uint8_t payload[MAX_PAYLOAD];
    uint8_t checksum;       // XOR of header through payload

    // Command IDs
    static constexpr uint8_t CMD_SET_POSITION  = 0x01;
    static constexpr uint8_t CMD_READ_STATUS   = 0x02;
    static constexpr uint8_t CMD_SET_SPEED     = 0x03;
    static constexpr uint8_t CMD_SET_COMPLIANCE= 0x04;
    static constexpr uint8_t CMD_HEARTBEAT     = 0x0E;
    static constexpr uint8_t CMD_EMERGENCY_STOP= 0x0F;

    // Response IDs
    static constexpr uint8_t RSP_ACK_POSITION  = 0x81;
    static constexpr uint8_t RSP_STATUS        = 0x82;
    static constexpr uint8_t RSP_HB_ACK        = 0x8E;
    static constexpr uint8_t RSP_ESTOP_ACK     = 0x8F;
    static constexpr uint8_t RSP_NACK          = 0xFF;
};

// ── servo_feedback_t — received from Teensy ──
struct servo_feedback_t {
    float position[NUM_JOINTS];  // (rad)
    float current[NUM_JOINTS];   // (mA)
    uint64_t timestamp_us;
    uint32_t valid_mask;         // Which axes have valid data
};

// ── safety_status_t ──
struct safety_status_t {
    enum Level : uint8_t {
        NORMAL     = 0,
        SOFT_LIMIT = 1,  // L0
        SOFT_STOP  = 2,  // L1
        COMM_STOP  = 3,  // L2
        HARD_STOP  = 4   // L3
    };

    Level  level;
    uint32_t fault_mask;          // Which joints triggered
    float  efe_value;             // EFE at trigger time
    uint64_t timestamp_us;
};

// ── log_entry_t ──
struct log_entry_t {
    uint64_t timestamp_us;
    state_t  estimated_state;
    action_t action;
    observation_t observation;
    float    efe_total;
    float    efe_per_group[NUM_GROUPS];
    float    ess;                 // Effective sample size
    safety_status_t safety;
};

// ── FaceDetection (CPU-side, Phase 6 compatible) ──
struct FaceDetection {
    float x          = 0.0f;   // pixel
    float y          = 0.0f;
    float w          = 0.0f;
    float h          = 0.0f;
    float confidence = 0.0f;
    bool  detected   = false;
};

// ── GPU observation flat vector (for kernel upload) ──
struct gpu_observation_t {
    float visual[OBS_DIM_VISUAL];
    float joint_pos[NUM_JOINTS];
    float joint_cur[NUM_JOINTS];
    uint32_t valid_mask;
};

} // namespace phase7
