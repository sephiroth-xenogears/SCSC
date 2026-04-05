#pragma once
// ============================================================================
// Phase 7: Full-Body 17-Axis Active Inference — Safety Monitor (M10)
// Jetson-side safety: L0 (soft limit) and L1 (soft stop)
// Ref: SCSC-P7-ARCH-001 §9
// ============================================================================

#include "../include/types.h"
#include "../include/body_config.h"
#include <cstdio>
#include <cmath>

namespace phase7 {

class SafetyMonitor {
public:
    void init(const body_config_t& bcfg, const SafetyConfig& scfg) {
        bcfg_ = bcfg;
        scfg_ = scfg;
        status_.level = safety_status_t::NORMAL;
        status_.fault_mask = 0;
        efe_log_counter_ = 0;
        std::printf("[M10:Safety] Initialized (soft_limit=%.0f%%, overcurrent=%.0fmA, efe_thresh=%.0f)\n",
                    scfg_.soft_limit_ratio * 100.0f, scfg_.overcurrent_mA, scfg_.efe_anomaly_thresh);
    }

    // ── Validate and clamp action (called before sending to Teensy) ──
    action_t validateAction(const action_t& raw, const servo_feedback_t& fb) {
        action_t safe = raw;
        status_.level = safety_status_t::NORMAL;
        status_.fault_mask = 0;

        for (int j = 0; j < NUM_JOINTS; j++) {
            float cmd = safe.joint_cmd[j];
            float pos = fb.position[j];
            float cur = fb.current[j];

            float range = bcfg_.joint_max[j] - bcfg_.joint_min[j];
            float margin = range * (1.0f - scfg_.soft_limit_ratio);

            // ── L0: Soft limit — reduce velocity near joint limits ──
            float dist_to_min = pos - bcfg_.joint_min[j];
            float dist_to_max = bcfg_.joint_max[j] - pos;

            if (dist_to_min < margin && cmd < 0.0f) {
                float scale = dist_to_min / margin;
                safe.joint_cmd[j] *= std::fmax(0.0f, scale);
                if (status_.level < safety_status_t::SOFT_LIMIT)
                    status_.level = safety_status_t::SOFT_LIMIT;
                status_.fault_mask |= (1u << j);
            }
            if (dist_to_max < margin && cmd > 0.0f) {
                float scale = dist_to_max / margin;
                safe.joint_cmd[j] *= std::fmax(0.0f, scale);
                if (status_.level < safety_status_t::SOFT_LIMIT)
                    status_.level = safety_status_t::SOFT_LIMIT;
                status_.fault_mask |= (1u << j);
            }

            // ── L1: Hard clamp to velocity limits ──
            float vmax = bcfg_.joint_vel_max[j];
            safe.joint_cmd[j] = std::fmin(vmax, std::fmax(-vmax, safe.joint_cmd[j]));

            // ── L1: Overcurrent check ──
            if (std::fabs(cur) > scfg_.overcurrent_mA) {
                safe.joint_cmd[j] = 0.0f;
                status_.level = safety_status_t::SOFT_STOP;
                status_.fault_mask |= (1u << j);
                std::fprintf(stderr, "[M10:Safety] Overcurrent joint %d: %.0fmA\n",
                             j, cur);
            }
        }

        // ── L1: EFE anomaly (once per step, outside joint loop) ──
        if (raw.efe_value > scfg_.efe_anomaly_thresh) {
            if (status_.level < safety_status_t::SOFT_STOP)
                status_.level = safety_status_t::SOFT_STOP;
            // Log at most once per 500 steps to avoid spam
            if (efe_log_counter_++ % 500 == 0) {
                std::fprintf(stderr, "[M10:Safety] EFE anomaly: %.2f (log throttled)\n",
                             raw.efe_value);
            }
        } else {
            efe_log_counter_ = 0;
        }

        auto now = std::chrono::steady_clock::now().time_since_epoch();
        status_.timestamp_us = std::chrono::duration_cast<
            std::chrono::microseconds>(now).count();
        status_.efe_value = raw.efe_value;

        return safe;
    }

    safety_status_t getStatus() const { return status_; }

private:
    body_config_t bcfg_;
    SafetyConfig  scfg_;
    safety_status_t status_;
    int efe_log_counter_ = 0;
};

} // namespace phase7