#pragma once
// ============================================================================
// Phase 7: Full-Body 17-Axis Active Inference — Observation Builder (M3)
// Merges visual detection (M2) + servo feedback (M8) → observation_t
// Ref: SCSC-P7-ARCH-001 §3.1 M3
// ============================================================================

#include "../include/types.h"
#include <chrono>
#include <cstring>

namespace phase7 {

class ObservationBuilder {
public:
    // Build observation from face detection + servo feedback
    observation_t build(const FaceDetection& face, int img_w, int img_h,
                        const servo_feedback_t* servo_fb = nullptr)
    {
        observation_t obs;
        std::memset(&obs, 0, sizeof(obs));

        auto now = std::chrono::steady_clock::now().time_since_epoch();
        obs.timestamp_us = std::chrono::duration_cast<
            std::chrono::microseconds>(now).count();

        // ── Visual observation ──
        obs.face_detected = face.detected;
        if (face.detected) {
            obs.bbox_cx = (face.x + face.w * 0.5f) / img_w * 2.0f - 1.0f;
            obs.bbox_cy = (face.y + face.h * 0.5f) / img_h * 2.0f - 1.0f;
            obs.bbox_w  = face.w / img_w * 2.0f;
            obs.bbox_h  = face.h / img_h * 2.0f;
            obs.valid_mask |= observation_t::VISUAL_VALID;
        }

        // ── Servo feedback ──
        if (servo_fb && servo_fb->valid_mask != 0) {
            for (int j = 0; j < NUM_JOINTS; j++) {
                if ((servo_fb->valid_mask >> j) & 1) {
                    obs.joint_pos[j] = servo_fb->position[j];
                    obs.joint_cur[j] = servo_fb->current[j];
                    obs.setServoValid(j);
                }
            }
        }

        return obs;
    }

    // Pack observation_t into flat GPU-uploadable format
    static gpu_observation_t toGPU(const observation_t& obs) {
        gpu_observation_t g;

        g.visual[obs::DETECTED]    = obs.face_detected ? 1.0f : 0.0f;
        g.visual[obs::BBOX_CX]    = obs.bbox_cx;
        g.visual[obs::BBOX_CY]    = obs.bbox_cy;
        g.visual[obs::BBOX_WIDTH]  = obs.bbox_w;
        g.visual[obs::BBOX_HEIGHT] = obs.bbox_h;

        for (int j = 0; j < NUM_JOINTS; j++) {
            g.joint_pos[j] = obs.joint_pos[j];
            g.joint_cur[j] = obs.joint_cur[j];
        }
        g.valid_mask = obs.valid_mask;

        return g;
    }
};

} // namespace phase7
