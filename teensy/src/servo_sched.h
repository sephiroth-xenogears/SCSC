// ============================================================================
// Phase 7: Teensy 4.1 Firmware — Servo Manager (T3)
// Joint configuration, angle limits, command dispatch to FlexPWM
// Ref: SCSC-P7-ARCH-001 §3.2 T3 (PWM revision)
// ============================================================================
#pragma once

#include <Arduino.h>
#include "pwm_servo.h"

// ── Joint Limit Configuration ──
struct JointLimits {
    int16_t min_cdeg;  // Minimum angle (centidegrees)
    int16_t max_cdeg;  // Maximum angle (centidegrees)
};

// ── Servo Manager ──
class ServoManager {
public:
    static constexpr int NUM_JOINTS = PwmServo::NUM_JOINTS;

    // ── Init ──
    void init(FlexPwmServoDriver* driver) {
        driver_ = driver;
        update_count_ = 0;

        // Default joint limits — Manoi PF01 / KRS-4024HV
        // ±135° hardware max, but tighten per-joint for mechanical safety
        setLimits(HEAD_PAN,       -9000,  9000);  // ±90°

        setLimits(L_SH_PITCH,   -11500, 11500);   // ±115°
        setLimits(L_SH_ROLL,     -8600,  1700);   // -86° .. +17°
        setLimits(L_ELBOW,      -13500,     0);    // -135° .. 0°

        setLimits(R_SH_PITCH,   -11500, 11500);   // ±115°
        setLimits(R_SH_ROLL,     -1700,  8600);   // -17° .. +86°
        setLimits(R_ELBOW,           0, 13500);    //   0° .. +135°

        setLimits(L_HIP_PITCH,  -10300,  4600);   // -103° .. +46°
        setLimits(L_HIP_ROLL,    -2900,  4600);   // -29° .. +46°
        setLimits(L_KNEE,            0, 13500);    //   0° .. +135°
        setLimits(L_ANKLE_PITCH, -3000,  5000);   // -30° .. +50°
        setLimits(L_ANKLE_ROLL,  -2500,  2500);   // ±25°

        setLimits(R_HIP_PITCH,   -4600, 10300);   // -46° .. +103°
        setLimits(R_HIP_ROLL,    -4600,  2900);   // -46° .. +29°
        setLimits(R_KNEE,       -13500,     0);    // -135° .. 0°
        setLimits(R_ANKLE_PITCH, -3000,  5000);   // -30° .. +50°
        setLimits(R_ANKLE_ROLL,  -2500,  2500);   // ±25°

        Serial.println("[T3:ServoMgr] Init complete with joint limits");
    }

    // ── Set angle with limit enforcement ──
    bool setAngle(uint8_t joint, int16_t angle_cdeg) {
        if (joint >= NUM_JOINTS || !driver_) return false;

        // Enforce joint limits
        if (angle_cdeg < limits_[joint].min_cdeg) angle_cdeg = limits_[joint].min_cdeg;
        if (angle_cdeg > limits_[joint].max_cdeg) angle_cdeg = limits_[joint].max_cdeg;

        driver_->setAngle(joint, angle_cdeg);
        return true;
    }

    // ── Set all 17 angles at once ──
    void setAllAngles(const int16_t* angles_cdeg) {
        for (int j = 0; j < NUM_JOINTS; j++) {
            setAngle(j, angles_cdeg[j]);
        }
    }

    // ── Apply to hardware ──
    void applyAll() {
        if (driver_) {
            driver_->applyAll();
            update_count_++;
        }
    }

    // ── Enable/Disable ──
    void enableAll() {
        if (driver_) driver_->enableAll();
    }

    void disableAll() {
        if (driver_) driver_->disableAll();
    }

    void setEnabled(uint8_t joint, bool en) {
        if (driver_) driver_->setEnabled(joint, en);
    }

    void setEnabledMask(uint32_t mask) {
        if (driver_) driver_->setEnabledMask(mask);
    }

    // ── Move all to center ──
    void centerAll() {
        if (driver_) driver_->centerAll();
    }

    // ── Queries ──
    int16_t getTargetAngle(uint8_t joint) const {
        return driver_ ? driver_->getTargetAngle(joint) : 0;
    }

    bool isEnabled(uint8_t joint) const {
        return driver_ ? driver_->isEnabled(joint) : false;
    }

    uint32_t getEnabledMask() const {
        return driver_ ? driver_->getEnabledMask() : 0;
    }

    uint32_t getUpdateCount() const { return update_count_; }

    JointLimits getLimits(uint8_t joint) const {
        if (joint < NUM_JOINTS) return limits_[joint];
        return {0, 0};
    }

    // ── Get all current angles for status report ──
    void getAllAngles(int16_t* out_angles) const {
        for (int j = 0; j < NUM_JOINTS; j++) {
            out_angles[j] = driver_ ? driver_->getTargetAngle(j) : 0;
        }
    }

private:
    void setLimits(uint8_t joint, int16_t min_cdeg, int16_t max_cdeg) {
        if (joint < NUM_JOINTS) {
            limits_[joint].min_cdeg = min_cdeg;
            limits_[joint].max_cdeg = max_cdeg;
        }
    }

    FlexPwmServoDriver* driver_ = nullptr;
    JointLimits limits_[NUM_JOINTS];
    uint32_t update_count_ = 0;
};
