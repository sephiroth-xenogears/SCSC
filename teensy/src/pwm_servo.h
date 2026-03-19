// ============================================================================
// Phase 7: Teensy 4.1 Firmware — FlexPWM Servo Driver (T2)
// KRS-4024HV (Red Version / ICS 2.0) — Standard PWM control
// PWM: 0.7ms(-135°) ~ 1.5ms(0°) ~ 2.3ms(+135°) @ 50Hz
// Ref: SCSC-P7-ARCH-001 §3.2 T2 (PWM revision)
// ============================================================================
#pragma once

#include <Arduino.h>

// ── PWM Servo Constants ──
namespace PwmServo {
    static constexpr int NUM_JOINTS = 17;

    // Pulse width (microseconds)
    static constexpr uint16_t PULSE_MIN_US   = 700;   // -135°
    static constexpr uint16_t PULSE_CENTER_US = 1500;  //    0°
    static constexpr uint16_t PULSE_MAX_US   = 2300;   // +135°
    static constexpr uint16_t PULSE_RANGE_US = PULSE_MAX_US - PULSE_MIN_US; // 1600

    // Angle (centidegrees: 0.01° units)
    static constexpr int16_t ANGLE_MIN_CDEG  = -13500; // -135.00°
    static constexpr int16_t ANGLE_MAX_CDEG  =  13500; // +135.00°
    static constexpr int32_t ANGLE_RANGE_CDEG = ANGLE_MAX_CDEG - ANGLE_MIN_CDEG; // 27000

    // PWM timing
    static constexpr float    PWM_FREQ_HZ    = 50.0f;
    static constexpr uint32_t PWM_PERIOD_US  = 20000;  // 1/50Hz
    static constexpr uint8_t  PWM_RESOLUTION = 16;     // 16-bit
    static constexpr uint32_t PWM_MAX_VALUE  = 65535;   // 2^16 - 1

    // Teensy 4.1 pin assignments (confirmed)
    // Pins 2-13: FlexPWM / QuadTimer
    // Pins 22-25, 28: FlexPWM
    static constexpr uint8_t SERVO_PINS[NUM_JOINTS] = {
         2,  // J0:  HEAD_PAN        (FlexPWM4_2_A)
         3,  // J1:  L_SH_PITCH      (FlexPWM4_2_B)
         4,  // J2:  L_SH_ROLL       (FlexPWM2_0_A)
         5,  // J3:  L_ELBOW         (FlexPWM2_1_A)
         6,  // J4:  R_SH_PITCH      (FlexPWM2_2_A)
         7,  // J5:  R_SH_ROLL       (FlexPWM1_3_B)
         8,  // J6:  R_ELBOW         (FlexPWM1_3_A)
         9,  // J7:  L_HIP_PITCH     (FlexPWM2_2_B)
        10,  // J8:  L_HIP_ROLL      (QuadTimer1_0)
        11,  // J9:  L_KNEE          (QuadTimer1_2)
        12,  // J10: L_ANKLE_PITCH   (QuadTimer1_1)
        13,  // J11: L_ANKLE_ROLL    (QuadTimer2_0)
        22,  // J12: R_HIP_PITCH     (FlexPWM4_0_A)
        23,  // J13: R_HIP_ROLL      (FlexPWM4_0_B)
        24,  // J14: R_KNEE          (FlexPWM1_2_A)
        25,  // J15: R_ANKLE_PITCH   (FlexPWM1_2_B)
        28   // J16: R_ANKLE_ROLL    (FlexPWM3_1_B)
    };
}

// ── Joint Index Enum ──
enum JointIdx : uint8_t {
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
    L_ANKLE_PITCH   = 10,
    L_ANKLE_ROLL    = 11,
    R_HIP_PITCH     = 12,
    R_HIP_ROLL      = 13,
    R_KNEE          = 14,
    R_ANKLE_PITCH   = 15,
    R_ANKLE_ROLL    = 16
};

// ── FlexPWM Servo Driver ──
class FlexPwmServoDriver {
public:
    // ── Init: configure all 17 PWM channels ──
    void init() {
        // Set PWM resolution (16-bit for maximum precision)
        analogWriteResolution(PwmServo::PWM_RESOLUTION);

        // Configure each servo pin
        for (int j = 0; j < PwmServo::NUM_JOINTS; j++) {
            uint8_t pin = PwmServo::SERVO_PINS[j];
            pinMode(pin, OUTPUT);
            analogWriteFrequency(pin, PwmServo::PWM_FREQ_HZ);

            // Set to center position
            target_cdeg_[j] = 0;
            current_duty_[j] = angleToDuty(0);
            enabled_[j] = false; // Disabled until explicitly enabled

            analogWrite(pin, 0); // Off initially
        }

        Serial.printf("[T2:FlexPWM] Init: %d channels @ %.0fHz, %d-bit\n",
                      PwmServo::NUM_JOINTS, PwmServo::PWM_FREQ_HZ,
                      PwmServo::PWM_RESOLUTION);
    }

    // ── Set target angle for a joint (centidegrees) ──
    void setAngle(uint8_t joint, int16_t angle_cdeg) {
        if (joint >= PwmServo::NUM_JOINTS) return;

        // Clamp to valid range
        if (angle_cdeg < PwmServo::ANGLE_MIN_CDEG) angle_cdeg = PwmServo::ANGLE_MIN_CDEG;
        if (angle_cdeg > PwmServo::ANGLE_MAX_CDEG) angle_cdeg = PwmServo::ANGLE_MAX_CDEG;

        target_cdeg_[joint] = angle_cdeg;
    }

    // ── Apply all target angles to PWM hardware ──
    void applyAll() {
        for (int j = 0; j < PwmServo::NUM_JOINTS; j++) {
            if (enabled_[j]) {
                uint16_t duty = angleToDuty(target_cdeg_[j]);
                current_duty_[j] = duty;
                analogWrite(PwmServo::SERVO_PINS[j], duty);
            }
        }
    }

    // ── Apply single joint ──
    void applySingle(uint8_t joint) {
        if (joint >= PwmServo::NUM_JOINTS) return;

        if (enabled_[joint]) {
            uint16_t duty = angleToDuty(target_cdeg_[joint]);
            current_duty_[joint] = duty;
            analogWrite(PwmServo::SERVO_PINS[joint], duty);
        }
    }

    // ── Enable / Disable ──
    void setEnabled(uint8_t joint, bool en) {
        if (joint >= PwmServo::NUM_JOINTS) return;

        enabled_[joint] = en;
        if (!en) {
            analogWrite(PwmServo::SERVO_PINS[joint], 0); // PWM off
        }
    }

    void setEnabledMask(uint32_t mask) {
        for (int j = 0; j < PwmServo::NUM_JOINTS; j++) {
            setEnabled(j, (mask >> j) & 1);
        }
    }

    void enableAll() {
        for (int j = 0; j < PwmServo::NUM_JOINTS; j++) {
            enabled_[j] = true;
        }
    }

    void disableAll() {
        for (int j = 0; j < PwmServo::NUM_JOINTS; j++) {
            enabled_[j] = false;
            analogWrite(PwmServo::SERVO_PINS[j], 0);
        }
        Serial.println("[T2:FlexPWM] All servos DISABLED (PWM off)");
    }

    // ── Move all to center (0°) ──
    void centerAll() {
        for (int j = 0; j < PwmServo::NUM_JOINTS; j++) {
            target_cdeg_[j] = 0;
        }
        applyAll();
        Serial.println("[T2:FlexPWM] All servos centered (0°)");
    }

    // ── Queries ──
    int16_t getTargetAngle(uint8_t joint) const {
        return (joint < PwmServo::NUM_JOINTS) ? target_cdeg_[joint] : 0;
    }

    uint16_t getCurrentDuty(uint8_t joint) const {
        return (joint < PwmServo::NUM_JOINTS) ? current_duty_[joint] : 0;
    }

    bool isEnabled(uint8_t joint) const {
        return (joint < PwmServo::NUM_JOINTS) ? enabled_[joint] : false;
    }

    uint32_t getEnabledMask() const {
        uint32_t mask = 0;
        for (int j = 0; j < PwmServo::NUM_JOINTS; j++) {
            if (enabled_[j]) mask |= (1u << j);
        }
        return mask;
    }

    // ── Conversion Utilities (public for testing) ──

    // Centidegrees → PWM duty (16-bit)
    static uint16_t angleToDuty(int16_t angle_cdeg) {
        // Clamp
        if (angle_cdeg < PwmServo::ANGLE_MIN_CDEG) angle_cdeg = PwmServo::ANGLE_MIN_CDEG;
        if (angle_cdeg > PwmServo::ANGLE_MAX_CDEG) angle_cdeg = PwmServo::ANGLE_MAX_CDEG;

        // angle_cdeg → pulse_us
        // normalized = (angle_cdeg - ANGLE_MIN) / ANGLE_RANGE = [0.0, 1.0]
        // pulse_us = PULSE_MIN + normalized * PULSE_RANGE
        uint32_t pulse_us = PwmServo::PULSE_MIN_US +
            (uint32_t)(angle_cdeg - PwmServo::ANGLE_MIN_CDEG) *
            PwmServo::PULSE_RANGE_US / PwmServo::ANGLE_RANGE_CDEG;

        // pulse_us → duty16
        // duty = pulse_us / PWM_PERIOD_US * PWM_MAX_VALUE
        uint32_t duty = (uint32_t)pulse_us * PwmServo::PWM_MAX_VALUE / PwmServo::PWM_PERIOD_US;

        return (uint16_t)duty;
    }

    // PWM duty (16-bit) → centidegrees (inverse)
    static int16_t dutyToAngle(uint16_t duty) {
        uint32_t pulse_us = (uint32_t)duty * PwmServo::PWM_PERIOD_US / PwmServo::PWM_MAX_VALUE;

        if (pulse_us < PwmServo::PULSE_MIN_US) pulse_us = PwmServo::PULSE_MIN_US;
        if (pulse_us > PwmServo::PULSE_MAX_US) pulse_us = PwmServo::PULSE_MAX_US;

        int32_t angle_cdeg = PwmServo::ANGLE_MIN_CDEG +
            (int32_t)(pulse_us - PwmServo::PULSE_MIN_US) *
            PwmServo::ANGLE_RANGE_CDEG / PwmServo::PULSE_RANGE_US;

        return (int16_t)angle_cdeg;
    }

private:
    int16_t  target_cdeg_[PwmServo::NUM_JOINTS];  // Target angle (centidegrees)
    uint16_t current_duty_[PwmServo::NUM_JOINTS];  // Current PWM duty value
    bool     enabled_[PwmServo::NUM_JOINTS];        // Per-joint enable flag
};
