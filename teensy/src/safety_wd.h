// ============================================================================
// Phase 7: Teensy 4.1 Firmware — Safety Watchdog (T4)
// Heartbeat timeout, emergency stop, angle limit monitoring
// Ref: SCSC-P7-ARCH-001 §3.2 T4 (PWM revision)
// ============================================================================
#pragma once

#include <Arduino.h>
#include "servo_sched.h" // ServoManager

// ── Safety Configuration ──
struct SafetyConfig {
    uint32_t heartbeat_timeout_ms = 200;  // 200ms timeout (5Hz min from Jetson)
    bool     hold_on_comm_fault   = true; // Hold last position on comm fault (vs disable)
};

// ── Safety Level ──
enum class SafetyLevel : uint8_t {
    NORMAL     = 0,  // All OK
    COMM_FAULT = 2,  // Heartbeat timeout — hold position or disable
    EMERGENCY  = 3   // Emergency stop — all PWM off
};

// ── Safety Watchdog ──
class SafetyWatchdog {
public:
    // ── Init ──
    void init(const SafetyConfig* config, ServoManager* manager) {
        config_ = config;
        manager_ = manager;

        last_heartbeat_ms_ = millis();
        emergency_active_ = false;
        level_ = SafetyLevel::NORMAL;
        fault_mask_ = 0;

        Serial.println("[T4:SafetyWD] Init complete");
    }

    // ── Update: call from main loop ──
    void update() {
        if (!config_ || !manager_) return;

        // Skip checks during emergency
        if (emergency_active_) return;

        // Check heartbeat timeout
        uint32_t elapsed = millis() - last_heartbeat_ms_;
        if (elapsed > config_->heartbeat_timeout_ms) {
            triggerCommFault();
        }
    }

    // ── Heartbeat received ──
    void notifyHeartbeat() {
        last_heartbeat_ms_ = millis();

        // Recover from comm fault
        if (level_ == SafetyLevel::COMM_FAULT) {
            level_ = SafetyLevel::NORMAL;
            fault_mask_ = 0;
            digitalWrite(LED_BUILTIN, LOW);
            Serial.println("[T4:SafetyWD] Heartbeat restored");
        }
    }

    // ── Emergency Stop ──
    void triggerEmergencyStop() {
        if (emergency_active_) return;

        emergency_active_ = true;
        level_ = SafetyLevel::EMERGENCY;

        // Disable all PWM outputs
        manager_->disableAll();

        digitalWrite(LED_BUILTIN, HIGH);
        Serial.println("[T4:SafetyWD] *** EMERGENCY STOP — All PWM OFF ***");
    }

    // ── Reset Emergency (requires explicit Jetson command) ──
    void resetEmergency() {
        if (!emergency_active_) return;

        emergency_active_ = false;
        level_ = SafetyLevel::NORMAL;
        fault_mask_ = 0;
        last_heartbeat_ms_ = millis();

        digitalWrite(LED_BUILTIN, LOW);
        Serial.println("[T4:SafetyWD] Emergency RESET");
    }

    // ── Queries ──
    SafetyLevel getLevel() const { return level_; }
    bool isEmergency() const { return emergency_active_; }
    uint32_t getFaultMask() const { return fault_mask_; }

    uint32_t getTimeSinceHeartbeat() const {
        return millis() - last_heartbeat_ms_;
    }

private:
    void triggerCommFault() {
        if (level_ == SafetyLevel::COMM_FAULT) return; // Already triggered

        level_ = SafetyLevel::COMM_FAULT;

        if (!config_->hold_on_comm_fault) {
            // Disable servos on comm loss
            manager_->disableAll();
            Serial.println("[T4:SafetyWD] COMM_FAULT: Servos DISABLED");
        } else {
            // Hold last position (PWM keeps running with last values)
            Serial.println("[T4:SafetyWD] COMM_FAULT: Holding last position");
        }

        digitalWrite(LED_BUILTIN, HIGH);
        Serial.printf("[T4:SafetyWD] No heartbeat for %lu ms\n",
                     millis() - last_heartbeat_ms_);
    }

    const SafetyConfig* config_ = nullptr;
    ServoManager* manager_ = nullptr;

    uint32_t last_heartbeat_ms_ = 0;
    bool emergency_active_ = false;
    SafetyLevel level_ = SafetyLevel::NORMAL;
    uint32_t fault_mask_ = 0;
};
