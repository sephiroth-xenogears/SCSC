// ============================================================================
// Phase 7: Teensy 4.1 Firmware — Main (T5)
// KRS-4024HV (Red Version) PWM control + Jetson UART communication
// 17-axis Manoi PF01 humanoid — Active Inference drive layer
// Ref: SCSC-P7-ARCH-001 §3.2 (PWM revision)
// ============================================================================

#include <Arduino.h>

#include "uart_host.h"    // T1: Jetson UART protocol
#include "pwm_servo.h"    // T2: FlexPWM servo driver
#include "servo_sched.h"  // T3: Servo manager (joint limits)
#include "safety_wd.h"    // T4: Safety watchdog

// ============================================================================
// Global Configuration
// ============================================================================
static constexpr int      NUM_JOINTS      = 17;
static constexpr uint32_t JETSON_BAUDRATE  = 1250000; // 1.25 Mbaud via Serial1
static constexpr uint32_t STATUS_INTERVAL_MS = 100;   // 10 Hz status reports

// ── Pin Assignments (Teensy 4.1) ──
// Serial1 (pins 0/1): Jetson communication
// PWM pins 2-13, 22-25, 28: 17 servos (see pwm_servo.h)
// Pins 14-17 (A0-A3): FSR analog input (Phase 8 reserved)
// Pins 18-19 (SDA/SCL): IMU I2C (Phase 8 reserved)

// ── Module Instances ──
UartHost           g_uart;
FlexPwmServoDriver g_pwm;
ServoManager       g_servo;
SafetyWatchdog     g_safety;
SafetyConfig       g_safety_config;

// ── Timing ──
uint32_t g_loop_count = 0;
uint32_t g_last_status_ms = 0;

// ── Forward Declarations ──
void onSetAngles(const int16_t* angles, uint8_t count);
void onSetAnglesSparse(const uint8_t* ids, const int16_t* angles, uint8_t count);
void onSetEnable(uint32_t mask);
void onEmergencyStop();
void onHeartbeat(uint32_t timestamp);
void onQueryStatus();
void sendStatusReport();
void printDebugStats();

// ============================================================================
// Setup
// ============================================================================
void setup() {
    // ── Debug Console (USB Serial) ──
    Serial.begin(115200);
    delay(500);
    Serial.println("================================================");
    Serial.println("  SCSC Phase 7: Teensy 4.1 Firmware (PWM)");
    Serial.println("  KRS-4024HV Red Version / 17-Axis Manoi PF01");
    Serial.println("  Build: " __DATE__ " " __TIME__);
    Serial.println("================================================");

    // ── LED ──
    pinMode(LED_BUILTIN, OUTPUT);
    digitalWrite(LED_BUILTIN, LOW);

    // ── T1: UART Host (Jetson AGX Orin) ──
    Serial.println("[T5] Init T1:UartHost...");
    if (!g_uart.init(Serial1, JETSON_BAUDRATE)) {
        Serial.println("[T5] FATAL: UartHost init failed!");
        while (1) { digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN)); delay(100); }
    }

    // Register UART callbacks
    g_uart.onSetAngles(onSetAngles);
    g_uart.onSetAnglesSparse(onSetAnglesSparse);
    g_uart.onSetEnable(onSetEnable);
    g_uart.onEmergencyStop(onEmergencyStop);
    g_uart.onHeartbeat(onHeartbeat);
    g_uart.onQueryStatus(onQueryStatus);

    // ── T2: FlexPWM Servo Driver ──
    Serial.println("[T5] Init T2:FlexPWM...");
    g_pwm.init();

    // ── T3: Servo Manager ──
    Serial.println("[T5] Init T3:ServoManager...");
    g_servo.init(&g_pwm);

    // ── T4: Safety Watchdog ──
    Serial.println("[T5] Init T4:SafetyWatchdog...");
    g_safety_config.heartbeat_timeout_ms = 200;
    g_safety_config.hold_on_comm_fault = true;
    g_safety.init(&g_safety_config, &g_servo);

    // ── Initial State: center all servos, but keep disabled ──
    g_servo.centerAll();
    Serial.println("[T5] Servos centered (PWM disabled until SET_ENABLE)");

    // ── Ready ──
    g_last_status_ms = millis();
    Serial.println("[T5] ================================================");
    Serial.println("[T5] Firmware ready. Waiting for Jetson...");
    Serial.println("[T5] ================================================");

    // Ready blink
    for (int i = 0; i < 3; i++) {
        digitalWrite(LED_BUILTIN, HIGH); delay(80);
        digitalWrite(LED_BUILTIN, LOW);  delay(80);
    }
}

// ============================================================================
// Main Loop
// ============================================================================
void loop() {
    // ── T1: Process UART from Jetson ──
    g_uart.update();

    // ── T4: Safety watchdog ──
    g_safety.update();

    // ── T3: Apply servo positions to PWM hardware ──
    // PWM is 50Hz (20ms period), so applying every loop iteration is fine
    // (hardware only changes on next PWM cycle boundary)
    if (!g_safety.isEmergency()) {
        g_servo.applyAll();
    }

    g_loop_count++;

    // ── Periodic status report to Jetson (10 Hz) ──
    uint32_t now = millis();
    if (now - g_last_status_ms >= STATUS_INTERVAL_MS) {
        g_last_status_ms = now;
        sendStatusReport();
    }

    // ── Debug output (1 Hz) ──
    if ((g_loop_count % 50000) == 0) {
        printDebugStats();
    }
}

// ============================================================================
// UART Callbacks
// ============================================================================

// ── CMD_SET_ANGLES (0x01): Set all 17 angles ──
void onSetAngles(const int16_t* angles, uint8_t count) {
    if (g_safety.isEmergency()) return;
    g_servo.setAllAngles(angles);
}

// ── CMD_SET_ANGLES_SPARSE (0x02): Set specific joints ──
void onSetAnglesSparse(const uint8_t* ids, const int16_t* angles, uint8_t count) {
    if (g_safety.isEmergency()) return;
    for (uint8_t i = 0; i < count; i++) {
        g_servo.setAngle(ids[i], angles[i]);
    }
}

// ── CMD_SET_ENABLE (0x10): Enable/disable servo bitmask ──
void onSetEnable(uint32_t mask) {
    if (g_safety.isEmergency()) return;
    g_servo.setEnabledMask(mask);
    Serial.printf("[T5] Servo enable mask: 0x%05lX\n", mask);
}

// ── CMD_EMERGENCY_STOP (0x0F) ──
void onEmergencyStop() {
    g_safety.triggerEmergencyStop();
}

// ── CMD_HEARTBEAT (0x0E) ──
void onHeartbeat(uint32_t timestamp) {
    g_safety.notifyHeartbeat();
    g_uart.sendHeartbeatAck(timestamp, millis());
}

// ── CMD_QUERY_STATUS (0x03) ──
void onQueryStatus() {
    sendStatusReport();
}

// ============================================================================
// Status Report
// ============================================================================
void sendStatusReport() {
    int16_t angles[NUM_JOINTS];
    g_servo.getAllAngles(angles);

    g_uart.sendStatus(
        (uint8_t)g_safety.getLevel(),
        g_safety.getFaultMask(),
        angles,
        millis()
    );
}

// ============================================================================
// Debug Output
// ============================================================================
void printDebugStats() {
    Serial.printf("[T5] loops=%lu | updates=%lu | RX:ok=%lu err=%lu | Safety=%d | HB=%lu ms\n",
                  g_loop_count,
                  g_servo.getUpdateCount(),
                  g_uart.getRxGoodCount(),
                  g_uart.getRxErrorCount(),
                  (int)g_safety.getLevel(),
                  g_safety.getTimeSinceHeartbeat());

    // Sample 3 joints
    for (int j = 0; j < 3; j++) {
        Serial.printf("  J%d: %+6d cdeg  en=%d  pin=%d\n",
                      j,
                      g_servo.getTargetAngle(j),
                      g_servo.isEnabled(j),
                      PwmServo::SERVO_PINS[j]);
    }
}
