#pragma once
// ============================================================================
// Phase 7: Full-Body 17-Axis Active Inference — Servo Commander (M8)
// UART packet construction matching Teensy wire protocol:
//   [SYNC0=0xAA][SYNC1=0x55][CMD][SEQ][LEN][PAYLOAD...][CHECKSUM]
// Angles in int16_t centidegrees (0.01° units, ±13500)
// Includes loopback stub for testing without Teensy
// Ref: SCSC-P7-ARCH-001 §3.1 M8, §8
// ============================================================================

#include "../include/types.h"
#include "../include/body_config.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>

// Linux UART
#include <fcntl.h>
#include <termios.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <errno.h>

namespace phase7 {

// ── Custom baud rate support (avoids asm/termbits.h header conflict) ──
#ifndef BOTHER
#define BOTHER 0010000
#endif

class ServoCommander {
public:
    ~ServoCommander() { shutdown(); }

    // ── Init (real UART or loopback stub) ──
    bool init(const std::string& uart_path, int baudrate, bool loopback = false) {
        loopback_ = loopback;
        seq_num_ = 0;
        tx_count_ = 0;
        tx_error_count_ = 0;
        std::memset(tracked_pos_, 0, sizeof(tracked_pos_));

        if (loopback_) {
            std::printf("[M8:ServoCmd] Loopback stub mode (no UART)\n");
            std::memset(&sim_feedback_, 0, sizeof(sim_feedback_));
            sim_feedback_.valid_mask = 0xFFFFFFFF;
            return true;
        }

        // ── Real UART open ──
        fd_ = ::open(uart_path.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);
        if (fd_ < 0) {
            std::printf("[M8:ServoCmd] UART open failed: %s (%s)\n",
                        uart_path.c_str(), strerror(errno));
            return false;
        }

        // Clear O_NONBLOCK after open (blocking writes, non-blocking reads via VMIN/VTIME)
        int flags = fcntl(fd_, F_GETFL, 0);
        fcntl(fd_, F_SETFL, flags & ~O_NONBLOCK);

        if (!configureUart(baudrate)) {
            std::printf("[M8:ServoCmd] UART configure failed\n");
            ::close(fd_);
            fd_ = -1;
            return false;
        }

        tcflush(fd_, TCIOFLUSH);

        uart_path_ = uart_path;
        baudrate_ = baudrate;
        std::memset(&sim_feedback_, 0, sizeof(sim_feedback_));

        std::printf("[M8:ServoCmd] UART opened: %s @ %d baud (fd=%d)\n",
                    uart_path.c_str(), baudrate, fd_);
        return true;
    }

    // ── Shutdown ──
    void shutdown() {
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
            std::printf("[M8:ServoCmd] UART closed\n");
        }
    }

    // ── Build and send CMD_SET_ANGLES (0x01) ──
    // Teensy expects: LEN=34, payload = 17 × int16_t centidegrees (big-endian)
    bool sendPositions(const action_t& action, const body_config_t& cfg) {
        // Integrate velocity → position (shared by loopback and real UART)
        constexpr float DT = 0.01f; // 100Hz
        for (int j = 0; j < NUM_JOINTS; j++) {
            tracked_pos_[j] += action.joint_cmd[j] * DT;
            tracked_pos_[j] = std::fmin(cfg.joint_max[j],
                              std::fmax(cfg.joint_min[j], tracked_pos_[j]));
        }

        if (loopback_) {
            return simulateResponse(action, cfg);
        }

        // Build payload: 17 × int16_t big-endian = 34 bytes
        uint8_t payload[NUM_JOINTS * 2];
        for (int j = 0; j < NUM_JOINTS; j++) {
            int16_t cdeg = radToCentideg(tracked_pos_[j]);
            payload[j * 2]     = (cdeg >> 8) & 0xFF;
            payload[j * 2 + 1] = cdeg & 0xFF;
        }

        return sendPacketWire(CMD_SET_ANGLES, payload, NUM_JOINTS * 2);
    }

    // ── Send heartbeat (CMD_HEARTBEAT 0x0E) ──
    bool sendHeartbeat(uint32_t timestamp) {
        uint8_t payload[4];
        payload[0] = (timestamp >> 24) & 0xFF;
        payload[1] = (timestamp >> 16) & 0xFF;
        payload[2] = (timestamp >> 8)  & 0xFF;
        payload[3] = timestamp & 0xFF;

        if (loopback_) return true;
        return sendPacketWire(CMD_HEARTBEAT, payload, 4);
    }

    // ── Send emergency stop (CMD_EMERGENCY_STOP 0x0F) ──
    bool sendEmergencyStop() {
        std::printf("[M8:ServoCmd] EMERGENCY STOP sent\n");
        if (loopback_) return true;
        return sendPacketWire(CMD_EMERGENCY_STOP, nullptr, 0);
    }

    // ── Drain incoming bytes from Teensy (call periodically to prevent buffer overflow) ──
    // TODO Phase 8+: parse RSP_STATUS (0x83) to populate real feedback
    void drainResponses() {
        if (fd_ < 0) return;
        uint8_t buf[256];
        while (::read(fd_, buf, sizeof(buf)) > 0) {}
    }

    // ── Get latest feedback ──
    servo_feedback_t getLatestFeedback() const {
        return sim_feedback_;
    }

    bool isLoopback() const { return loopback_; }
    uint32_t getTxCount() const { return tx_count_; }
    uint32_t getTxErrorCount() const { return tx_error_count_; }

private:
    // ── Protocol constants (must match teensy/src/uart_host.h) ──
    static constexpr uint8_t SYNC_0 = 0xAA;
    static constexpr uint8_t SYNC_1 = 0x55;
    static constexpr uint8_t CMD_SET_ANGLES     = 0x01;
    static constexpr uint8_t CMD_HEARTBEAT      = 0x0E;
    static constexpr uint8_t CMD_EMERGENCY_STOP = 0x0F;
    static constexpr int MAX_FRAME = 70;

    // ── Centidegree limits (must match teensy/src/pwm_servo.h) ──
    static constexpr int16_t CDEG_MIN = -13500;
    static constexpr int16_t CDEG_MAX =  13500;

    // ── UART configuration ──
    bool configureUart(int baudrate) {
        struct termios tio;
        if (tcgetattr(fd_, &tio) != 0) {
            std::printf("[M8:ServoCmd] tcgetattr: %s\n", strerror(errno));
            return false;
        }

        cfmakeraw(&tio);
        tio.c_cflag |= (CLOCAL | CREAD);
        tio.c_cflag &= ~(PARENB | CSTOPB | CRTSCTS);
        tio.c_cflag &= ~CSIZE;
        tio.c_cflag |= CS8;
        tio.c_cc[VMIN]  = 0;
        tio.c_cc[VTIME] = 0;

        // Try standard baud first
        speed_t speed = mapStandardBaud(baudrate);
        if (speed != 0) {
            cfsetispeed(&tio, speed);
            cfsetospeed(&tio, speed);
            if (tcsetattr(fd_, TCSANOW, &tio) != 0) {
                std::printf("[M8:ServoCmd] tcsetattr: %s\n", strerror(errno));
                return false;
            }
            std::printf("[M8:ServoCmd] Baud: %d (standard)\n", baudrate);
            return true;
        }

        // Non-standard baud: apply base config first, then override via termios2
        cfsetispeed(&tio, B115200);
        cfsetospeed(&tio, B115200);
        if (tcsetattr(fd_, TCSANOW, &tio) != 0) {
            std::printf("[M8:ServoCmd] tcsetattr: %s\n", strerror(errno));
            return false;
        }

        return setCustomBaud(baudrate);
    }

    // ── Custom baud rate via ioctl TCGETS2/TCSETS2 + BOTHER (Jetson Tegra) ──
    bool setCustomBaud(int baudrate) {
        // Local termios2 definition (avoids asm/termbits.h conflict with termios.h)
        struct termios2 {
            unsigned int c_iflag;
            unsigned int c_oflag;
            unsigned int c_cflag;
            unsigned int c_lflag;
            unsigned char c_line;
            unsigned char c_cc[19];
            unsigned int c_ispeed;
            unsigned int c_ospeed;
        };

        // ioctl numbers for aarch64 Linux:
        //   TCGETS2 = _IOR('T', 0x2A, struct termios2) = 0x802C542A
        //   TCSETS2 = _IOW('T', 0x2B, struct termios2) = 0x402C542B
        static constexpr unsigned long IOC_TCGETS2 = 0x802C542Au;
        static constexpr unsigned long IOC_TCSETS2 = 0x402C542Bu;

        struct termios2 tio2;
        if (ioctl(fd_, IOC_TCGETS2, &tio2) < 0) {
            std::printf("[M8:ServoCmd] TCGETS2 failed (%s), falling back to 1000000\n",
                        strerror(errno));
            return setFallbackBaud();
        }

        tio2.c_cflag &= ~CBAUD;
        tio2.c_cflag |= BOTHER;
        tio2.c_ispeed = baudrate;
        tio2.c_ospeed = baudrate;

        if (ioctl(fd_, IOC_TCSETS2, &tio2) < 0) {
            std::printf("[M8:ServoCmd] TCSETS2 failed (%s), falling back to 1000000\n",
                        strerror(errno));
            return setFallbackBaud();
        }

        std::printf("[M8:ServoCmd] Baud: %d (custom via BOTHER)\n", baudrate);
        return true;
    }

    // ── Fallback: use B1000000 if custom baud fails ──
    bool setFallbackBaud() {
        struct termios tio;
        tcgetattr(fd_, &tio);
        cfsetispeed(&tio, B1000000);
        cfsetospeed(&tio, B1000000);
        if (tcsetattr(fd_, TCSANOW, &tio) != 0) return false;
        std::printf("[M8:ServoCmd] WARNING: Using 1000000 baud as fallback\n");
        std::printf("[M8:ServoCmd] Teensy JETSON_BAUDRATE must also be 1000000!\n");
        return true;
    }

    static speed_t mapStandardBaud(int baud) {
        switch (baud) {
            case 9600:    return B9600;
            case 19200:   return B19200;
            case 38400:   return B38400;
            case 57600:   return B57600;
            case 115200:  return B115200;
            case 230400:  return B230400;
            case 460800:  return B460800;
            case 500000:  return B500000;
            case 576000:  return B576000;
            case 921600:  return B921600;
            case 1000000: return B1000000;
            case 1152000: return B1152000;
            case 1500000: return B1500000;
            case 2000000: return B2000000;
            default:      return 0;
        }
    }

    // ── Send raw packet (Teensy wire format) ──
    // [0xAA][0x55][CMD][SEQ][LEN][PAYLOAD...][CHECKSUM]
    // Checksum = XOR of all bytes from SYNC0 through last PAYLOAD byte
    bool sendPacketWire(uint8_t cmd, const uint8_t* payload, uint8_t len) {
        if (fd_ < 0) return false;

        uint8_t frame[MAX_FRAME];
        frame[0] = SYNC_0;
        frame[1] = SYNC_1;
        frame[2] = cmd;
        frame[3] = seq_num_++;
        frame[4] = len;

        if (payload && len > 0) {
            std::memcpy(&frame[5], payload, len);
        }

        uint8_t cs = 0;
        for (int i = 0; i < 5 + len; i++) cs ^= frame[i];
        frame[5 + len] = cs;

        int total = 6 + len;
        ssize_t written = ::write(fd_, frame, total);
        if (written != total) {
            tx_error_count_++;
            if (written < 0) {
                std::printf("[M8:ServoCmd] write: %s\n", strerror(errno));
            }
            return false;
        }

        tx_count_++;
        return true;
    }

    // ── Rad → centidegrees ──
    static int16_t radToCentideg(float rad) {
        float cdeg = rad * (18000.0f / M_PI); // rad × (180/π) × 100
        if (cdeg < (float)CDEG_MIN) cdeg = (float)CDEG_MIN;
        if (cdeg > (float)CDEG_MAX) cdeg = (float)CDEG_MAX;
        return (int16_t)cdeg;
    }

    // ── Loopback simulation (unchanged behavior) ──
    bool simulateResponse(const action_t& action, const body_config_t& cfg) {
        constexpr float SIM_DT = 0.01f;
        constexpr float SIM_TAU = 0.05f;
        float alpha = SIM_DT / (SIM_TAU + SIM_DT);

        for (int j = 0; j < NUM_JOINTS; j++) {
            float target = tracked_pos_[j];
            sim_feedback_.position[j] += alpha * (target - sim_feedback_.position[j]);
            sim_feedback_.position[j] = std::fmin(cfg.joint_max[j],
                                        std::fmax(cfg.joint_min[j],
                                                  sim_feedback_.position[j]));
            sim_feedback_.current[j] = std::fabs(action.joint_cmd[j]) * 100.0f;
        }

        auto now = std::chrono::steady_clock::now().time_since_epoch();
        sim_feedback_.timestamp_us = std::chrono::duration_cast<
            std::chrono::microseconds>(now).count();
        sim_feedback_.valid_mask = 0xFFFFFFFF;
        return true;
    }

    // ── Member Variables ──
    bool loopback_ = false;
    uint8_t seq_num_ = 0;
    std::string uart_path_;
    int baudrate_ = 0;
    int fd_ = -1;
    float tracked_pos_[NUM_JOINTS] = {};
    uint32_t tx_count_ = 0;
    uint32_t tx_error_count_ = 0;
    servo_feedback_t sim_feedback_;
};

} // namespace phase7
