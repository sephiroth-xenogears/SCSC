// ============================================================================
// Phase 7: Teensy 4.1 Firmware — UART Host Interface (T1)
// Communication with Jetson AGX Orin via Serial1
// Binary packet protocol: [SYNC0][SYNC1][CMD][SEQ][LEN][PAYLOAD][CHECKSUM]
// Ref: SCSC-P7-ARCH-001 §3.2 T1 (PWM revision)
// ============================================================================
#pragma once

#include <Arduino.h>

// ── Packet Protocol ──
namespace Proto {
    static constexpr uint8_t SYNC_0 = 0xAA;
    static constexpr uint8_t SYNC_1 = 0x55;
    static constexpr int NUM_JOINTS  = 17;
    static constexpr int MAX_PAYLOAD = 64;

    // Header: SYNC(2) + CMD(1) + SEQ(1) + LEN(1) = 5 bytes
    // Max frame: 5 + MAX_PAYLOAD + CHECKSUM(1) = 70 bytes
    static constexpr int MAX_FRAME = 5 + MAX_PAYLOAD + 1;

    // ── Commands (Jetson → Teensy) ──
    static constexpr uint8_t CMD_SET_ANGLES        = 0x01; // 17 × int16 (34B)
    static constexpr uint8_t CMD_SET_ANGLES_SPARSE = 0x02; // count(1) + N×[id(1)+angle(2)]
    static constexpr uint8_t CMD_QUERY_STATUS      = 0x03; // (0B)
    static constexpr uint8_t CMD_SET_ENABLE        = 0x10; // bitmask(4B)
    static constexpr uint8_t CMD_HEARTBEAT         = 0x0E; // timestamp(4B)
    static constexpr uint8_t CMD_EMERGENCY_STOP    = 0x0F; // (0B)

    // ── Responses (Teensy → Jetson) ──
    static constexpr uint8_t RSP_ACK         = 0x81; // echo_cmd(1B)
    static constexpr uint8_t RSP_STATUS      = 0x83; // safety(1)+fault(4)+17×angle(34)+uptime(4) = 43B
    static constexpr uint8_t RSP_HB_ACK      = 0x8E; // echo_ts(4)+uptime(4) = 8B
    static constexpr uint8_t RSP_ESTOP_ACK   = 0x8F; // (0B)
    static constexpr uint8_t RSP_NACK        = 0xFF; // error_code(1B)

    struct Packet {
        uint8_t sync[2];
        uint8_t cmd;
        uint8_t seq;
        uint8_t len;          // Payload length in bytes
        uint8_t payload[MAX_PAYLOAD];
        uint8_t checksum;
    };
}

// ── Callback Types ──
using SetAnglesCallback = void (*)(const int16_t* angles, uint8_t count);
using SetAnglesSparseCallback = void (*)(const uint8_t* joint_ids, const int16_t* angles, uint8_t count);
using SetEnableCallback = void (*)(uint32_t mask);
using EmergencyStopCallback = void (*)();
using HeartbeatCallback = void (*)(uint32_t timestamp);
using QueryStatusCallback = void (*)();

// ── UartHost ──
class UartHost {
public:
    // ── Init ──
    bool init(HardwareSerial& serial, uint32_t baudrate) {
        serial_ = &serial;
        serial_->begin(baudrate);

        rx_state_ = RX_WAIT_SYNC_0;
        rx_idx_ = 0;
        tx_seq_ = 0;
        rx_error_count_ = 0;
        rx_good_count_ = 0;

        cb_set_angles_ = nullptr;
        cb_set_angles_sparse_ = nullptr;
        cb_set_enable_ = nullptr;
        cb_estop_ = nullptr;
        cb_heartbeat_ = nullptr;
        cb_query_status_ = nullptr;

        Serial.printf("[T1:UartHost] Init @ %lu baud\n", baudrate);
        return true;
    }

    // ── Register Callbacks ──
    void onSetAngles(SetAnglesCallback cb)                { cb_set_angles_ = cb; }
    void onSetAnglesSparse(SetAnglesSparseCallback cb)    { cb_set_angles_sparse_ = cb; }
    void onSetEnable(SetEnableCallback cb)                { cb_set_enable_ = cb; }
    void onEmergencyStop(EmergencyStopCallback cb)        { cb_estop_ = cb; }
    void onHeartbeat(HeartbeatCallback cb)                { cb_heartbeat_ = cb; }
    void onQueryStatus(QueryStatusCallback cb)            { cb_query_status_ = cb; }

    // ── Process incoming bytes (call from main loop) ──
    void update() {
        while (serial_->available()) {
            processByte(serial_->read());
        }
    }

    // ── Send ACK ──
    void sendAck(uint8_t echo_cmd) {
        uint8_t payload[1] = { echo_cmd };
        sendPacket(Proto::RSP_ACK, payload, 1);
    }

    // ── Send NACK ──
    void sendNack(uint8_t error_code) {
        uint8_t payload[1] = { error_code };
        sendPacket(Proto::RSP_NACK, payload, 1);
    }

    // ── Send Heartbeat ACK ──
    void sendHeartbeatAck(uint32_t echo_ts, uint32_t uptime_ms) {
        uint8_t payload[8];
        encodeU32(payload, 0, echo_ts);
        encodeU32(payload, 4, uptime_ms);
        sendPacket(Proto::RSP_HB_ACK, payload, 8);
    }

    // ── Send Emergency Stop ACK ──
    void sendEstopAck() {
        sendPacket(Proto::RSP_ESTOP_ACK, nullptr, 0);
    }

    // ── Send Status Report ──
    // Payload: safety_level(1) + fault_mask(4) + 17×angle_cdeg(34) + uptime_ms(4) = 43B
    void sendStatus(uint8_t safety_level, uint32_t fault_mask,
                    const int16_t* angles, uint32_t uptime_ms) {
        uint8_t payload[43];
        payload[0] = safety_level;
        encodeU32(payload, 1, fault_mask);

        for (int j = 0; j < Proto::NUM_JOINTS; j++) {
            encodeI16(payload, 5 + j * 2, angles[j]);
        }

        encodeU32(payload, 39, uptime_ms);
        sendPacket(Proto::RSP_STATUS, payload, 43);
    }

    // ── Statistics ──
    uint32_t getRxGoodCount() const { return rx_good_count_; }
    uint32_t getRxErrorCount() const { return rx_error_count_; }

private:
    // ── RX State Machine ──
    enum RxState {
        RX_WAIT_SYNC_0,
        RX_WAIT_SYNC_1,
        RX_CMD,
        RX_SEQ,
        RX_LEN,
        RX_PAYLOAD,
        RX_CHECKSUM
    };

    void processByte(uint8_t b) {
        switch (rx_state_) {
            case RX_WAIT_SYNC_0:
                if (b == Proto::SYNC_0) {
                    rx_pkt_.sync[0] = b;
                    rx_state_ = RX_WAIT_SYNC_1;
                }
                break;

            case RX_WAIT_SYNC_1:
                if (b == Proto::SYNC_1) {
                    rx_pkt_.sync[1] = b;
                    rx_state_ = RX_CMD;
                } else {
                    rx_state_ = RX_WAIT_SYNC_0;
                }
                break;

            case RX_CMD:
                rx_pkt_.cmd = b;
                rx_state_ = RX_SEQ;
                break;

            case RX_SEQ:
                rx_pkt_.seq = b;
                rx_state_ = RX_LEN;
                break;

            case RX_LEN:
                rx_pkt_.len = b;
                if (b > Proto::MAX_PAYLOAD) {
                    rx_error_count_++;
                    rx_state_ = RX_WAIT_SYNC_0;
                } else if (b == 0) {
                    rx_state_ = RX_CHECKSUM;
                } else {
                    rx_idx_ = 0;
                    rx_state_ = RX_PAYLOAD;
                }
                break;

            case RX_PAYLOAD:
                rx_pkt_.payload[rx_idx_++] = b;
                if (rx_idx_ >= rx_pkt_.len) {
                    rx_state_ = RX_CHECKSUM;
                }
                break;

            case RX_CHECKSUM:
                rx_pkt_.checksum = b;
                processPacket();
                rx_state_ = RX_WAIT_SYNC_0;
                break;
        }
    }

    // ── Packet Processing ──
    void processPacket() {
        uint8_t calc_cs = computeChecksum(rx_pkt_);
        if (calc_cs != rx_pkt_.checksum) {
            rx_error_count_++;
            return;
        }

        rx_good_count_++;

        switch (rx_pkt_.cmd) {
            case Proto::CMD_SET_ANGLES:
                handleSetAngles();
                break;
            case Proto::CMD_SET_ANGLES_SPARSE:
                handleSetAnglesSparse();
                break;
            case Proto::CMD_QUERY_STATUS:
                handleQueryStatus();
                break;
            case Proto::CMD_SET_ENABLE:
                handleSetEnable();
                break;
            case Proto::CMD_HEARTBEAT:
                handleHeartbeat();
                break;
            case Proto::CMD_EMERGENCY_STOP:
                handleEmergencyStop();
                break;
            default:
                sendNack(0x01); // Unknown command
                break;
        }
    }

    // ── Command Handlers ──

    void handleSetAngles() {
        // Payload: 17 × int16_t (34 bytes)
        if (rx_pkt_.len != Proto::NUM_JOINTS * 2) {
            sendNack(0x02); // Length mismatch
            return;
        }

        int16_t angles[Proto::NUM_JOINTS];
        for (int j = 0; j < Proto::NUM_JOINTS; j++) {
            angles[j] = decodeI16(rx_pkt_.payload, j * 2);
        }

        if (cb_set_angles_) cb_set_angles_(angles, Proto::NUM_JOINTS);
        sendAck(Proto::CMD_SET_ANGLES);
    }

    void handleSetAnglesSparse() {
        // Payload: count(1) + N × [joint_id(1) + angle(2)]
        if (rx_pkt_.len < 1) {
            sendNack(0x02);
            return;
        }

        uint8_t count = rx_pkt_.payload[0];
        if (rx_pkt_.len != 1 + count * 3) {
            sendNack(0x02);
            return;
        }

        uint8_t ids[Proto::NUM_JOINTS];
        int16_t angles[Proto::NUM_JOINTS];
        for (uint8_t i = 0; i < count && i < Proto::NUM_JOINTS; i++) {
            int offset = 1 + i * 3;
            ids[i] = rx_pkt_.payload[offset];
            angles[i] = decodeI16(rx_pkt_.payload, offset + 1);
        }

        if (cb_set_angles_sparse_) cb_set_angles_sparse_(ids, angles, count);
        sendAck(Proto::CMD_SET_ANGLES_SPARSE);
    }

    void handleQueryStatus() {
        if (cb_query_status_) cb_query_status_();
        // Status response is sent by the callback
    }

    void handleSetEnable() {
        if (rx_pkt_.len != 4) {
            sendNack(0x02);
            return;
        }

        uint32_t mask = decodeU32(rx_pkt_.payload, 0);
        if (cb_set_enable_) cb_set_enable_(mask);
        sendAck(Proto::CMD_SET_ENABLE);
    }

    void handleHeartbeat() {
        if (rx_pkt_.len != 4) {
            sendNack(0x02);
            return;
        }

        uint32_t ts = decodeU32(rx_pkt_.payload, 0);
        if (cb_heartbeat_) cb_heartbeat_(ts);
        // HB ACK is sent by the callback
    }

    void handleEmergencyStop() {
        if (cb_estop_) cb_estop_();
        sendEstopAck();
    }

    // ── Packet Transmission ──
    void sendPacket(uint8_t cmd, const uint8_t* payload, uint8_t len) {
        Proto::Packet pkt;
        pkt.sync[0] = Proto::SYNC_0;
        pkt.sync[1] = Proto::SYNC_1;
        pkt.cmd = cmd;
        pkt.seq = tx_seq_++;
        pkt.len = len;

        if (payload && len > 0) {
            memcpy(pkt.payload, payload, len);
        }

        pkt.checksum = computeChecksum(pkt);

        // Write frame: sync(2) + cmd(1) + seq(1) + len(1) + payload(len) + cksum(1)
        uint8_t frame[Proto::MAX_FRAME];
        frame[0] = pkt.sync[0];
        frame[1] = pkt.sync[1];
        frame[2] = pkt.cmd;
        frame[3] = pkt.seq;
        frame[4] = pkt.len;
        if (len > 0) memcpy(&frame[5], pkt.payload, len);
        frame[5 + len] = pkt.checksum;

        serial_->write(frame, 6 + len);
    }

    // ── Checksum: XOR of all bytes from sync[0] through payload[len-1] ──
    static uint8_t computeChecksum(const Proto::Packet& pkt) {
        uint8_t cs = pkt.sync[0] ^ pkt.sync[1];
        cs ^= pkt.cmd;
        cs ^= pkt.seq;
        cs ^= pkt.len;
        for (uint8_t i = 0; i < pkt.len; i++) {
            cs ^= pkt.payload[i];
        }
        return cs;
    }

    // ── Encoding / Decoding Helpers ──
    static void encodeU32(uint8_t* buf, int offset, uint32_t val) {
        buf[offset + 0] = (val >> 24) & 0xFF;
        buf[offset + 1] = (val >> 16) & 0xFF;
        buf[offset + 2] = (val >>  8) & 0xFF;
        buf[offset + 3] = val & 0xFF;
    }

    static void encodeI16(uint8_t* buf, int offset, int16_t val) {
        buf[offset + 0] = (val >> 8) & 0xFF;
        buf[offset + 1] = val & 0xFF;
    }

    static uint32_t decodeU32(const uint8_t* buf, int offset) {
        return ((uint32_t)buf[offset] << 24) |
               ((uint32_t)buf[offset + 1] << 16) |
               ((uint32_t)buf[offset + 2] << 8) |
               buf[offset + 3];
    }

    static int16_t decodeI16(const uint8_t* buf, int offset) {
        return (int16_t)((buf[offset] << 8) | buf[offset + 1]);
    }

    // ── Member Variables ──
    HardwareSerial* serial_ = nullptr;
    RxState rx_state_ = RX_WAIT_SYNC_0;
    Proto::Packet rx_pkt_;
    int rx_idx_ = 0;
    uint8_t tx_seq_ = 0;

    uint32_t rx_good_count_ = 0;
    uint32_t rx_error_count_ = 0;

    SetAnglesCallback cb_set_angles_;
    SetAnglesSparseCallback cb_set_angles_sparse_;
    SetEnableCallback cb_set_enable_;
    EmergencyStopCallback cb_estop_;
    HeartbeatCallback cb_heartbeat_;
    QueryStatusCallback cb_query_status_;
};
