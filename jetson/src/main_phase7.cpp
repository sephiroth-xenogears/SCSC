// ============================================================================
// Phase 7: Full-Body 17-Axis Active Inference — Main Program (M12)
// Platform: Jetson AGX Orin (sm_87) + Teensy 4.1 + Manoi PF01
// Multi-threaded: Capture, Vision, Inference, CommTx, CommRx, Safety, Log
// Ref: SCSC-P7-ARCH-001 §7, §12
// ============================================================================

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <csignal>
#include <atomic>
#include <thread>
#include <string>

#include "include/types.h"
#include "include/body_config.h"
#include "include/spsc_queue.h"
#include "src/camera.h"
#include "src/face_detect.h"
#include "src/observation.h"
#include "src/servo_cmd.h"
#include "src/safety.h"
#include "src/logger.h"
#include "cuda/aif_engine.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace phase7;

// ── Global flags ──
static std::atomic<bool> g_running{true};
static std::atomic<bool> g_emergency{false};
static void signalHandler(int) { g_running.store(false); }

// ── Inter-thread queues (§7.2) ──
static SPSCQueue<observation_t, 4>      q_observation;
static SPSCQueue<action_t, 4>           q_action;
static SPSCQueue<servo_feedback_t, 4>   q_servo_feedback;
static SPSCQueue<log_entry_t, 64>       q_log;

// ── Shared state (atomic-safe) ──
static std::atomic<int> g_safety_level{0};
static servo_feedback_t g_latest_feedback;  // updated by CommRx only

// ============================================================================
// Command-line arguments
// ============================================================================
struct Args {
    int    camera_id   = 0;
    std::string model_dir  = "models";
    std::string uart_path  = "/dev/ttyTHS1";
    int    baudrate    = 1250000;
    std::string log_dir    = "./logs";
    bool   loopback    = true;  // Default: stub until Teensy connected
    bool   headless    = true;
    int    max_steps   = 0;     // 0 = unlimited
};

static Args parseArgs(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--camera"   && i+1 < argc) a.camera_id = std::atoi(argv[++i]);
        else if (arg == "--models"  && i+1 < argc) a.model_dir = argv[++i];
        else if (arg == "--uart"    && i+1 < argc) a.uart_path = argv[++i];
        else if (arg == "--baudrate"&& i+1 < argc) a.baudrate  = std::atoi(argv[++i]);
        else if (arg == "--log-dir" && i+1 < argc) a.log_dir   = argv[++i];
        else if (arg == "--loopback")  a.loopback = true;
        else if (arg == "--real-servo") a.loopback = false;
        else if (arg == "--headless")  a.headless = true;
        else if (arg == "--steps"   && i+1 < argc) a.max_steps = std::atoi(argv[++i]);
        else if (arg == "--help") {
            std::printf(
                "Usage: phase7_main [options]\n"
                "  --camera ID       Camera device (default: 0)\n"
                "  --models DIR      DNN model dir (default: models)\n"
                "  --uart PATH       UART device (default: /dev/ttyTHS1)\n"
                "  --baudrate N      UART baud (default: 1250000)\n"
                "  --log-dir DIR     Log output (default: ./logs)\n"
                "  --loopback        Servo stub mode (default)\n"
                "  --real-servo      Use real UART to Teensy\n"
                "  --steps N         Max inference steps (0=unlimited)\n"
                "  --help            This help\n");
            std::exit(0);
        }
    }
    return a;
}

// ============================================================================
// T2: InferenceThread — 100Hz timer-driven AIF loop
// ============================================================================
static void inferenceThread(ActiveInferenceEngine& engine, SafetyMonitor& safety,
                            const Args& args)
{
    using Clock = std::chrono::steady_clock;
    const auto period = std::chrono::microseconds(10000); // 100Hz = 10ms
    int step = 0;

    std::printf("[T2:Inference] Started (100Hz)\n");

    while (g_running.load() && !g_emergency.load()) {
        auto t_start = Clock::now();

        // Consume latest observation
        observation_t obs;
        std::memset(&obs, 0, sizeof(obs));
        while (q_observation.tryPop(obs)) {} // drain, keep latest

        // AIF inference
        action_t raw_action = engine.step(obs);

        // Safety validation (M10)
        servo_feedback_t fb = g_latest_feedback;
        action_t safe_action = safety.validateAction(raw_action, fb);

        // Push to CommTx
        q_action.tryPush(safe_action);

        // Push to log
        log_entry_t entry;
        entry.timestamp_us = safe_action.timestamp_us;
        entry.estimated_state = engine.getEstimatedState();
        entry.action = safe_action;
        entry.observation = obs;
        entry.efe_total = safe_action.efe_value;
        for (int g = 0; g < NUM_GROUPS; g++)
            entry.efe_per_group[g] = safe_action.efe_components[g];
        entry.ess = engine.getESS();
        entry.safety = safety.getStatus();
        q_log.tryPush(entry);

        g_safety_level.store(safety.getStatus().level);
        step++;

        if (step % 100 == 0) {
            state_t st = engine.getEstimatedState();
            std::printf("[T2:%6d] EFE: %+.3f ESS: %.0f Safety: L%d | "
                        "pan_pos=%+.3f rad  pan_cmd=%+.3f rad/s\n",
                        step, safe_action.efe_value, engine.getESS(),
                        safety.getStatus().level,
                        st.data[state::PAN_THETA],            // estimated joint angle
                        safe_action.joint_cmd[HEAD_PAN]);      // velocity command
        }

        if (args.max_steps > 0 && step >= args.max_steps) {
            std::printf("[T2] Reached step limit (%d)\n", args.max_steps);
            g_running.store(false);
        }

        // Sleep remainder of period
        auto elapsed = Clock::now() - t_start;
        if (elapsed < period) {
            std::this_thread::sleep_for(period - elapsed);
        }
    }
    std::printf("[T2:Inference] Stopped (step=%d)\n", step);
}

// ============================================================================
// T1: VisionThread — 30fps face detection
// ============================================================================
static void visionThread(CameraCapture& camera, FaceDetector& detector,
                         ObservationBuilder& obs_builder, const Args& args)
{
    std::printf("[T1:Vision] Started (30fps)\n");
    cv::Mat frame;

    // Face snapshot saving
    int snap_count = 0;
    auto last_snap = std::chrono::steady_clock::now() - std::chrono::seconds(10);
    const auto snap_interval = std::chrono::seconds(1);  // max 1 save/sec
    const std::string snap_dir = args.log_dir + "/snapshots";
    system(("mkdir -p " + snap_dir).c_str());

    while (g_running.load() && !g_emergency.load()) {
        if (!camera.getLatestFrame(frame)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        FaceDetection face = detector.detect(frame);

        // Save snapshot on face detection (rate-limited)
        if (face.detected) {
            auto now = std::chrono::steady_clock::now();
            if (now - last_snap >= snap_interval) {
                last_snap = now;
                cv::Mat snap;
                frame.copyTo(snap);

                // Draw bounding box
                cv::rectangle(snap,
                    cv::Rect((int)face.x, (int)face.y, (int)face.w, (int)face.h),
                    cv::Scalar(0, 255, 0), 2);

                // Timestamp label
                char label[64];
                std::snprintf(label, sizeof(label), "conf=%.2f", face.confidence);
                cv::putText(snap, label,
                    cv::Point((int)face.x, (int)face.y - 8),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

                char fname[256];
                std::snprintf(fname, sizeof(fname), "%s/face_%04d.jpg",
                              snap_dir.c_str(), snap_count++);
                cv::imwrite(fname, snap);

                if (snap_count <= 3 || snap_count % 10 == 0) {
                    std::printf("[T1:Snap] Saved %s (%.0fx%.0f, conf=%.2f)\n",
                                fname, face.w, face.h, face.confidence);
                }
            }
        }

        // Build observation with latest servo feedback
        servo_feedback_t fb;
        while (q_servo_feedback.tryPop(fb)) {
            g_latest_feedback = fb;
        }

        observation_t obs = obs_builder.build(face, frame.cols, frame.rows, &g_latest_feedback);
        q_observation.tryPush(obs);
    }
    std::printf("[T1:Vision] Stopped (%d snapshots saved to %s)\n", snap_count, snap_dir.c_str());
}

// ============================================================================
// T3: CommTxThread — send actions to Teensy
// ============================================================================
static void commTxThread(ServoCommander& servo, const body_config_t& bcfg) {
    std::printf("[T3:CommTx] Started\n");
    int heartbeat_counter = 0;

    while (g_running.load() && !g_emergency.load()) {
        action_t action;
        if (q_action.tryPop(action)) {
            servo.sendPositions(action, bcfg);
        }

        // Heartbeat every 100ms
        if (++heartbeat_counter >= 10) {
            auto now = std::chrono::steady_clock::now().time_since_epoch();
            uint32_t ts = (uint32_t)std::chrono::duration_cast<
                std::chrono::milliseconds>(now).count();
            servo.sendHeartbeat(ts);
            heartbeat_counter = 0;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    servo.sendEmergencyStop();
    std::printf("[T3:CommTx] Stopped\n");
}

// ============================================================================
// T6: LogThread — write CSV logs
// ============================================================================
static void logThread(Logger& logger) {
    std::printf("[T6:Log] Started\n");
    int count = 0;

    while (g_running.load()) {
        log_entry_t entry;
        while (q_log.tryPop(entry)) {
            logger.log(count++, entry.estimated_state, entry.action,
                       entry.ess, entry.safety);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Drain remaining
    log_entry_t entry;
    while (q_log.tryPop(entry)) {
        logger.log(count++, entry.estimated_state, entry.action,
                   entry.ess, entry.safety);
    }
    logger.close();
    std::printf("[T6:Log] Stopped (%d entries)\n", count);
}

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
    std::printf("========================================================\n");
    std::printf("  Phase 7: Full-Body 17-Axis Active Inference Control\n");
    std::printf("  Platform: Jetson AGX Orin + Teensy 4.1 + Manoi PF01\n");
    std::printf("========================================================\n\n");

    std::signal(SIGINT, signalHandler);
    Args args = parseArgs(argc, argv);

    // ── CUDA Info ──
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::printf("[CUDA] %s (SM %d.%d, %d SMs)\n",
                prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    // ── Body Configuration (M9) ──
    body_config_t bcfg = makeDefaultBodyConfig();
    SafetyConfig scfg = makeDefaultSafetyConfig();

    // ── M1: Camera ──
    std::printf("\n[Init] Modules...\n");
    CameraCapture camera;
    if (!camera.init(args.camera_id, 640, 480, 30)) {
        std::fprintf(stderr, "[FATAL] Camera init failed\n");
        return 1;
    }
    camera.start();

    // Wait for first frame
    cv::Mat frame;
    for (int i = 0; i < 100 && !camera.getLatestFrame(frame); i++)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    if (frame.empty()) {
        std::fprintf(stderr, "[FATAL] No frames\n");
        camera.stop();
        return 1;
    }

    // ── M2: Face Detector ──
    FaceDetector detector;
    bool det_ok = detector.init(args.model_dir);

    // ── M3: Observation Builder ──
    ObservationBuilder obs_builder;

    // ── M7: AIF Engine ──
    ActiveInferenceEngine engine;
    if (!engine.init()) {
        std::fprintf(stderr, "[FATAL] AIF engine init failed\n");
        camera.stop();
        return 1;
    }

    // ── M8: Servo Commander ──
    ServoCommander servo;
    servo.init(args.uart_path, args.baudrate, args.loopback);

    // ── M10: Safety Monitor ──
    SafetyMonitor safety;
    safety.init(bcfg, scfg);

    // ── M11: Logger ──
    Logger logger;
    system(("mkdir -p " + args.log_dir).c_str());
    logger.init(args.log_dir);

    // ── Launch threads (§7.1) ──
    std::printf("\n[Run] Launching threads...\n\n");

    std::thread t_vision([&]() { visionThread(camera, detector, obs_builder, args); });
    std::thread t_inference([&]() { inferenceThread(engine, safety, args); });
    std::thread t_comm_tx([&]() { commTxThread(servo, bcfg); });
    std::thread t_log([&]() { logThread(logger); });

    // Main thread: wait for shutdown
    while (g_running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    // ── Shutdown ──
    std::printf("\n[Shutdown] Stopping threads...\n");
    g_running.store(false);

    t_vision.join();
    t_inference.join();
    t_comm_tx.join();
    t_log.join();

    camera.stop();
    engine.destroy();

    std::printf("[Done] Phase 7 complete.\n");
    return 0;
}
