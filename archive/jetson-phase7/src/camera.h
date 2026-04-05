#pragma once
// ============================================================================
// Phase 7: Full-Body 17-Axis Active Inference — Async Camera Capture (M1)
// Triple-buffer, lock-free design
// Migrated from Phase 6 — functionally identical
// Ref: SCSC-P7-ARCH-001 §3.1 M1
// ============================================================================

#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <thread>
#include <atomic>
#include <chrono>
#include <cstdio>

namespace phase7 {

// ── Lock-free Triple Buffer ──
class TripleBuffer {
public:
    void init(int width, int height) {
        for (int i = 0; i < 3; i++) {
            buffers_[i].create(height, width, CV_8UC3);
            buffers_[i].setTo(cv::Scalar(0, 0, 0));
        }
        write_idx_.store(0, std::memory_order_relaxed);
        ready_idx_.store(1, std::memory_order_relaxed);
        read_idx_.store(2, std::memory_order_relaxed);
        has_new_.store(false, std::memory_order_relaxed);
    }

    cv::Mat& getWriteBuffer() {
        return buffers_[write_idx_.load(std::memory_order_relaxed)];
    }

    void publishWrite() {
        int w = write_idx_.load(std::memory_order_relaxed);
        int r = ready_idx_.exchange(w, std::memory_order_acq_rel);
        write_idx_.store(r, std::memory_order_relaxed);
        has_new_.store(true, std::memory_order_release);
    }

    bool getLatest(cv::Mat& out) {
        if (!has_new_.load(std::memory_order_acquire)) return false;
        has_new_.store(false, std::memory_order_relaxed);
        int rd = read_idx_.load(std::memory_order_relaxed);
        int ry = ready_idx_.exchange(rd, std::memory_order_acq_rel);
        read_idx_.store(ry, std::memory_order_relaxed);
        buffers_[ry].copyTo(out);
        return true;
    }

private:
    cv::Mat buffers_[3];
    std::atomic<int> write_idx_{0};
    std::atomic<int> ready_idx_{1};
    std::atomic<int> read_idx_{2};
    std::atomic<bool> has_new_{false};
};

// ── Async Camera (M1: CameraCapture) ──
class CameraCapture {
public:
    ~CameraCapture() { stop(); }

    bool init(int device_id, int width, int height, int fps) {
        device_id_ = device_id;
        width_  = width;
        height_ = height;
        fps_    = fps;
        buffer_.init(width, height);

        cap_.open(device_id, cv::CAP_V4L2);
        if (!cap_.isOpened()) {
            cap_.open(device_id, cv::CAP_ANY);
        }
        if (!cap_.isOpened()) {
            std::fprintf(stderr, "[M1:Camera] Failed to open device %d\n", device_id);
            return false;
        }

        cap_.set(cv::CAP_PROP_FRAME_WIDTH,  width);
        cap_.set(cv::CAP_PROP_FRAME_HEIGHT, height);
        cap_.set(cv::CAP_PROP_FPS,          fps);

        int actual_w = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
        int actual_h = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
        std::printf("[M1:Camera] Opened device %d: %dx%d @ %dfps (requested %dx%d)\n",
                    device_id, actual_w, actual_h, fps, width, height);

        initialized_ = true;
        return true;
    }

    void start() {
        if (!initialized_ || running_.load()) return;
        running_.store(true, std::memory_order_release);
        thread_ = std::thread(&CameraCapture::captureLoop, this);
        std::printf("[M1:Camera] Capture thread started\n");
    }

    void stop() {
        running_.store(false, std::memory_order_release);
        if (thread_.joinable()) thread_.join();
        if (cap_.isOpened()) cap_.release();
    }

    bool getLatestFrame(cv::Mat& out) {
        return buffer_.getLatest(out);
    }

    float   getFPS()       const { return measured_fps_.load(std::memory_order_relaxed); }
    uint64_t frameCount()  const { return frame_count_.load(std::memory_order_relaxed); }
    bool    isRunning()    const { return running_.load(std::memory_order_relaxed); }

private:
    void captureLoop() {
        cv::Mat temp;
        auto t_prev = std::chrono::steady_clock::now();
        int count = 0;

        while (running_.load(std::memory_order_acquire)) {
            if (!cap_.read(temp) || temp.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            cv::Mat& wb = buffer_.getWriteBuffer();
            temp.copyTo(wb);
            buffer_.publishWrite();

            frame_count_.fetch_add(1, std::memory_order_relaxed);
            count++;

            if (count >= 30) {
                auto t_now  = std::chrono::steady_clock::now();
                float elapsed = std::chrono::duration<float>(t_now - t_prev).count();
                measured_fps_.store(count / elapsed, std::memory_order_relaxed);
                t_prev = t_now;
                count = 0;
            }
        }
    }

    cv::VideoCapture cap_;
    TripleBuffer buffer_;
    std::thread  thread_;

    int device_id_ = 0;
    int width_  = 640;
    int height_ = 480;
    int fps_    = 30;

    std::atomic<bool>     initialized_{false};
    std::atomic<bool>     running_{false};
    std::atomic<float>    measured_fps_{0.0f};
    std::atomic<uint64_t> frame_count_{0};
};

} // namespace phase7
