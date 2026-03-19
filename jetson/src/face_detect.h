#pragma once
// ============================================================================
// Phase 7: Full-Body 17-Axis Active Inference — Face Detector (M2)
// Strategy: SSD ResNet10 DNN → Haar Cascade fallback
// Migrated from Phase 6 — types updated to phase7::FaceDetection
// Ref: SCSC-P7-ARCH-001 §3.1 M2
// ============================================================================

#include <opencv2/dnn.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <string>
#include <cstdio>
#include <algorithm>

#include "../include/types.h"

namespace phase7 {

// ── Configuration ──
static constexpr float FACE_CONFIDENCE_THRESHOLD = 0.5f;

class FaceDetector {
public:
    bool init(const std::string& model_dir = "models") {
        // Try DNN first
        std::string prototxt   = model_dir + "/deploy.prototxt";
        std::string caffemodel = model_dir +
            "/res10_300x300_ssd_iter_140000_fp16.caffemodel";

        try {
            net_ = cv::dnn::readNetFromCaffe(prototxt, caffemodel);
            if (!net_.empty()) {
                net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
                use_dnn_ = true;
                std::printf("[M2:FaceDetect] DNN (SSD ResNet10) loaded — CUDA\n");
                return true;
            }
        } catch (...) {
            std::printf("[M2:FaceDetect] DNN load failed, trying fallbacks\n");
        }

        // Fallback: DNN CPU
        if (!use_dnn_) {
            try {
                net_ = cv::dnn::readNetFromCaffe(prototxt, caffemodel);
                if (!net_.empty()) {
                    net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                    net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
                    use_dnn_ = true;
                    std::printf("[M2:FaceDetect] DNN (SSD ResNet10) loaded — CPU\n");
                    return true;
                }
            } catch (...) {}
        }

        // Fallback: Haar Cascade
        const char* haar_paths[] = {
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt2.xml",
        };
        for (auto path : haar_paths) {
            if (haar_.load(path)) {
                use_dnn_ = false;
                std::printf("[M2:FaceDetect] Haar Cascade loaded: %s\n", path);
                return true;
            }
        }

        std::fprintf(stderr, "[M2:FaceDetect] ERROR: No backend available!\n");
        return false;
    }

    FaceDetection detect(const cv::Mat& frame) {
        return use_dnn_ ? detectDNN(frame) : detectHaar(frame);
    }

    bool isDNN() const { return use_dnn_; }

private:
    FaceDetection detectDNN(const cv::Mat& frame) {
        cv::Mat blob = cv::dnn::blobFromImage(
            frame, 1.0, cv::Size(300, 300),
            cv::Scalar(104.0, 177.0, 123.0), false, false);

        net_.setInput(blob);
        cv::Mat detections = net_.forward();

        cv::Mat det_mat(detections.size[2], detections.size[3],
                        CV_32F, detections.ptr<float>());

        std::vector<FaceDetection> faces;
        for (int i = 0; i < det_mat.rows; i++) {
            float conf = det_mat.at<float>(i, 2);
            if (conf < FACE_CONFIDENCE_THRESHOLD) continue;

            float x1 = std::clamp(det_mat.at<float>(i, 3) * frame.cols, 0.0f, (float)frame.cols);
            float y1 = std::clamp(det_mat.at<float>(i, 4) * frame.rows, 0.0f, (float)frame.rows);
            float x2 = std::clamp(det_mat.at<float>(i, 5) * frame.cols, 0.0f, (float)frame.cols);
            float y2 = std::clamp(det_mat.at<float>(i, 6) * frame.rows, 0.0f, (float)frame.rows);

            FaceDetection fd;
            fd.x = x1; fd.y = y1;
            fd.w = x2 - x1; fd.h = y2 - y1;
            fd.confidence = conf;
            fd.detected = true;

            if (fd.w > 5.0f && fd.h > 5.0f) faces.push_back(fd);
        }
        return selectBest(faces);
    }

    FaceDetection detectHaar(const cv::Mat& frame) {
        cv::Mat gray;
        if (frame.channels() == 3) cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        else gray = frame;
        cv::equalizeHist(gray, gray);

        std::vector<cv::Rect> rects;
        haar_.detectMultiScale(gray, rects, 1.1, 3, 0, cv::Size(30, 30));

        std::vector<FaceDetection> faces;
        for (auto& r : rects) {
            FaceDetection fd;
            fd.x = (float)r.x; fd.y = (float)r.y;
            fd.w = (float)r.width; fd.h = (float)r.height;
            fd.confidence = 1.0f;
            fd.detected = true;
            faces.push_back(fd);
        }
        return selectBest(faces);
    }

    FaceDetection selectBest(const std::vector<FaceDetection>& faces) {
        if (faces.empty()) return FaceDetection{};
        return *std::max_element(faces.begin(), faces.end(),
            [](const FaceDetection& a, const FaceDetection& b) {
                return (a.w * a.h) < (b.w * b.h);
            });
    }

    cv::dnn::Net          net_;
    cv::CascadeClassifier haar_;
    bool                  use_dnn_ = false;
};

} // namespace phase7
