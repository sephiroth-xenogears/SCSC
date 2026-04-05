#pragma once
// ============================================================================
// Phase 7: Full-Body 17-Axis Active Inference — Logger (M11)
// CSV logging for state, action, EFE, safety
// Ref: SCSC-P7-ARCH-001 §3.1 M11
// ============================================================================

#include "../include/types.h"
#include <cstdio>
#include <string>
#include <chrono>

namespace phase7 {

class Logger {
public:
    bool init(const std::string& log_dir) {
        std::string path = log_dir + "/phase7_log.csv";
        fp_ = std::fopen(path.c_str(), "w");
        if (!fp_) {
            std::fprintf(stderr, "[M11:Logger] Failed to open %s\n", path.c_str());
            return false;
        }

        // Header
        std::fprintf(fp_, "timestamp_us,step,efe_total");
        for (int g = 0; g < NUM_GROUPS; g++)
            std::fprintf(fp_, ",efe_g%d", g);
        std::fprintf(fp_, ",ess,safety_level,fault_mask");
        for (int j = 0; j < NUM_JOINTS; j++)
            std::fprintf(fp_, ",cmd_%d", j);
        for (int d = 0; d < STATE_DIM; d++)
            std::fprintf(fp_, ",state_%d", d);
        std::fprintf(fp_, ",confidence,exploration\n");

        std::printf("[M11:Logger] Logging to %s\n", path.c_str());
        return true;
    }

    void log(int step, const state_t& state, const action_t& action,
             float ess, const safety_status_t& safety) {
        if (!fp_) return;

        std::fprintf(fp_, "%lu,%d,%.4f",
                     (unsigned long)action.timestamp_us, step, action.efe_value);

        for (int g = 0; g < NUM_GROUPS; g++)
            std::fprintf(fp_, ",%.4f", action.efe_components[g]);

        std::fprintf(fp_, ",%.1f,%d,%u", ess, safety.level, safety.fault_mask);

        for (int j = 0; j < NUM_JOINTS; j++)
            std::fprintf(fp_, ",%.4f", action.joint_cmd[j]);

        for (int d = 0; d < STATE_DIM; d++)
            std::fprintf(fp_, ",%.4f", state.data[d]);

        std::fprintf(fp_, ",%.4f,%d\n", action.confidence, action.exploration_flag ? 1 : 0);

        if (step % 100 == 0) std::fflush(fp_);
    }

    void close() {
        if (fp_) { std::fclose(fp_); fp_ = nullptr; }
    }

    ~Logger() { close(); }

private:
    FILE* fp_ = nullptr;
};

} // namespace phase7
