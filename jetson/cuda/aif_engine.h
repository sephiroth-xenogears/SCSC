#pragma once
// ============================================================================
// Phase 7: Full-Body 17-Axis Active Inference — AIF Engine (M7)
// Integrates M4 (GenModel), M5 (ParticleFilter), M6 (EFEPlanner)
// Manages CUDA memory, streams, and the inference loop
// Ref: SCSC-P7-ARCH-001 §3.1 M7
// ============================================================================

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../include/types.h"
#include "../include/body_config.h"
#include "gen_model.cuh"
#include "particle_filter.cuh"
#include "efe_planner.cuh"

namespace phase7 {

// ── Helpers ──
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n", \
                     __FILE__, __LINE__, cudaGetErrorString(err)); \
        return false; \
    } \
} while(0)

#define CUDA_CHECK_VOID(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::fprintf(stderr, "[CUDA ERROR] %s:%d: %s\n", \
                     __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while(0)

// ── Engine Configuration ──
struct EngineConfig {
    int   num_particles;
    float ess_threshold_ratio;
    unsigned long long rng_seed;
};

inline EngineConfig makeDefaultEngineConfig() {
    return { 5000, 0.5f, 42ULL };
}

// ============================================================================
class ActiveInferenceEngine {
public:
    // ── Lifecycle ──
    bool init(const EngineConfig& ecfg   = makeDefaultEngineConfig(),
              const body_config_t& bcfg  = makeDefaultBodyConfig(),
              const GenModelParams& mcfg = makeDefaultGenModelParams())
    {
        N_ = ecfg.num_particles;
        ess_threshold_ = ecfg.ess_threshold_ratio * N_;
        h_body_cfg_ = bcfg;
        h_model_params_ = mcfg;

        int nblocks = (N_ + PF_BLOCK_SIZE - 1) / PF_BLOCK_SIZE;

        // ── Allocate GPU memory ──
        CUDA_CHECK(cudaMalloc(&d_particles_,    N_ * STATE_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_particles_tmp_, N_ * STATE_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_weights_,       N_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_log_weights_,   N_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_cdf_,           N_ * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_block_maxes_,   nblocks * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_partial_sums_,  nblocks * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_block_ess_,     nblocks * sizeof(float)));

        CUDA_CHECK(cudaMalloc(&d_rng_states_,    N_ * sizeof(curandState)));
        CUDA_CHECK(cudaMalloc(&d_action_,        ACTION_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_state_est_,     STATE_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_covariance_,    STATE_DIM * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_observation_,   sizeof(gpu_observation_t)));

        CUDA_CHECK(cudaMalloc(&d_body_cfg_,      sizeof(body_config_t)));
        CUDA_CHECK(cudaMalloc(&d_model_params_,  sizeof(GenModelParams)));

        // EFE planner memory
        int cand_total = NUM_GROUPS * CANDIDATES_PER_GROUP * ACTION_DIM;
        CUDA_CHECK(cudaMalloc(&d_group_candidates_, cand_total * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_group_efe_,       NUM_GROUPS * CANDIDATES_PER_GROUP * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_top_k_indices_,   NUM_GROUPS * TOP_K_PER_GROUP * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_top_k_efe_,       NUM_GROUPS * TOP_K_PER_GROUP * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_combination_map_,  NUM_FINAL_CANDIDATES * NUM_GROUPS * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_combined_efe_,     NUM_FINAL_CANDIDATES * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_combined_efe_per_group_, NUM_FINAL_CANDIDATES * NUM_GROUPS * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_selected_efe_,     sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_selected_conf_,    sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_selected_efe_pg_,  NUM_GROUPS * sizeof(float)));

        // ── Upload configs ──
        CUDA_CHECK(cudaMemcpy(d_body_cfg_, &h_body_cfg_, sizeof(body_config_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_model_params_, &h_model_params_, sizeof(GenModelParams), cudaMemcpyHostToDevice));

        // ── Initialize RNG and particles ──
        initRngKernel<<<nblocks, PF_BLOCK_SIZE>>>(d_rng_states_, ecfg.rng_seed, N_);
        initParticlesKernel<<<nblocks, PF_BLOCK_SIZE>>>(d_particles_, d_rng_states_, N_);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Zero action
        CUDA_CHECK(cudaMemset(d_action_, 0, ACTION_DIM * sizeof(float)));

        std::printf("[M7:AIFEngine] Initialized: %d particles, %d-dim state, %d-dim action\n",
                    N_, STATE_DIM, ACTION_DIM);
        initialized_ = true;
        return true;
    }

    // ── Main inference step ──
    action_t step(const observation_t& obs) {
        if (!initialized_) {
            action_t a; a.clear();
            return a;
        }

        int nblocks = (N_ + PF_BLOCK_SIZE - 1) / PF_BLOCK_SIZE;

        // 1. Upload observation
        gpu_observation_t gpu_obs = ObservationBuilder::toGPU(obs);
        cudaMemcpy(d_observation_, &gpu_obs, sizeof(gpu_observation_t), cudaMemcpyHostToDevice);

        // 2. Predict
        predictKernel<<<nblocks, PF_BLOCK_SIZE>>>(
            d_particles_, d_action_, d_body_cfg_, d_model_params_,
            d_rng_states_, N_);

        // 3. Likelihood
        likelihoodKernel<<<nblocks, PF_BLOCK_SIZE>>>(
            d_particles_, d_observation_, d_body_cfg_, d_model_params_,
            d_log_weights_, N_);

        // 4. Normalize weights (3-pass)
        findMaxWeightKernel<<<nblocks, PF_BLOCK_SIZE>>>(d_log_weights_, d_block_maxes_, N_);
        // Find global max on host (simple for small nblocks)
        float h_maxes[128];
        cudaMemcpy(h_maxes, d_block_maxes_, nblocks * sizeof(float), cudaMemcpyDeviceToHost);
        float max_w = h_maxes[0];
        for (int i = 1; i < nblocks; i++) max_w = std::max(max_w, h_maxes[i]);

        normalizeWeightsKernel<<<nblocks, PF_BLOCK_SIZE>>>(
            d_log_weights_, d_weights_, d_partial_sums_, max_w, N_);
        finalNormalizeKernel<<<nblocks, PF_BLOCK_SIZE>>>(
            d_weights_, d_partial_sums_, nblocks, N_);

        // 5. ESS check + conditional resample
        essKernel<<<nblocks, PF_BLOCK_SIZE>>>(d_weights_, d_block_ess_, N_);
        float h_ess_parts[128];
        cudaMemcpy(h_ess_parts, d_block_ess_, nblocks * sizeof(float), cudaMemcpyDeviceToHost);
        float sum_w2 = 0.0f;
        for (int i = 0; i < nblocks; i++) sum_w2 += h_ess_parts[i];
        float ess = (sum_w2 > 0.0f) ? 1.0f / sum_w2 : 0.0f;
        last_ess_ = ess;

        if (ess < ess_threshold_) {
            buildCdfKernel<<<1, 1>>>(d_weights_, d_cdf_, N_);
            float u0 = (float)std::rand() / RAND_MAX;
            systematicResampleKernel<<<nblocks, PF_BLOCK_SIZE>>>(
                d_cdf_, d_particles_, d_particles_tmp_, u0, N_);
            std::swap(d_particles_, d_particles_tmp_);
        }

        // 6. State estimation
        estimateStateKernel<<<1, PF_BLOCK_SIZE>>>(
            d_particles_, d_weights_, d_state_est_, d_covariance_, N_);

        // 7. EFE planning — Stage 1: per-group
        generateAndUploadCandidates();
        for (int g = 0; g < NUM_GROUPS; g++) {
            efeGroupKernel<<<CANDIDATES_PER_GROUP, EFE_BLOCK_SIZE>>>(
                d_particles_, d_weights_, d_group_candidates_,
                d_group_efe_, d_body_cfg_, d_model_params_, g, N_);
        }

        // Stage 1.5: top-K selection
        selectTopKKernel<<<1, NUM_GROUPS>>>(d_group_efe_, d_top_k_indices_, d_top_k_efe_);

        // Stage 2: combined evaluation
        generateAndUploadCombinationMap();
        efeCombinedKernel<<<NUM_FINAL_CANDIDATES, EFE_BLOCK_SIZE>>>(
            d_particles_, d_weights_, d_group_candidates_,
            d_top_k_indices_, d_combination_map_,
            d_combined_efe_, d_combined_efe_per_group_,
            d_body_cfg_, d_model_params_, N_);

        // Softmax selection
        softmaxActionKernel<<<1, NUM_FINAL_CANDIDATES>>>(
            d_combined_efe_, d_group_candidates_,
            d_top_k_indices_, d_combination_map_,
            d_action_, d_selected_efe_, d_selected_efe_pg_,
            d_selected_conf_, d_body_cfg_,
            h_model_params_.action_temperature, d_rng_states_,
            NUM_FINAL_CANDIDATES);

        cudaDeviceSynchronize();

        // 8. Download results
        action_t result;
        cudaMemcpy(result.joint_cmd, d_action_, ACTION_DIM * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&result.efe_value, d_selected_efe_, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&result.confidence, d_selected_conf_, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(result.efe_components, d_selected_efe_pg_, NUM_GROUPS * sizeof(float), cudaMemcpyDeviceToHost);

        auto now = std::chrono::steady_clock::now().time_since_epoch();
        result.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(now).count();
        result.exploration_flag = (ess < ess_threshold_ * 0.3f);

        step_count_++;
        return result;
    }

    // ── Get estimated state ──
    state_t getEstimatedState() {
        state_t s;
        cudaMemcpy(s.data, d_state_est_, STATE_DIM * sizeof(float), cudaMemcpyDeviceToHost);
        return s;
    }

    float getESS() const { return last_ess_; }
    int   getStepCount() const { return step_count_; }

    // ── Cleanup ──
    void destroy() {
        if (!initialized_) return;

        cudaFree(d_particles_);     cudaFree(d_particles_tmp_);
        cudaFree(d_weights_);       cudaFree(d_log_weights_);
        cudaFree(d_cdf_);           cudaFree(d_block_maxes_);
        cudaFree(d_partial_sums_);  cudaFree(d_block_ess_);
        cudaFree(d_rng_states_);    cudaFree(d_action_);
        cudaFree(d_state_est_);     cudaFree(d_covariance_);
        cudaFree(d_observation_);
        cudaFree(d_body_cfg_);      cudaFree(d_model_params_);

        cudaFree(d_group_candidates_); cudaFree(d_group_efe_);
        cudaFree(d_top_k_indices_);    cudaFree(d_top_k_efe_);
        cudaFree(d_combination_map_);  cudaFree(d_combined_efe_);
        cudaFree(d_combined_efe_per_group_);
        cudaFree(d_selected_efe_);     cudaFree(d_selected_conf_);
        cudaFree(d_selected_efe_pg_);

        initialized_ = false;
        std::printf("[M7:AIFEngine] Destroyed\n");
    }

private:
    void generateAndUploadCandidates() {
        int total = NUM_GROUPS * CANDIDATES_PER_GROUP * ACTION_DIM;
        float* h_cand = (float*)std::malloc(total * sizeof(float));
        generateGroupCandidates(h_cand, h_body_cfg_, step_count_);
        cudaMemcpy(d_group_candidates_, h_cand, total * sizeof(float), cudaMemcpyHostToDevice);
        std::free(h_cand);
    }

    void generateAndUploadCombinationMap() {
        int total = NUM_FINAL_CANDIDATES * NUM_GROUPS;
        int* h_map = (int*)std::malloc(total * sizeof(int));
        generateCombinationMap(h_map, step_count_ + 7);
        cudaMemcpy(d_combination_map_, h_map, total * sizeof(int), cudaMemcpyHostToDevice);
        std::free(h_map);
    }

    // ── Configuration ──
    int N_ = 0;
    float ess_threshold_ = 0.0f;
    body_config_t   h_body_cfg_;
    GenModelParams  h_model_params_;

    // ── State ──
    bool initialized_ = false;
    int  step_count_  = 0;
    float last_ess_   = 0.0f;

    // ── GPU Memory: Particle Filter (M5) ──
    float* d_particles_     = nullptr;
    float* d_particles_tmp_ = nullptr;
    float* d_weights_       = nullptr;
    float* d_log_weights_   = nullptr;
    float* d_cdf_           = nullptr;
    float* d_block_maxes_   = nullptr;
    float* d_partial_sums_  = nullptr;
    float* d_block_ess_     = nullptr;
    curandState* d_rng_states_ = nullptr;

    // ── GPU Memory: State/Action ──
    float* d_action_     = nullptr;
    float* d_state_est_  = nullptr;
    float* d_covariance_ = nullptr;
    gpu_observation_t* d_observation_ = nullptr;

    // ── GPU Memory: Config ──
    body_config_t*  d_body_cfg_     = nullptr;
    GenModelParams* d_model_params_ = nullptr;

    // ── GPU Memory: EFE Planner (M6) ──
    float* d_group_candidates_        = nullptr;
    float* d_group_efe_               = nullptr;
    int*   d_top_k_indices_           = nullptr;
    float* d_top_k_efe_               = nullptr;
    int*   d_combination_map_         = nullptr;
    float* d_combined_efe_            = nullptr;
    float* d_combined_efe_per_group_  = nullptr;
    float* d_selected_efe_            = nullptr;
    float* d_selected_conf_           = nullptr;
    float* d_selected_efe_pg_         = nullptr;
};

} // namespace phase7
