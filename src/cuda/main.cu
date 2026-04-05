/**
 * SCSC Host Runner — Profile Mode
 * =================================
 * Loads test vectors from binary files, launches kernel, prints results.
 * Used for Nsight Compute profiling (M2) and manual verification.
 *
 * Usage: ./scsc_binary [test_vectors_dir]
 *   Default: ../tests/test_vectors/
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>

#include <cuda_runtime.h>
#include "gen_model.cuh"
#include "active_inference_kernel.cuh"

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// ============================================================
// Binary file loader
// ============================================================
static bool load_bin(const char* path, void* dst, size_t bytes) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s\n", path);
        return false;
    }
    size_t read = fread(dst, 1, bytes, f);
    fclose(f);
    if (read != bytes) {
        fprintf(stderr, "ERROR: %s: expected %zu bytes, got %zu\n", path, bytes, read);
        return false;
    }
    return true;
}

int main(int argc, char* argv[]) {
    std::string vec_dir = "../tests/test_vectors";
    if (argc > 1) vec_dir = argv[1];

    printf("=== SCSC Active Inference — Profile Mode ===\n");
    printf("Test vectors: %s/\n\n", vec_dir.c_str());

    // ---- Load test vectors ----
    GenerativeModelHost model;
    float mu_init[STATE_DIM];
    float obs[STATE_DIM];

    bool ok = true;
    ok &= load_bin((vec_dir + "/A.bin").c_str(),      model.A,    sizeof(model.A));
    ok &= load_bin((vec_dir + "/B.bin").c_str(),      model.B,    sizeof(model.B));
    ok &= load_bin((vec_dir + "/C.bin").c_str(),      model.C,    sizeof(model.C));
    ok &= load_bin((vec_dir + "/D.bin").c_str(),      model.D,    sizeof(model.D));
    ok &= load_bin((vec_dir + "/Pi_o.bin").c_str(),   model.Pi_o, sizeof(model.Pi_o));
    ok &= load_bin((vec_dir + "/Pi_x.bin").c_str(),   model.Pi_x, sizeof(model.Pi_x));
    ok &= load_bin((vec_dir + "/mu_init.bin").c_str(), mu_init,    sizeof(mu_init));
    ok &= load_bin((vec_dir + "/o.bin").c_str(),       obs,        sizeof(obs));
    if (!ok) {
        fprintf(stderr, "Failed to load test vectors. Run: python3 src/python/test_reference.py --export tests/test_vectors\n");
        return EXIT_FAILURE;
    }
    printf("Loaded all test vectors.\n");

    // ---- Upload B to Constant Memory ----
    CUDA_CHECK(cudaMemcpyToSymbol(d_B, model.B, sizeof(model.B)));

    // ---- Allocate device memory ----
    float *d_A, *d_C, *d_D, *d_Pi_o, *d_Pi_x, *d_obs, *d_mu;
    InferenceOutput *d_out;

    CUDA_CHECK(cudaMalloc(&d_A,    STATE_DIM * STATE_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C,    STATE_DIM * STATE_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_D,    STATE_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Pi_o, STATE_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Pi_x, STATE_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_obs,  STATE_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mu,   STATE_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out,  sizeof(InferenceOutput)));

    // ---- Copy to device ----
    CUDA_CHECK(cudaMemcpy(d_A,    model.A,    STATE_DIM * STATE_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C,    model.C,    STATE_DIM * STATE_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_D,    model.D,    STATE_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Pi_o, model.Pi_o, STATE_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Pi_x, model.Pi_x, STATE_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_obs,  obs,        STATE_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mu,   mu_init,    STATE_DIM * sizeof(float), cudaMemcpyHostToDevice));

    // ---- Launch kernel (Profile mode) ----
    printf("Launching kernel <<<1, %d>>> ...\n", BLOCK_SIZE);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    launch_active_inference(
        d_A, d_C, d_D, d_Pi_o, d_Pi_x,
        d_obs, d_mu, d_out,
        0, NUM_POLICIES
    );
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    printf("Kernel time: %.3f ms\n\n", elapsed_ms);

    // ---- Read back results ----
    InferenceOutput h_out;
    CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(InferenceOutput), cudaMemcpyDeviceToHost));

    // ---- Print results ----
    printf("Stage 1 — VFE:\n");
    printf("  F = %.6f\n", h_out.F);
    printf("  mu[0..4] = ");
    for (int i = 0; i < 5; i++) printf("%.6f ", h_out.mu[i]);
    printf("...\n");

    printf("\nStage 2 — G(π):\n");
    for (int k = 0; k < NUM_POLICIES; k++) {
        const char* marker = (k == h_out.action) ? " <-- best" : "";
        printf("  G[%d] = %.6f  P = %.4f%s\n", k, h_out.G[k], h_out.probs[k], marker);
    }

    printf("\nStage 3 — Action:\n");
    printf("  Selected policy: %d\n", h_out.action);

    // ---- Compare with expected values ----
    float F_expected;
    float mu_expected[STATE_DIM];
    float G_expected[NUM_POLICIES];
    float probs_expected[NUM_POLICIES];
    int   action_expected;

    ok = true;
    ok &= load_bin((vec_dir + "/F_expected.bin").c_str(),      &F_expected,     sizeof(float));
    ok &= load_bin((vec_dir + "/mu_expected.bin").c_str(),     mu_expected,     sizeof(mu_expected));
    ok &= load_bin((vec_dir + "/G_expected.bin").c_str(),      G_expected,      sizeof(G_expected));
    ok &= load_bin((vec_dir + "/probs_expected.bin").c_str(),  probs_expected,  sizeof(probs_expected));
    ok &= load_bin((vec_dir + "/action_expected.bin").c_str(), &action_expected, sizeof(int));

    if (ok) {
        printf("\n=== Verification ===\n");
        float tol = 1e-5f;

        // F
        float f_err = fabsf(h_out.F - F_expected);
        printf("F:  cuda=%.6f  ref=%.6f  err=%.2e  %s\n",
               h_out.F, F_expected, f_err, f_err < tol ? "PASS" : "FAIL");

        // mu max error
        float mu_max_err = 0.0f;
        for (int i = 0; i < STATE_DIM; i++) {
            float e = fabsf(h_out.mu[i] - mu_expected[i]);
            if (e > mu_max_err) mu_max_err = e;
        }
        printf("mu: max_err=%.2e  %s\n", mu_max_err, mu_max_err < tol ? "PASS" : "FAIL");

        // G max error
        float g_max_err = 0.0f;
        for (int k = 0; k < NUM_POLICIES; k++) {
            float e = fabsf(h_out.G[k] - G_expected[k]);
            if (e > g_max_err) g_max_err = e;
        }
        printf("G:  max_err=%.2e  %s\n", g_max_err, g_max_err < tol ? "PASS" : "FAIL");

        // probs max error
        float p_max_err = 0.0f;
        for (int k = 0; k < NUM_POLICIES; k++) {
            float e = fabsf(h_out.probs[k] - probs_expected[k]);
            if (e > p_max_err) p_max_err = e;
        }
        printf("P:  max_err=%.2e  %s\n", p_max_err, p_max_err < tol ? "PASS" : "FAIL");

        // action
        printf("action: cuda=%d  ref=%d  %s\n",
               h_out.action, action_expected,
               h_out.action == action_expected ? "PASS" : "FAIL");
    }

    // ---- Cleanup ----
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_D));
    CUDA_CHECK(cudaFree(d_Pi_o));
    CUDA_CHECK(cudaFree(d_Pi_x));
    CUDA_CHECK(cudaFree(d_obs));
    CUDA_CHECK(cudaFree(d_mu));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    printf("\nDone.\n");
    return EXIT_SUCCESS;
}
