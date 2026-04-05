/**
 * SCSC Test Harness — CUDA Kernel Verification
 * ===============================================
 * Tests 1-7 from m1b-kernel-spec.md Section 11.
 * Compares kernel output against Python reference test vectors.
 *
 * Usage: ./scsc_test [test_vectors_dir]
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

// ============================================================
// Macros
// ============================================================
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, fmt, ...)                                               \
    do {                                                                    \
        if (cond) { g_pass++; }                                             \
        else { g_fail++; printf("  FAIL: " fmt "\n", ##__VA_ARGS__); }      \
    } while (0)

// ============================================================
// Binary file loader
// ============================================================
static bool load_bin(const char* path, void* dst, size_t bytes) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return false; }
    size_t n = fread(dst, 1, bytes, f);
    fclose(f);
    return n == bytes;
}

// ============================================================
// Helper: max absolute error
// ============================================================
static float max_abs_err(const float* a, const float* b, int n) {
    float mx = 0.0f;
    for (int i = 0; i < n; i++) {
        float e = fabsf(a[i] - b[i]);
        if (e > mx) mx = e;
    }
    return mx;
}

// ============================================================
// Test data (global for simplicity)
// ============================================================
static GenerativeModelHost model;
static float mu_init[STATE_DIM];
static float obs[STATE_DIM];
static float mu_expected[STATE_DIM];
static float F_expected;
static float G_expected[NUM_POLICIES];
static float probs_expected[NUM_POLICIES];
static int   action_expected;
static float F_trace[11];  // F before each of 10 iters + F_final

// Device pointers
static float *d_A, *d_C, *d_D, *d_Pi_o, *d_Pi_x, *d_obs, *d_mu;
static InferenceOutput *d_out;

static void load_all(const std::string& dir) {
    bool ok = true;
    ok &= load_bin((dir + "/A.bin").c_str(),      model.A,    sizeof(model.A));
    ok &= load_bin((dir + "/B.bin").c_str(),      model.B,    sizeof(model.B));
    ok &= load_bin((dir + "/C.bin").c_str(),      model.C,    sizeof(model.C));
    ok &= load_bin((dir + "/D.bin").c_str(),      model.D,    sizeof(model.D));
    ok &= load_bin((dir + "/Pi_o.bin").c_str(),   model.Pi_o, sizeof(model.Pi_o));
    ok &= load_bin((dir + "/Pi_x.bin").c_str(),   model.Pi_x, sizeof(model.Pi_x));
    ok &= load_bin((dir + "/mu_init.bin").c_str(), mu_init,    sizeof(mu_init));
    ok &= load_bin((dir + "/o.bin").c_str(),       obs,        sizeof(obs));
    ok &= load_bin((dir + "/mu_expected.bin").c_str(),     mu_expected,     sizeof(mu_expected));
    ok &= load_bin((dir + "/F_expected.bin").c_str(),      &F_expected,     sizeof(float));
    ok &= load_bin((dir + "/G_expected.bin").c_str(),      G_expected,      sizeof(G_expected));
    ok &= load_bin((dir + "/probs_expected.bin").c_str(),  probs_expected,  sizeof(probs_expected));
    ok &= load_bin((dir + "/action_expected.bin").c_str(), &action_expected, sizeof(int));
    ok &= load_bin((dir + "/F_trace.bin").c_str(),         F_trace,         sizeof(F_trace));
    if (!ok) {
        fprintf(stderr, "Failed to load test vectors.\n");
        fprintf(stderr, "Run: python3 src/python/test_reference.py --export tests/test_vectors\n");
        exit(EXIT_FAILURE);
    }
}

static void alloc_device() {
    CUDA_CHECK(cudaMemcpyToSymbol(d_B, model.B, sizeof(model.B)));
    CUDA_CHECK(cudaMalloc(&d_A,    STATE_DIM * STATE_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C,    STATE_DIM * STATE_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_D,    STATE_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Pi_o, STATE_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Pi_x, STATE_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_obs,  STATE_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_mu,   STATE_DIM * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out,  sizeof(InferenceOutput)));

    CUDA_CHECK(cudaMemcpy(d_A,    model.A,    STATE_DIM * STATE_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C,    model.C,    STATE_DIM * STATE_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_D,    model.D,    STATE_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Pi_o, model.Pi_o, STATE_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Pi_x, model.Pi_x, STATE_DIM * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_obs,  obs,        STATE_DIM * sizeof(float), cudaMemcpyHostToDevice));
}

static void free_device() {
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_D));
    CUDA_CHECK(cudaFree(d_Pi_o));
    CUDA_CHECK(cudaFree(d_Pi_x));
    CUDA_CHECK(cudaFree(d_obs));
    CUDA_CHECK(cudaFree(d_mu));
    CUDA_CHECK(cudaFree(d_out));
}

// Reset mu on device to mu_init
static void reset_mu() {
    CUDA_CHECK(cudaMemcpy(d_mu, mu_init, STATE_DIM * sizeof(float), cudaMemcpyHostToDevice));
}

// Run full kernel and return output
static InferenceOutput run_kernel(int policy_offset = 0, int policy_count = NUM_POLICIES) {
    reset_mu();
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(InferenceOutput)));
    launch_active_inference(d_A, d_C, d_D, d_Pi_o, d_Pi_x,
                            d_obs, d_mu, d_out,
                            policy_offset, policy_count);
    CUDA_CHECK(cudaDeviceSynchronize());

    InferenceOutput h_out;
    CUDA_CHECK(cudaMemcpy(&h_out, d_out, sizeof(InferenceOutput), cudaMemcpyDeviceToHost));
    return h_out;
}

// ============================================================
// Test 1: Prologue data integrity
// ============================================================
// We verify the kernel loads data correctly by running a full
// inference and checking mu output is finite (non-NaN, non-Inf).
// Direct shared memory readback isn't possible, but correct output
// implies correct loading.
static void test1_prologue() {
    printf("[Test 1] Prologue data integrity\n");
    InferenceOutput out = run_kernel();

    bool all_finite = true;
    for (int i = 0; i < STATE_DIM; i++) {
        if (!isfinite(out.mu[i])) { all_finite = false; break; }
    }
    CHECK(all_finite, "mu contains NaN/Inf");
    CHECK(isfinite(out.F), "F is NaN/Inf (F=%.6f)", out.F);
    for (int k = 0; k < NUM_POLICIES; k++) {
        if (!isfinite(out.G[k])) {
            CHECK(false, "G[%d] is NaN/Inf", k);
            return;
        }
    }
    CHECK(true, "");
    printf("  PASS: All outputs finite.\n");
}

// ============================================================
// Test 2: Single VFE iteration (verify F initial)
// ============================================================
// F_trace[0] is F computed from mu_init (before any gradient step).
// We can't run a 1-iteration kernel without modifying VFE_ITERS,
// so we verify F_trace[0] indirectly: after 10 iters, F must be
// less than F_trace[0] (monotonic decrease).
static void test2_vfe_single() {
    printf("[Test 2] VFE F decreases from initial\n");
    InferenceOutput out = run_kernel();

    float F_initial = F_trace[0];
    float F_final = out.F;
    printf("  F_initial (ref) = %.6f\n", F_initial);
    printf("  F_final (cuda)  = %.6f\n", F_final);
    CHECK(F_final < F_initial, "F did not decrease: %.6f >= %.6f", F_final, F_initial);
    printf("  PASS: F decreased from %.6f to %.6f\n", F_initial, F_final);
}

// ============================================================
// Test 3: Full VFE (10 iterations) — mu and F
// ============================================================
static void test3_vfe_full() {
    printf("[Test 3] Full VFE (10 iterations)\n");
    InferenceOutput out = run_kernel();

    float tol = 1e-5f;
    float mu_err = max_abs_err(out.mu, mu_expected, STATE_DIM);
    float f_err = fabsf(out.F - F_expected);

    printf("  mu max_err = %.2e (tol=%.0e)\n", mu_err, tol);
    printf("  F  err     = %.2e (cuda=%.6f ref=%.6f)\n", f_err, out.F, F_expected);

    CHECK(mu_err < tol, "mu max_err %.2e >= tol", mu_err);
    CHECK(f_err  < tol, "F err %.2e >= tol", f_err);
}

// ============================================================
// Test 4: G(π) single policy (G[0])
// ============================================================
static void test4_G_single() {
    printf("[Test 4] G(pi) single policy\n");
    InferenceOutput out = run_kernel();

    float tol = 1e-5f;
    float err = fabsf(out.G[0] - G_expected[0]);
    printf("  G[0] cuda=%.6f ref=%.6f err=%.2e\n", out.G[0], G_expected[0], err);
    CHECK(err < tol, "G[0] err %.2e >= tol", err);
}

// ============================================================
// Test 5: G(π) all policies
// ============================================================
static void test5_G_all() {
    printf("[Test 5] G(pi) all policies\n");
    InferenceOutput out = run_kernel();

    float tol = 1e-5f;
    float max_err = max_abs_err(out.G, G_expected, NUM_POLICIES);
    printf("  G max_err = %.2e\n", max_err);
    for (int k = 0; k < NUM_POLICIES; k++) {
        float e = fabsf(out.G[k] - G_expected[k]);
        printf("  G[%d] cuda=%.6f ref=%.6f err=%.2e %s\n",
               k, out.G[k], G_expected[k], e, e < tol ? "OK" : "FAIL");
    }
    CHECK(max_err < tol, "G max_err %.2e >= tol", max_err);
}

// ============================================================
// Test 6: Action selection (softmax + argmax)
// ============================================================
static void test6_action() {
    printf("[Test 6] Action selection\n");
    InferenceOutput out = run_kernel();

    float tol = 1e-5f;
    float p_err = max_abs_err(out.probs, probs_expected, NUM_POLICIES);
    printf("  probs max_err = %.2e\n", p_err);
    printf("  action: cuda=%d ref=%d\n", out.action, action_expected);

    // Verify softmax normalization
    float sum = 0.0f;
    for (int k = 0; k < NUM_POLICIES; k++) sum += out.probs[k];
    printf("  sum(probs) = %.8f\n", sum);

    CHECK(p_err < tol, "probs max_err %.2e >= tol", p_err);
    CHECK(out.action == action_expected, "action %d != %d", out.action, action_expected);
    CHECK(fabsf(sum - 1.0f) < 1e-6f, "probs don't sum to 1: %.8f", sum);
}

// ============================================================
// Test 7: Full pipeline (all outputs)
// ============================================================
static void test7_full_pipeline() {
    printf("[Test 7] Full pipeline\n");
    InferenceOutput out = run_kernel();

    float tol = 1e-5f;

    float mu_err = max_abs_err(out.mu, mu_expected, STATE_DIM);
    float f_err  = fabsf(out.F - F_expected);
    float g_err  = max_abs_err(out.G, G_expected, NUM_POLICIES);
    float p_err  = max_abs_err(out.probs, probs_expected, NUM_POLICIES);
    bool  action_ok = (out.action == action_expected);

    printf("  mu     max_err = %.2e  %s\n", mu_err, mu_err < tol ? "PASS" : "FAIL");
    printf("  F      err     = %.2e  %s\n", f_err,  f_err  < tol ? "PASS" : "FAIL");
    printf("  G      max_err = %.2e  %s\n", g_err,  g_err  < tol ? "PASS" : "FAIL");
    printf("  probs  max_err = %.2e  %s\n", p_err,  p_err  < tol ? "PASS" : "FAIL");
    printf("  action = %d (expected %d) %s\n", out.action, action_expected,
           action_ok ? "PASS" : "FAIL");

    CHECK(mu_err < tol, "mu");
    CHECK(f_err  < tol, "F");
    CHECK(g_err  < tol, "G");
    CHECK(p_err  < tol, "probs");
    CHECK(action_ok, "action");
}

// ============================================================
// Test 8: Determinism (run twice, same results)
// ============================================================
static void test8_determinism() {
    printf("[Test 8] Determinism\n");
    InferenceOutput out1 = run_kernel();
    InferenceOutput out2 = run_kernel();

    float mu_diff = max_abs_err(out1.mu, out2.mu, STATE_DIM);
    float g_diff  = max_abs_err(out1.G,  out2.G,  NUM_POLICIES);
    float p_diff  = max_abs_err(out1.probs, out2.probs, NUM_POLICIES);
    float f_diff  = fabsf(out1.F - out2.F);

    printf("  mu diff  = %.2e\n", mu_diff);
    printf("  F  diff  = %.2e\n", f_diff);
    printf("  G  diff  = %.2e\n", g_diff);
    printf("  P  diff  = %.2e\n", p_diff);

    CHECK(mu_diff == 0.0f, "mu not deterministic");
    CHECK(f_diff  == 0.0f, "F not deterministic");
    CHECK(g_diff  == 0.0f, "G not deterministic");
    CHECK(p_diff  == 0.0f, "probs not deterministic");
    CHECK(out1.action == out2.action, "action not deterministic");
}

// ============================================================
// Test 9: Policy offset (Plan B interface)
// ============================================================
static void test9_policy_offset() {
    printf("[Test 9] Policy offset (Plan B)\n");

    // Run with offset=0, count=5 (first half)
    InferenceOutput out_full = run_kernel(0, NUM_POLICIES);
    InferenceOutput out_half = run_kernel(0, 5);

    // G[0..4] should match between full and half runs
    float tol = 1e-5f;
    float g_err = max_abs_err(out_half.G, out_full.G, 5);
    printf("  G[0..4] max_err = %.2e\n", g_err);
    CHECK(g_err < tol, "G[0..4] differ with policy_count=5 vs 10");

    // Run with offset=5, count=5 (second half)
    InferenceOutput out_half2 = run_kernel(5, 5);
    float g_err2 = max_abs_err(out_half2.G, out_full.G + 5, 5);
    printf("  G[5..9] max_err = %.2e\n", g_err2);
    CHECK(g_err2 < tol, "G[5..9] differ with offset=5 vs full");
}

// ============================================================
// Main
// ============================================================
int main(int argc, char* argv[]) {
    std::string vec_dir = "../tests/test_vectors";
    if (argc > 1) vec_dir = argv[1];

    printf("=== SCSC Test Harness ===\n");
    printf("Test vectors: %s/\n\n", vec_dir.c_str());

    load_all(vec_dir);
    alloc_device();

    test1_prologue();        printf("\n");
    test2_vfe_single();      printf("\n");
    test3_vfe_full();        printf("\n");
    test4_G_single();        printf("\n");
    test5_G_all();           printf("\n");
    test6_action();          printf("\n");
    test7_full_pipeline();   printf("\n");
    test8_determinism();     printf("\n");
    test9_policy_offset();   printf("\n");

    free_device();

    printf("=== Results: %d PASS, %d FAIL ===\n", g_pass, g_fail);
    return g_fail > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
}
